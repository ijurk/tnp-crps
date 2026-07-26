"""Disk-backed raw tabular task source."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import get_worker_info


def _load_shard(path: Path) -> dict[str, Any]:
    """Load a bank shard created by build_tabicl_bank.py."""
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    required = {
        "x",
        "y",
        "num_features",
        "seq_len",
        "max_features",
    }

    missing = required.difference(payload)

    if missing:
        raise RuntimeError(
            f"Shard {path} is missing keys: {sorted(missing)}."
        )

    x = payload["x"]
    y = payload["y"]
    num_features = payload["num_features"]

    if x.ndim != 3:
        raise RuntimeError(
            f"Shard x must have rank three, got {tuple(x.shape)}."
        )

    if y.ndim != 3 or y.shape[-1] != 1:
        raise RuntimeError(
            f"Shard y must have shape [T, N, 1], got {tuple(y.shape)}."
        )

    if num_features.ndim != 1:
        raise RuntimeError(
            "Shard num_features must have shape [T]."
        )

    if not (
        x.shape[0]
        == y.shape[0]
        == num_features.shape[0]
    ):
        raise RuntimeError(
            f"Inconsistent shard task counts in {path}."
        )

    return payload


class TabICLDiskTaskSource:
    """Read raw numerical TabICL tasks sequentially from shards."""

    def __init__(
        self,
        *,
        bank_dir: str,
        split: str,
        seed: int,
        shuffle: bool,
        cycle: bool = True,
    ) -> None:
        self.bank_dir = Path(bank_dir).expanduser().resolve()
        self.split = str(split)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.cycle = bool(cycle)

        self.split_dir = self.bank_dir / self.split
        self.shard_paths = sorted(
            self.split_dir.glob("shard_*.pt")
        )

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shard_*.pt files found in {self.split_dir}."
            )

        self._worker_id: Optional[int] = None
        self._rng: Optional[random.Random] = None
        self._shard_order: list[int] = []
        self._shard_cursor = 0

        self._current_path: Optional[Path] = None
        self._current_payload: Optional[dict[str, Any]] = None
        self._task_order: list[int] = []
        self._task_cursor = 0
        self._cycle_index = 0

    def _initialise_worker_state(self) -> None:
        worker_info = get_worker_info()
        worker_id = -1 if worker_info is None else worker_info.id

        if self._worker_id == worker_id:
            return

        self._worker_id = worker_id

        worker_seed = self.seed + 1_000_003 * (worker_id + 1)
        self._rng = random.Random(worker_seed)

        self._cycle_index = 0
        self._reset_shard_order()

        self._current_path = None
        self._current_payload = None
        self._task_order = []
        self._task_cursor = 0

    def _reset_shard_order(self) -> None:
        assert self._rng is not None

        self._shard_order = list(range(len(self.shard_paths)))

        if self.shuffle:
            self._rng.shuffle(self._shard_order)

        self._shard_cursor = 0

    def _load_next_shard(self) -> None:
        assert self._rng is not None

        if self._shard_cursor >= len(self._shard_order):
            if not self.cycle:
                raise RuntimeError(
                    f"Exhausted non-cycling bank split {self.split}."
                )

            self._cycle_index += 1
            self._reset_shard_order()

        shard_index = self._shard_order[self._shard_cursor]
        self._shard_cursor += 1

        path = self.shard_paths[shard_index]
        payload = _load_shard(path)

        num_tasks = int(payload["x"].shape[0])

        self._current_path = path
        self._current_payload = payload
        self._task_order = list(range(num_tasks))

        if self.shuffle:
            self._rng.shuffle(self._task_order)

        self._task_cursor = 0

    def sample_task(
        self,
        seq_len: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        self._initialise_worker_state()

        if (
            self._current_payload is None
            or self._task_cursor >= len(self._task_order)
        ):
            self._load_next_shard()

        assert self._current_payload is not None
        assert self._current_path is not None

        task_index = self._task_order[self._task_cursor]
        self._task_cursor += 1

        stored_seq_len = int(
            self._current_payload["seq_len"]
        )

        if int(seq_len) != stored_seq_len:
            raise ValueError(
                f"Requested seq_len={seq_len}, but bank contains "
                f"seq_len={stored_seq_len}."
            )

        num_features = int(
            self._current_payload["num_features"][task_index]
        )

        x = self._current_payload["x"][
            task_index,
            :,
            :num_features,
        ].clone()

        y = self._current_payload["y"][
            task_index,
        ].clone()

        metadata = {
            "source": "tabicl_graph_scm_raw_numeric_disk",
            "bank_dir": str(self.bank_dir),
            "split": self.split,
            "shard": self._current_path.name,
            "task_index": task_index,
            "active_num_features": num_features,
            "cycle_index": self._cycle_index,
        }

        return x, y, metadata