from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from tnp.data.base import GroundTruthPredictor
from tnp.data.synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorUniformInput,
)


REGIME_NAMES = (
    "lower",
    "upper",
)


class BinaryLatentForkGroundTruthPredictor(GroundTruthPredictor):
    """Ground-truth predictor for the binary latent fork acid test.

    Data-generating process:

        z ∈ {-1, +1}
        g ~ Matérn-5/2 GP
        f_z(x) = g(x) + z * delta * gate(x)
        y(x) = f_z(x) + eps

    The gate is exactly zero before the fork and exactly one after the
    transition interval. Therefore, if all context points are pre-fork, the
    context contains no information about z.
    """

    plot_posterior_summary = True

    def __init__(
        self,
        *,
        fork_locations: torch.Tensor,
        lengthscale: float,
        base_scale: float,
        delta: float,
        transition_width: float,
        noise_std: float,
        jitter: float = 1e-5,
    ):
        self.fork_locations = fork_locations.detach().cpu().reshape(-1)

        self.lengthscale = float(lengthscale)
        self.base_scale = float(base_scale)
        self.delta = float(delta)
        self.transition_width = float(transition_width)
        self.noise_std = float(noise_std)
        self.jitter = float(jitter)

        if self.lengthscale <= 0.0:
            raise ValueError(f"lengthscale must be positive, got {self.lengthscale}.")
        if self.base_scale < 0.0:
            raise ValueError(f"base_scale must be non-negative, got {self.base_scale}.")
        if self.transition_width < 0.0:
            raise ValueError(
                f"transition_width must be non-negative, got {self.transition_width}."
            )
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}.")

        self.num_regimes = 2

        # Compatibility with existing latent-fork plotting helpers.
        self.sampled_regimes: Optional[torch.Tensor] = None  # ids 0/1
        self.regime_z: Optional[torch.Tensor] = None  # values -1/+1

        # Numeric metadata for plotting/diagnostics.
        self.delta_value = torch.full_like(self.fork_locations, self.delta)
        self.transition_width_value = torch.full_like(
            self.fork_locations,
            self.transition_width,
        )
        self.base_scale_value = torch.full_like(self.fork_locations, self.base_scale)

        # Optional dense realised function for plotting.
        self.dense_ground_truth_x: Optional[torch.Tensor] = None
        self.dense_ground_truth_y: Optional[torch.Tensor] = None
        self.dense_ground_truth_label = "Realised latent function"

        self._result_cache = None

    def set_sampled_regimes(self, regimes: torch.Tensor) -> None:
        """Attach realised latent regime ids for plotting/diagnostics."""
        regimes_cpu = regimes.detach().cpu().long().reshape(-1)

        self.sampled_regimes = regimes_cpu
        self.regime_z = torch.where(
            regimes_cpu == 0,
            -torch.ones_like(regimes_cpu),
            torch.ones_like(regimes_cpu),
        )

        self._result_cache = None

    def regime_name(self, regime_id: int) -> str:
        regime_id = int(regime_id)
        if regime_id < 0 or regime_id >= len(REGIME_NAMES):
            return str(regime_id)
        return REGIME_NAMES[regime_id]

    def regime_z_from_id(self, regime_id: int) -> int:
        return -1 if int(regime_id) == 0 else 1

    # Compatibility alias for notebooks/plotting.
    def regime_z_value(self, regime_id: int) -> int:
        return self.regime_z_from_id(regime_id)

    def set_dense_ground_truth(
        self,
        x_plot: torch.Tensor,
        y_plot: torch.Tensor,
        *,
        label: str = "Realised latent function",
    ) -> None:
        """Store exact dense realised latent function for plotting."""
        self.dense_ground_truth_x = x_plot.detach().cpu()
        self.dense_ground_truth_y = y_plot.detach().cpu()
        self.dense_ground_truth_label = label

    def _stable_cholesky(
        self,
        cov: torch.Tensor,
        *,
        context: str,
        initial_jitter: Optional[float] = None,
        max_tries: int = 8,
    ) -> torch.Tensor:
        """Stable Cholesky factorisation for nearly singular GP covariances."""
        if cov.ndim < 2 or cov.shape[-1] != cov.shape[-2]:
            raise ValueError(
                f"{context}: expected square covariance matrix, got {cov.shape}."
            )

        if not torch.isfinite(cov).all():
            raise ValueError(f"{context}: covariance contains NaN or Inf.")

        work = 0.5 * (cov + cov.transpose(-1, -2))

        # Dense plotting grids can make covariance matrices nearly singular.
        # Factorising in float64 is much more stable.
        if work.dtype in (torch.float16, torch.bfloat16, torch.float32):
            work = work.double()

        n = work.shape[-1]
        eye = torch.eye(n, device=work.device, dtype=work.dtype)

        jitter = float(self.jitter if initial_jitter is None else initial_jitter)
        jitter = max(jitter, 1e-8)

        last_info = None

        for _ in range(max_tries):
            chol, info = torch.linalg.cholesky_ex(work + jitter * eye)

            if int(info.max().item()) == 0:
                return chol

            last_info = info.detach().cpu()
            jitter *= 10.0

        raise RuntimeError(
            f"{context}: Cholesky failed after {max_tries} attempts. "
            f"Final jitter={jitter:.3e}. "
            f"Max cholesky_ex info="
            f"{int(last_info.max().item()) if last_info is not None else 'unknown'}."
        )

    def _covariance(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Matérn-5/2 covariance with marginal std approximately base_scale.

        Supports x1 [N, 1] or [B, N, 1].
        If x2 is provided, supports matching leading batch dimensions.
        """
        if x2 is None:
            x2 = x1

        if x1.shape[-1] != 1 or x2.shape[-1] != 1:
            raise ValueError(
                "BinaryLatentForkGroundTruthPredictor currently supports dim_x=1. "
                f"Got x1={x1.shape}, x2={x2.shape}."
            )

        x1s = x1[..., 0]
        x2s = x2[..., 0]

        diff = x1s.unsqueeze(-1) - x2s.unsqueeze(-2)
        r = diff.abs() / self.lengthscale

        sqrt5_r = math.sqrt(5.0) * r

        return (self.base_scale**2) * (
            1.0 + sqrt5_r + (5.0 / 3.0) * r.square()
        ) * torch.exp(-sqrt5_r)

    def _fork_locations_for_batch(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        fork = self.fork_locations.to(device=device, dtype=dtype).reshape(-1)
        stored = int(fork.shape[0])

        if stored == batch_size:
            return fork

        if stored == 1:
            return fork.expand(batch_size)

        if batch_size <= stored:
            # Plotters often slice tensors to B=1 while keeping the original
            # gt_pred object. Use the corresponding leading tasks.
            return fork[:batch_size]

        raise ValueError(
            "Stored binary fork batch size does not match requested batch size. "
            f"Stored={stored}, requested={batch_size}."
        )

    def _regimes_for_batch(
        self,
        *,
        regimes: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if regimes is None:
            if self.sampled_regimes is not None:
                regimes = self.sampled_regimes
            else:
                return torch.randint(
                    low=0,
                    high=2,
                    size=(batch_size,),
                    device=device,
                )

        regimes = regimes.to(device=device).long().reshape(-1)
        stored = int(regimes.shape[0])

        if stored == batch_size:
            return regimes

        if stored == 1:
            return regimes.expand(batch_size)

        if batch_size <= stored:
            return regimes[:batch_size]

        raise ValueError(
            "Stored binary fork regimes batch size does not match requested batch size. "
            f"Stored={stored}, requested={batch_size}."
        )

    def gate(
        self,
        x: torch.Tensor,
        fork_locations: torch.Tensor,
    ) -> torch.Tensor:
        """Exact-zero smoothstep gate.

        t = (x - x0) / transition_width

        gate = 0                 if t <= 0
        gate = 3t^2 - 2t^3        if 0 < t < 1
        gate = 1                 if t >= 1

        Args:
            x: [B, N, 1]
            fork_locations: [B]

        Returns:
            gate: [B, N, 1]
        """
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, N, 1], got {x.shape}.")

        fork = fork_locations.to(device=x.device, dtype=x.dtype).reshape(-1)

        if fork.shape[0] != x.shape[0]:
            raise ValueError(
                f"fork_locations batch size {fork.shape[0]} does not match "
                f"x batch size {x.shape[0]}."
            )

        raw_t = x[..., 0] - fork[:, None]

        if self.transition_width <= 0.0:
            return (raw_t >= 0.0).to(dtype=x.dtype).unsqueeze(-1)

        t = raw_t / self.transition_width
        clipped = t.clamp(0.0, 1.0)
        smooth = clipped.square() * (3.0 - 2.0 * clipped)

        gate = torch.where(
            t <= 0.0,
            torch.zeros_like(smooth),
            torch.where(t >= 1.0, torch.ones_like(smooth), smooth),
        )

        return gate.unsqueeze(-1)

    def offsets_for_regimes(
        self,
        *,
        x: torch.Tensor,
        fork_locations: torch.Tensor,
        regimes: torch.Tensor,
    ) -> torch.Tensor:
        """Return realised branch offsets for one global regime per task.

        regime 0 -> lower branch, z=-1
        regime 1 -> upper branch, z=+1

        Args:
            x: [B, N, 1]
            fork_locations: [B]
            regimes: [B]

        Returns:
            offsets: [B, N, 1]
        """
        batch_size = x.shape[0]

        regimes = regimes.to(device=x.device).long().reshape(-1)
        if regimes.shape[0] != batch_size:
            raise ValueError(
                f"regimes batch size {regimes.shape[0]} does not match "
                f"x batch size {batch_size}."
            )

        z = torch.where(
            regimes == 0,
            -torch.ones_like(regimes, dtype=x.dtype),
            torch.ones_like(regimes, dtype=x.dtype),
        ).view(batch_size, 1, 1)

        return z * self.delta * self.gate(x, fork_locations)

    def _base_latent_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample noiseless shared Matérn base GP values g(x)."""
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, N, 1]. Got {x.shape}.")

        output_dtype = x.dtype

        cov = self._covariance(x)
        batch_size, n, _ = x.shape

        chol = self._stable_cholesky(
            cov,
            context="_base_latent_sample",
            initial_jitter=self.jitter,
        )

        eps = torch.randn(
            batch_size,
            n,
            1,
            device=x.device,
            dtype=chol.dtype,
        )

        return (chol @ eps).to(dtype=output_dtype)

    def sample_outputs(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
        *,
        regimes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample noisy observations y(x) for the binary fork process."""
        if sample_shape != torch.Size():
            raise NotImplementedError(
                "BinaryLatentForkGroundTruthPredictor.sample_outputs currently "
                "supports only empty sample_shape."
            )

        squeezed = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeezed = True

        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, N, 1] or [N, 1]. Got {x.shape}.")

        batch_size = x.shape[0]

        fork_locations = self._fork_locations_for_batch(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )

        regimes = self._regimes_for_batch(
            regimes=regimes,
            batch_size=batch_size,
            device=x.device,
        )

        self.set_sampled_regimes(regimes)

        base = self._base_latent_sample(x)

        offsets = self.offsets_for_regimes(
            x=x,
            fork_locations=fork_locations,
            regimes=regimes,
        )

        f = base + offsets

        if self.noise_std > 0.0:
            y = f + self.noise_std * torch.randn_like(f)
        else:
            y = f

        if squeezed:
            y = y.squeeze(0)

        return y

    def sample_joint_observations_and_latent_function(
        self,
        *,
        x_observed: torch.Tensor,
        x_plot: torch.Tensor,
        regimes: Optional[torch.Tensor] = None,
        store: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Jointly sample task observations and dense noiseless latent truth.

        This is for plotting. It samples the original task locations and dense
        plotting grid from the same finite-dimensional GP draw, so the plotted
        dense curve is an exact realised latent function on x_plot.

        Args:
            x_observed: [B, Nobs, 1], original task inputs.
            x_plot: [B, Nplot, 1], dense plotting grid.
            regimes: optional [B] realised regime ids.
            store: whether to store dense ground truth on this object.

        Returns:
            y_observed: [B, Nobs, 1], noisy observations at original task inputs.
            f_plot: [B, Nplot, 1], noiseless realised latent function on x_plot.
        """
        if x_observed.ndim != 3 or x_observed.shape[-1] != 1:
            raise ValueError(f"Expected x_observed [B, N, 1]. Got {x_observed.shape}.")
        if x_plot.ndim != 3 or x_plot.shape[-1] != 1:
            raise ValueError(f"Expected x_plot [B, Nplot, 1]. Got {x_plot.shape}.")

        if x_plot.shape[0] != x_observed.shape[0]:
            if x_plot.shape[0] == 1:
                x_plot = x_plot.expand(x_observed.shape[0], -1, -1)
            else:
                raise ValueError(
                    "x_observed and x_plot must have matching batch size, "
                    f"or x_plot must have batch size 1. Got "
                    f"{x_observed.shape[0]} and {x_plot.shape[0]}."
                )

        device = x_observed.device
        dtype = x_observed.dtype
        batch_size = x_observed.shape[0]

        x_plot = x_plot.to(device=device, dtype=dtype)

        fork_locations = self._fork_locations_for_batch(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        regimes = self._regimes_for_batch(
            regimes=regimes,
            batch_size=batch_size,
            device=device,
        )

        x_joint = torch.cat([x_observed, x_plot], dim=1)
        base_joint = self._base_latent_sample(x_joint)

        n_observed = x_observed.shape[1]
        base_observed = base_joint[:, :n_observed, :]
        base_plot = base_joint[:, n_observed:, :]

        offsets_observed = self.offsets_for_regimes(
            x=x_observed,
            fork_locations=fork_locations,
            regimes=regimes,
        )
        offsets_plot = self.offsets_for_regimes(
            x=x_plot,
            fork_locations=fork_locations,
            regimes=regimes,
        )

        f_observed = base_observed + offsets_observed
        f_plot = base_plot + offsets_plot

        if self.noise_std > 0.0:
            y_observed = f_observed + self.noise_std * torch.randn_like(f_observed)
        else:
            y_observed = f_observed

        if store:
            self.set_sampled_regimes(regimes.detach().cpu())
            self.set_dense_ground_truth(
                x_plot.detach().cpu(),
                f_plot.detach().cpu(),
                label="Realised latent function",
            )
            self._result_cache = None

        return y_observed, f_plot

    def _mvn_log_prob_from_cholesky(
        self,
        y: torch.Tensor,
        chol: torch.Tensor,
    ) -> torch.Tensor:
        """Log N(y; 0, chol chol^T)."""
        y_col = y.reshape(-1, 1).to(dtype=chol.dtype)
        alpha = torch.cholesky_solve(y_col, chol)
        quad = (y_col * alpha).sum()
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        n = y.numel()

        return -0.5 * (n * math.log(2.0 * math.pi) + logdet + quad)

    def _conditional_for_regime(
        self,
        *,
        xc_i: torch.Tensor,
        yc_i: torch.Tensor,
        xt_i: torch.Tensor,
        fork_location_i: torch.Tensor,
        regime_id: int,
        include_target_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Component posterior for one task and one regime.

        Returns:
            mean_y: [Nt]
            var_y: [Nt]
            cov_y: [Nt, Nt]
            log_context_evidence: scalar
        """
        device = xc_i.device

        xc = xc_i.to(dtype=torch.float64)
        xt = xt_i.to(device=device, dtype=torch.float64)
        yc = yc_i.to(device=device, dtype=torch.float64).reshape(-1)

        fork = fork_location_i.to(device=device, dtype=torch.float64).reshape(1)
        regime = torch.tensor([int(regime_id)], device=device, dtype=torch.long)

        o_c = self.offsets_for_regimes(
            x=xc[None, :, :],
            fork_locations=fork,
            regimes=regime,
        )[0, :, 0]

        o_t = self.offsets_for_regimes(
            x=xt[None, :, :],
            fork_locations=fork,
            regimes=regime,
        )[0, :, 0]

        y_base = yc - o_c

        nc = xc.shape[0]
        nt = xt.shape[0]

        kcc = self._covariance(xc)
        kct = self._covariance(xc, xt)
        ktt = self._covariance(xt)

        eye_c = torch.eye(nc, device=device, dtype=torch.float64)
        eye_t = torch.eye(nt, device=device, dtype=torch.float64)

        kcc_obs = kcc + (self.noise_std**2 + self.jitter) * eye_c

        chol_c = self._stable_cholesky(
            kcc_obs,
            context="_conditional_for_regime/kcc",
            initial_jitter=self.jitter,
        )

        alpha = torch.cholesky_solve(y_base[:, None], chol_c).squeeze(-1)

        mean_base = kct.transpose(-1, -2) @ alpha

        solve_kct = torch.cholesky_solve(kct, chol_c)
        cov_base = ktt - kct.transpose(-1, -2) @ solve_kct
        cov_base = 0.5 * (cov_base + cov_base.transpose(-1, -2))

        if include_target_noise:
            cov_y = cov_base + (self.noise_std**2 + self.jitter) * eye_t
        else:
            cov_y = cov_base + self.jitter * eye_t

        cov_y = 0.5 * (cov_y + cov_y.transpose(-1, -2))

        mean_y = mean_base + o_t
        var_y = cov_y.diagonal().clamp_min(self.jitter)

        log_context_evidence = self._mvn_log_prob_from_cholesky(
            y_base,
            chol_c,
        )

        return mean_y, var_y, cov_y, log_context_evidence

    def posterior_marginal_components(
        self,
        *,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        include_target_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exact binary-mixture marginal components.

        Returns:
            component_means: [B, 2, Nt, 1], lower then upper.
            component_scales: [B, 2, Nt, 1].
            regime_weights: [B, 2].

        This path avoids constructing a full Nt x Nt covariance matrix and is
        therefore suitable for large marginal evaluations.
        """
        if xc.ndim != 3 or yc.ndim != 3 or xt.ndim != 3:
            raise ValueError(
                "Expected xc, yc and xt to be rank-3 tensors. "
                f"Got xc={xc.shape}, yc={yc.shape}, xt={xt.shape}."
            )
        if xc.shape[0] != yc.shape[0] or xc.shape[0] != xt.shape[0]:
            raise ValueError(
                "xc, yc and xt must have matching batch dimensions. "
                f"Got {xc.shape[0]}, {yc.shape[0]}, {xt.shape[0]}."
            )
        if xc.shape[-1] != 1 or xt.shape[-1] != 1 or yc.shape[-1] != 1:
            raise ValueError(
                "Binary latent fork marginal components require Dx=Dy=1."
            )

        old_device = xc.device
        old_dtype = xc.dtype
        batch_size = int(xc.shape[0])

        fork_locations = self._fork_locations_for_batch(
            batch_size=batch_size,
            device=old_device,
            dtype=old_dtype,
        )

        all_means = []
        all_scales = []
        all_weights = []

        for task_index in range(batch_size):
            task_means = []
            task_log_evidence = []

            xc_i = xc[task_index].to(dtype=torch.float64)
            xt_i = xt[task_index].to(dtype=torch.float64)
            yc_i = yc[task_index, :, 0].to(dtype=torch.float64)
            fork_i = fork_locations[task_index].to(dtype=torch.float64).reshape(1)
            num_context = int(xc_i.shape[0])

            # The base-GP covariance does not depend on the latent branch. Reuse
            # one context factorisation for both mixture components.
            kcc = self._covariance(xc_i)
            kct = self._covariance(xc_i, xt_i)
            eye_c = torch.eye(
                num_context,
                device=old_device,
                dtype=torch.float64,
            )
            kcc_obs = kcc + (self.noise_std**2 + self.jitter) * eye_c
            chol_c = self._stable_cholesky(
                kcc_obs,
                context="posterior_marginal_components/kcc",
                initial_jitter=self.jitter,
            )
            solve_kct = torch.cholesky_solve(kct, chol_c)

            # Matérn covariance at zero separation equals base_scale^2. The
            # conditional variance is identical under the two deterministic
            # branch offsets.
            var_base = self.base_scale**2 - (kct * solve_kct).sum(dim=0)
            if include_target_noise:
                var_base = var_base + self.noise_std**2
            var_base = (var_base + self.jitter).clamp_min(self.jitter)

            for regime_id in (0, 1):
                regime = torch.tensor(
                    [regime_id],
                    device=old_device,
                    dtype=torch.long,
                )

                context_offset = self.offsets_for_regimes(
                    x=xc_i[None, :, :],
                    fork_locations=fork_i,
                    regimes=regime,
                )[0, :, 0]
                target_offset = self.offsets_for_regimes(
                    x=xt_i[None, :, :],
                    fork_locations=fork_i,
                    regimes=regime,
                )[0, :, 0]

                centred_context = yc_i - context_offset
                alpha = torch.cholesky_solve(
                    centred_context[:, None],
                    chol_c,
                ).squeeze(-1)
                mean_base = kct.transpose(-1, -2) @ alpha

                task_means.append(mean_base + target_offset)
                task_log_evidence.append(
                    self._mvn_log_prob_from_cholesky(
                        centred_context,
                        chol_c,
                    )
                )

            means = torch.stack(task_means, dim=0)
            scales = var_base.sqrt().expand(2, -1).clone()
            log_weights = torch.stack(task_log_evidence, dim=0) - math.log(2.0)
            weights = torch.softmax(log_weights, dim=0)

            all_means.append(means)
            all_scales.append(scales)
            all_weights.append(weights)

        component_means = torch.stack(all_means, dim=0).unsqueeze(-1)
        component_scales = torch.stack(all_scales, dim=0).unsqueeze(-1)
        regime_weights = torch.stack(all_weights, dim=0)

        return (
            component_means.to(device=old_device, dtype=old_dtype),
            component_scales.to(device=old_device, dtype=old_dtype),
            regime_weights.to(device=old_device, dtype=old_dtype),
        )

    def predictive_marginal_samples(
        self,
        *,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample the exact univariate posterior marginals efficiently.

        One global regime is drawn per task and ensemble member, preserving the
        binary branch variable. Conditional residuals are sampled independently
        across targets because this method is intended only for marginal scoring.
        Use :meth:`predictive_samples` when joint GP path covariance is required.
        """
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}.")

        means, scales, weights = self.posterior_marginal_components(
            xc=xc,
            yc=yc,
            xt=xt,
            include_target_noise=True,
        )

        batch_size = int(xt.shape[0])
        upper_probability = weights[:, 1].reshape(1, batch_size, 1, 1)
        choose_upper = torch.rand(
            num_samples,
            batch_size,
            1,
            1,
            device=xt.device,
            dtype=xt.dtype,
        ) < upper_probability

        lower_mean = means[:, 0].unsqueeze(0)
        upper_mean = means[:, 1].unsqueeze(0)
        lower_scale = scales[:, 0].unsqueeze(0)
        upper_scale = scales[:, 1].unsqueeze(0)

        selected_mean = torch.where(choose_upper, upper_mean, lower_mean)
        selected_scale = torch.where(choose_upper, upper_scale, lower_scale)

        return selected_mean + selected_scale * torch.randn(
            num_samples,
            *xt.shape[:-1],
            1,
            device=xt.device,
            dtype=xt.dtype,
        )

    def sample_paired_regime_observations(
        self,
        *,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Sample paired lower/upper counterfactual observations.

        The two regimes share the same base-GP realisation and observation-noise
        draw, differing only through the deterministic branch offset. This makes
        upper- and lower-reveal intervention tasks exactly paired.

        Returns:
            Tensor [B, 2, N, 1], ordered as lower then upper.
        """
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, N, 1], got {x.shape}.")

        batch_size = int(x.shape[0])
        fork_locations = self._fork_locations_for_batch(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )

        base = self._base_latent_sample(x)
        shared_noise = (
            self.noise_std * torch.randn_like(base)
            if self.noise_std > 0.0
            else torch.zeros_like(base)
        )

        regimes = torch.arange(2, device=x.device, dtype=torch.long)
        outputs = []
        for regime_id in regimes.tolist():
            regime_tensor = torch.full(
                (batch_size,),
                int(regime_id),
                device=x.device,
                dtype=torch.long,
            )
            offsets = self.offsets_for_regimes(
                x=x,
                fork_locations=fork_locations,
                regimes=regime_tensor,
            )
            outputs.append(base + offsets + shared_noise)

        return torch.stack(outputs, dim=1)

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ):
        """Old-style gt_pred contract: mean, std, gt_loglik.

        The returned mean/std are oracle marginal posterior summaries under
        the binary mixture over regimes.
        """
        old_device = xc.device
        old_dtype = xc.dtype

        if xc.ndim != 3 or yc.ndim != 3 or xt.ndim != 3:
            raise ValueError(
                f"Expected xc/yc/xt rank 3. Got xc={xc.shape}, yc={yc.shape}, xt={xt.shape}."
            )

        batch_size = xc.shape[0]
        fork_locations = self._fork_locations_for_batch(
            batch_size=batch_size,
            device=old_device,
            dtype=old_dtype,
        )

        mean_list = []
        std_list = []
        gt_loglik_list = []

        for i in range(batch_size):
            comp_means = []
            comp_vars = []
            comp_log_evidence = []

            for regime_id in (0, 1):
                mean_y, var_y, _, log_evidence = self._conditional_for_regime(
                    xc_i=xc[i],
                    yc_i=yc[i, :, 0],
                    xt_i=xt[i],
                    fork_location_i=fork_locations[i],
                    regime_id=regime_id,
                    include_target_noise=True,
                )

                comp_means.append(mean_y)
                comp_vars.append(var_y)
                comp_log_evidence.append(log_evidence)

            comp_means_t = torch.stack(comp_means, dim=0)  # [2, Nt]
            comp_vars_t = torch.stack(comp_vars, dim=0)  # [2, Nt]

            logw = torch.stack(comp_log_evidence, dim=0)
            logw = logw - math.log(2.0)  # uniform prior over regimes
            weights = torch.softmax(logw, dim=0)  # [2]

            mean = (weights[:, None] * comp_means_t).sum(dim=0)

            second = (
                weights[:, None]
                * (comp_vars_t + comp_means_t.square())
            ).sum(dim=0)

            var = (second - mean.square()).clamp_min(self.jitter)
            std = var.sqrt()

            mean_list.append(mean.to(device=old_device, dtype=old_dtype))
            std_list.append(std.to(device=old_device, dtype=old_dtype))

            if yt is not None:
                y_target = yt[i, :, 0].to(device=old_device, dtype=old_dtype)

                comp_scales = comp_vars_t.sqrt().to(device=old_device, dtype=old_dtype)
                comp_locs = comp_means_t.to(device=old_device, dtype=old_dtype)
                weights_old = weights.to(device=old_device, dtype=old_dtype)

                log_probs = torch.distributions.Normal(
                    loc=comp_locs,
                    scale=comp_scales.clamp_min(math.sqrt(self.jitter)),
                ).log_prob(y_target[None, :])

                gt_loglik = torch.logsumexp(
                    weights_old[:, None].log() + log_probs,
                    dim=0,
                )
                gt_loglik_list.append(gt_loglik)

        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)

        gt_loglik = (
            torch.stack(gt_loglik_list, dim=0)
            if gt_loglik_list
            else None
        )

        return mean, std, gt_loglik

    def predictive_samples(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        num_samples: int = 128,
    ) -> torch.Tensor:
        """Draw coherent exact posterior paths efficiently.

        One regime is sampled per complete target path. The full conditional GP
        covariance is retained within that regime.
        """
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}.")

        old_device = xc.device
        old_dtype = xc.dtype
        batch_size = int(xc.shape[0])
        num_targets = int(xt.shape[1])

        fork_locations = self._fork_locations_for_batch(
            batch_size=batch_size,
            device=old_device,
            dtype=old_dtype,
        )

        output = torch.empty(
            num_samples,
            batch_size,
            num_targets,
            1,
            device=old_device,
            dtype=old_dtype,
        )

        for task_index in range(batch_size):
            means = []
            covariances = []
            log_evidence = []

            for regime_id in (0, 1):
                mean_y, _, cov_y, evidence = self._conditional_for_regime(
                    xc_i=xc[task_index],
                    yc_i=yc[task_index, :, 0],
                    xt_i=xt[task_index],
                    fork_location_i=fork_locations[task_index],
                    regime_id=regime_id,
                    include_target_noise=True,
                )
                means.append(mean_y)
                covariances.append(cov_y)
                log_evidence.append(evidence)

            weights = torch.softmax(
                torch.stack(log_evidence, dim=0) - math.log(2.0),
                dim=0,
            )
            chosen = torch.multinomial(
                weights,
                num_samples=num_samples,
                replacement=True,
            )

            task_samples = torch.empty(
                num_samples,
                num_targets,
                1,
                device=old_device,
                dtype=torch.float64,
            )

            for regime_id in (0, 1):
                selected = torch.nonzero(chosen == regime_id, as_tuple=False).reshape(-1)
                if selected.numel() == 0:
                    continue

                chol = self._stable_cholesky(
                    covariances[regime_id],
                    context="predictive_samples/cov_y",
                    initial_jitter=self.jitter,
                )
                epsilon = torch.randn(
                    int(selected.numel()),
                    num_targets,
                    1,
                    device=old_device,
                    dtype=chol.dtype,
                )
                mean = means[regime_id].reshape(1, num_targets, 1).to(chol.dtype)
                draws = mean + torch.matmul(chol.unsqueeze(0), epsilon)
                task_samples[selected] = draws

            output[:, task_index] = task_samples.to(dtype=old_dtype)

        return output


class BinaryLatentForkGeneratorBase(SyntheticGenerator):
    """Simple binary latent fork generator.

    This is a controlled acid test:
      - fixed fork location;
      - two regimes only;
      - one global regime per task;
      - pre-fork context should not reveal z.
    """

    def __init__(
        self,
        *,
        lengthscale: float = 1.0,
        base_scale: float = 0.5,
        delta: float = 2.0,
        fork_x0: float = 0.0,
        transition_width: float = 0.1,
        noise_std: float = 0.05,
        jitter: float = 1e-5,
        return_gt_pred: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if int(self.dim) != 1:
            raise ValueError(
                f"BinaryLatentForkGenerator currently supports dim=1 only. Got dim={self.dim}."
            )

        self.lengthscale = float(lengthscale)
        self.base_scale = float(base_scale)
        self.delta = float(delta)
        self.fork_x0 = float(fork_x0)
        self.transition_width = float(transition_width)
        self.noise_std = float(noise_std)
        self.jitter = float(jitter)
        self.return_gt_pred = bool(return_gt_pred)

    def set_up_binary_latent_fork(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BinaryLatentForkGroundTruthPredictor:
        fork_locations = torch.full(
            (batch_size,),
            self.fork_x0,
            device=device,
            dtype=dtype,
        )

        return BinaryLatentForkGroundTruthPredictor(
            fork_locations=fork_locations,
            lengthscale=self.lengthscale,
            base_scale=self.base_scale,
            delta=self.delta,
            transition_width=self.transition_width,
            noise_std=self.noise_std,
            jitter=self.jitter,
        )

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[BinaryLatentForkGroundTruthPredictor]]:
        """Sample binary fork outputs.

        Args:
            x: [B, N, 1]

        Returns:
            y: [B, N, 1]
            gt_pred: BinaryLatentForkGroundTruthPredictor or None
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, N, D], got {x.shape}.")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected final dim={self.dim}, got {x.shape[-1]}.")

        batch_size = x.shape[0]

        gt_pred = self.set_up_binary_latent_fork(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )

        regimes = torch.randint(
            low=0,
            high=2,
            size=(batch_size,),
            device=x.device,
        )

        gt_pred.set_sampled_regimes(regimes)

        with torch.no_grad():
            y = gt_pred.sample_outputs(x, regimes=regimes).detach()

        return y, gt_pred if self.return_gt_pred else None


class BinaryLatentForkGenerator(
    BinaryLatentForkGeneratorBase,
    SyntheticGeneratorUniformInput,
):
    pass


class BinaryLatentForkGeneratorMixedContext(BinaryLatentForkGenerator):
    """Mixture of ambiguous and regime-revealing context tasks.

    Ambiguous family, with probability 1 - p_revealing:
        Every context input is sampled from the ordinary pre-fork
        context_range. These are the original binary-fork acid-test tasks.

    Revealing family, with probability p_revealing:
        Context inputs are sampled across full_context_range. One randomly
        selected context input is then overwritten by a location at or beyond
        the completed fork transition, guaranteeing that the context reveals
        the global regime.

    Target sampling is unchanged.

    Validation and test should continue to use BinaryLatentForkGenerator with
    pre-fork-only contexts. This subclass is intended for training only.
    """

    def __init__(
        self,
        *,
        p_revealing: float = 0.5,
        full_context_range: Optional[
            Tuple[Tuple[float, float], ...]
        ] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        p_revealing = float(p_revealing)

        if not 0.0 <= p_revealing <= 1.0:
            raise ValueError(
                "p_revealing must be in [0, 1], "
                f"got {p_revealing}."
            )

        self.p_revealing = p_revealing

        if full_context_range is None:
            self.full_context_range = self.target_range.clone()
        else:
            # Convert ListConfig/list/tuple inputs robustly.
            range_values = [
                [float(value) for value in row]
                for row in full_context_range
            ]

            self.full_context_range = torch.tensor(
                range_values,
                dtype=self.context_range.dtype,
                device=self.context_range.device,
            )

        if self.full_context_range.shape != self.context_range.shape:
            raise ValueError(
                "full_context_range must match context_range shape. "
                f"Got {tuple(self.full_context_range.shape)} versus "
                f"{tuple(self.context_range.shape)}."
            )

        # The exact-zero smoothstep gate is fully active for:
        #
        #     x >= fork_x0 + transition_width
        #
        # A point sampled from this interval therefore contains full regime
        # information.
        self.reveal_min = (
            float(self.fork_x0)
            + float(self.transition_width)
        )
        self.reveal_max = float(self.full_context_range[0, 1])

        if not self.reveal_min < self.reveal_max:
            raise ValueError(
                "No interval is available for a revealing context point. "
                "Expected fork_x0 + transition_width to be below the "
                "full-context upper bound, but got "
                f"reveal_min={self.reveal_min} and "
                f"reveal_max={self.reveal_max}."
            )

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        nc = int(nc)

        if nc < 1:
            raise ValueError(
                "Mixed-context training requires nc >= 1, "
                f"got nc={nc}."
            )

        batch_shape_tuple = tuple(batch_shape)

        rand_kwargs = {
            "device": self.context_range.device,
            "dtype": self.context_range.dtype,
        }

        # Original ambiguous, pre-fork context family.
        xc_pre = (
            torch.rand(
                (*batch_shape_tuple, nc, self.dim),
                **rand_kwargs,
            )
            * (
                self.context_range[:, 1]
                - self.context_range[:, 0]
            )
            + self.context_range[:, 0]
        )

        # Broad context family, including the pre-fork, transition, and
        # post-fork regions.
        xc_full = (
            torch.rand(
                (*batch_shape_tuple, nc, self.dim),
                **rand_kwargs,
            )
            * (
                self.full_context_range[:, 1]
                - self.full_context_range[:, 0]
            )
            + self.full_context_range[:, 0]
        )

        # Draw one guaranteed fully post-transition context point per task.
        x_reveal = (
            torch.rand(
                (*batch_shape_tuple, 1, self.dim),
                **rand_kwargs,
            )
            * (self.reveal_max - self.reveal_min)
            + self.reveal_min
        )

        reveal_idx = torch.randint(
            low=0,
            high=nc,
            size=batch_shape_tuple,
            device=xc_full.device,
        )

        reveal_mask = torch.nn.functional.one_hot(
            reveal_idx,
            num_classes=nc,
        ).bool().unsqueeze(-1)

        # x_reveal broadcasts from [..., 1, dim] to [..., nc, dim].
        xc_revealing = torch.where(
            reveal_mask,
            x_reveal,
            xc_full,
        )

        # Choose the context family independently for every task.
        use_revealing = (
            torch.rand(
                (*batch_shape_tuple, 1, 1),
                **rand_kwargs,
            )
            < self.p_revealing
        )

        xc = torch.where(
            use_revealing,
            xc_revealing,
            xc_pre,
        )

        if nt is None:
            return xc

        xt = (
            torch.rand(
                (*batch_shape_tuple, int(nt), self.dim),
                **rand_kwargs,
            )
            * (
                self.target_range[:, 1]
                - self.target_range[:, 0]
            )
            + self.target_range[:, 0]
        )

        return torch.cat([xc, xt], dim=1)