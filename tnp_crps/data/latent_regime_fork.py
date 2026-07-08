from __future__ import annotations

import math
import random
from typing import Iterable, Optional, Tuple, Union

import gpytorch
import torch

from tnp.data.base import GroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch, SyntheticGenerator
from tnp.networks.gp import RandomHyperparameterKernel


def _kernel_to_dense(kernel_output):
    if hasattr(kernel_output, "to_dense"):
        return kernel_output.to_dense()
    return kernel_output.evaluate()


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    """Smooth transition from 0 to 1 for t in [0, 1]."""
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smooth_square_wave(
    x: torch.Tensor,
    *,
    period: float,
    sharpness: float,
) -> torch.Tensor:
    """Smooth square-wave-like function in [-1, 1]."""
    return torch.tanh(sharpness * torch.sin(2.0 * math.pi * x / period))

REGIME_NAMES = (
    "up_shift",
    "down_shift",
    "long_step",
    "short_step",
)

class LatentRegimeForkGroundTruthPredictor(GroundTruthPredictor):
    """Oracle helper for the latent regime fork process.

    The data-generating process is:

        g ~ GP(0, k)
        z in {0, 1, 2, 3}
        f_z(x) = g(x) + psi(x; x0) h_z(x)
        y = f_z(x) + eps

    The context is sampled before the fork, where psi=0, so the branch is
    deliberately non-identifiable from context.
    """

    def __init__(
        self,
        *,
        kernel: gpytorch.kernels.Kernel,
        noise_std: float,
        fork_locations: torch.Tensor,
        delta: float,
        transition_width: float,
        long_period: float,
        short_period: float,
        step_sharpness: float,
        num_regimes: int = 4,
        jitter: float = 1e-5,
    ):
        if num_regimes != 4:
            raise ValueError("This first implementation expects num_regimes=4.")

        self.kernel = kernel
        self.noise_std = float(noise_std)
        self.fork_locations = fork_locations.detach()
        self.delta = float(delta)
        self.transition_width = float(transition_width)
        self.long_period = float(long_period)
        self.short_period = float(short_period)
        self.step_sharpness = float(step_sharpness)
        self.num_regimes = int(num_regimes)
        self.jitter = float(jitter)

        self._result_cache = None
        
        # Plot/diagnostic metadata. These do not affect training or evaluation.
        self.sampled_regimes: Optional[torch.Tensor] = None
        self.dense_ground_truth_x: Optional[torch.Tensor] = None
        self.dense_ground_truth_y: Optional[torch.Tensor] = None
        self.dense_ground_truth_label = "GT realised function"

    def _device(self, fallback: torch.device) -> torch.device:
        try:
            return self.kernel.device
        except AttributeError:
            return fallback

    def _covariance(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x2 is None:
            return _kernel_to_dense(self.kernel(x1))
        return _kernel_to_dense(self.kernel(x1, x2))

    def _stable_cholesky(
        self,
        cov: torch.Tensor,
        *,
        context: str,
        initial_jitter: Optional[float] = None,
        max_tries: int = 8,
    ) -> torch.Tensor:
        """Numerically stable Cholesky for GP covariance matrices.

        Dense plotting grids can make covariance matrices nearly singular,
        especially in float32. This helper symmetrises, factors in float64,
        and increases diagonal jitter if needed.
        """
        if cov.ndim < 2 or cov.shape[-1] != cov.shape[-2]:
            raise ValueError(
                f"{context}: expected square covariance matrix, got {cov.shape}."
            )

        if not torch.isfinite(cov).all():
            raise ValueError(f"{context}: covariance contains NaN or Inf.")

        work = 0.5 * (cov + cov.transpose(-1, -2))

        # Plotting/evaluation only: use float64 for stability if possible.
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
            f"Max cholesky_ex info={int(last_info.max().item()) if last_info is not None else 'unknown'}."
        )

    def gate(self, x: torch.Tensor, fork_locations: torch.Tensor) -> torch.Tensor:
        """Return psi(x; x0), shape broadcast-compatible with x[..., 0]."""
        x_scalar = x[..., 0]
        x0 = fork_locations.to(device=x.device, dtype=x.dtype).view(-1, 1)

        t = (x_scalar - x0) / self.transition_width
        return _smoothstep(t)

    def set_sampled_regimes(self, regimes: torch.Tensor) -> None:
        """Attach the realised latent regime ids for plotting/diagnostics."""
        self.sampled_regimes = regimes.detach().cpu()
        self._result_cache = None

    def regime_name(self, regime_id: int) -> str:
        regime_id = int(regime_id)
        if regime_id < 0 or regime_id >= len(REGIME_NAMES):
            return str(regime_id)
        return REGIME_NAMES[regime_id]

    def set_dense_ground_truth(
        self,
        x_plot: torch.Tensor,
        y_plot: torch.Tensor,
        *,
        label: str = "GT realised function",
    ) -> None:
        """Store a dense realised function for plotting.

        x_plot and y_plot should correspond to the same jointly sampled
        latent GP realisation as the plotted context/target observations.
        """
        self.dense_ground_truth_x = x_plot.detach().cpu()
        self.dense_ground_truth_y = y_plot.detach().cpu()
        self.dense_ground_truth_label = label

    def branch_offsets(
        self,
        x: torch.Tensor,
        fork_locations: torch.Tensor,
    ) -> torch.Tensor:
        """Return branch offsets for all regimes.

        Args:
            x: [B, N, 1]
            fork_locations: [B]

        Returns:
            offsets: [K, B, N, 1]
        """
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, N, 1]. Got {x.shape}.")

        x0 = fork_locations.to(device=x.device, dtype=x.dtype).view(-1, 1)
        x_rel = x[..., 0] - x0
        psi = self.gate(x, fork_locations)

        up = self.delta * torch.ones_like(x_rel)
        down = -self.delta * torch.ones_like(x_rel)
        long_step = self.delta * _smooth_square_wave(
            x_rel,
            period=self.long_period,
            sharpness=self.step_sharpness,
        )
        short_step = self.delta * _smooth_square_wave(
            x_rel,
            period=self.short_period,
            sharpness=self.step_sharpness,
        )

        offsets = torch.stack([up, down, long_step, short_step], dim=0)
        offsets = psi.unsqueeze(0) * offsets

        return offsets.unsqueeze(-1)

    def _offsets_for_regimes(
        self,
        *,
        x: torch.Tensor,
        fork_locations: torch.Tensor,
        regimes: torch.Tensor,
    ) -> torch.Tensor:
        """Return realised branch offsets for one regime per task.

        Args:
            x: [B, N, 1]
            fork_locations: [B]
            regimes: [B]

        Returns:
            offsets: [B, N, 1]
        """
        batch_size = x.shape[0]

        all_offsets = self.branch_offsets(
            x,
            fork_locations.to(device=x.device, dtype=x.dtype),
        )  # [K, B, N, 1]

        gather_idx = regimes.to(device=x.device).view(1, batch_size, 1, 1).expand(
            1,
            batch_size,
            x.shape[1],
            1,
        )

        return torch.gather(all_offsets, dim=0, index=gather_idx).squeeze(0)

    def _base_latent_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample noiseless base GP latent values g(x)."""
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
    
    def _base_observation_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample g(x)+eps from the base GP observation model."""
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, N, 1]. Got {x.shape}.")

        output_dtype = x.dtype

        cov = self._covariance(x)
        batch_size, n, _ = x.shape

        chol = self._stable_cholesky(
            cov,
            context="_base_observation_sample",
            initial_jitter=self.noise_std**2 + self.jitter,
        )

        eps = torch.randn(batch_size, n, 1, device=x.device, dtype=chol.dtype)

        return (chol @ eps).to(dtype=output_dtype)

    def sample_outputs(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
        regimes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample observed outputs y at x.

        Args:
            x: [B, N, 1]
            regimes: optional [B] regime ids in {0,1,2,3}

        Returns:
            y: [B, N, 1]
        """
        if sample_shape != torch.Size():
            raise NotImplementedError("sample_shape is not supported here.")

        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, N, 1]. Got {x.shape}.")

        batch_size = x.shape[0]

        if regimes is None:
            regimes = torch.randint(
                low=0,
                high=self.num_regimes,
                size=(batch_size,),
                device=x.device,
            )
        else:
            regimes = regimes.to(device=x.device)

        self.set_sampled_regimes(regimes)

        base = self._base_observation_sample(x)
        offsets = self._offsets_for_regimes(
            x=x,
            fork_locations=self.fork_locations.to(device=x.device, dtype=x.dtype),
            regimes=regimes,
        )

        return base + offsets

    def sample_joint_observations_and_latent_function(
        self,
        *,
        x_observed: torch.Tensor,
        x_plot: torch.Tensor,
        regimes: Optional[torch.Tensor] = None,
        store: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Jointly sample observed task values and dense noiseless truth.

        This is for function plots. It samples the original task locations and
        dense plotting grid from the same finite-dimensional GP draw, so the
        plotted dense curve is an exact realised latent function on x_plot.

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

        old_device = x_observed.device
        device = self._device(old_device)

        x_observed_device = x_observed.to(device=device)
        x_plot_device = x_plot.to(
            device=device,
            dtype=x_observed_device.dtype,
        )

        batch_size = x_observed_device.shape[0]

        if regimes is None:
            if self.sampled_regimes is not None and self.sampled_regimes.numel() >= batch_size:
                regimes = self.sampled_regimes[:batch_size]
            else:
                regimes = torch.randint(
                    low=0,
                    high=self.num_regimes,
                    size=(batch_size,),
                    device=device,
                )

        regimes = regimes.to(device=device)

        x_joint = torch.cat([x_observed_device, x_plot_device], dim=1)
        base_joint = self._base_latent_sample(x_joint)

        n_observed = x_observed_device.shape[1]
        base_observed = base_joint[:, :n_observed, :]
        base_plot = base_joint[:, n_observed:, :]

        fork_locations = self.fork_locations.to(
            device=device,
            dtype=x_observed_device.dtype,
        )

        offsets_observed = self._offsets_for_regimes(
            x=x_observed_device,
            fork_locations=fork_locations,
            regimes=regimes,
        )
        offsets_plot = self._offsets_for_regimes(
            x=x_plot_device,
            fork_locations=fork_locations,
            regimes=regimes,
        )

        f_observed = base_observed + offsets_observed
        f_plot = base_plot + offsets_plot

        if self.noise_std > 0.0:
            y_observed = f_observed + self.noise_std * torch.randn_like(f_observed)
        else:
            y_observed = f_observed

        y_observed = y_observed.to(old_device)
        f_plot = f_plot.to(old_device)

        if store:
            self.set_sampled_regimes(regimes.detach().cpu())
            self.set_dense_ground_truth(
                x_plot.to(old_device),
                f_plot,
                label="GT realised function",
            )
            self._result_cache = None

        return y_observed, f_plot
    
    def predictive_samples(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Oracle predictive samples from p(y_t | x_t, context).

        Returns:
            samples: [M, B, Nt, 1]
        """
        old_device = xc.device
        device = self._device(old_device)

        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        fork_locations = self.fork_locations.to(device)

        samples = []
        for i in range(num_samples):
            # Sample one latent regime per task.
            regimes = torch.randint(
                low=0,
                high=self.num_regimes,
                size=(xt.shape[0],),
                device=device,
            )

            base_sample = self._sample_base_posterior_observation(
                xc=xc,
                yc=yc,
                xt=xt,
            )

            offsets_all = self.branch_offsets(xt, fork_locations)
            gather_idx = regimes.view(1, xt.shape[0], 1, 1).expand(
                1,
                xt.shape[0],
                xt.shape[1],
                1,
            )
            offsets = torch.gather(offsets_all, dim=0, index=gather_idx).squeeze(0)

            samples.append(base_sample + offsets)

        return torch.stack(samples, dim=0).to(old_device)

    def _sample_base_posterior_observation(
        self,
        *,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Sample base GP observation y_t given context.

        Context branch offset is zero by construction, so yc is directly
        observation of the base function plus noise.
        """
        out = []

        for xc_i, yc_i, xt_i in zip(xc, yc, xt):
            nc = xc_i.shape[0]
            nt = xt_i.shape[0]

            kcc = self._covariance(xc_i)
            kct = self._covariance(xc_i, xt_i)
            ktt = self._covariance(xt_i)

            eye_c = torch.eye(nc, device=xc_i.device, dtype=xc_i.dtype)
            eye_t = torch.eye(nt, device=xt_i.device, dtype=xt_i.dtype)

            kcc = kcc + (self.noise_std**2 + self.jitter) * eye_c
            ktt = ktt + (self.noise_std**2 + self.jitter) * eye_t

            y_c = yc_i[..., 0]
            alpha = torch.linalg.solve(kcc, y_c)

            mean = kct.transpose(-1, -2) @ alpha
            solve_kct = torch.linalg.solve(kcc, kct)
            cov = ktt - kct.transpose(-1, -2) @ solve_kct
            cov = 0.5 * (cov + cov.transpose(-1, -2))
            cov = cov + self.jitter * eye_t

            chol = torch.linalg.cholesky(cov)
            eps = torch.randn(nt, 1, device=xt_i.device, dtype=xt_i.dtype)
            sample = mean[..., None] + chol @ eps

            out.append(sample)

        return torch.stack(out, dim=0)

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ):
        """Return mixture marginal mean/std and optional marginal log-likelihood.

        This preserves the existing old-style gt_pred contract:
            mean, std, gt_loglik

        Note: log-likelihood is per-target marginal mixture log-probability,
        not joint function log-likelihood.
        """
        old_device = xc.device
        device = self._device(old_device)

        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        if yt is not None:
            yt = yt.to(device)

        if yt is not None and self._result_cache is not None:
            mean = self._result_cache["mean"].to(old_device)
            std = self._result_cache["std"].to(old_device)
            gt_loglik = self._result_cache["gt_loglik"]
            if gt_loglik is not None:
                gt_loglik = gt_loglik.to(old_device)
            return mean, std, gt_loglik

        mean_list = []
        std_list = []
        gt_loglik_list = []

        fork_locations = self.fork_locations.to(device)

        for b, (xc_i, yc_i, xt_i) in enumerate(zip(xc, yc, xt)):
            nc = xc_i.shape[0]
            nt = xt_i.shape[0]

            kcc = self._covariance(xc_i)
            kct = self._covariance(xc_i, xt_i)
            ktt = self._covariance(xt_i)

            eye_c = torch.eye(nc, device=xc_i.device, dtype=xc_i.dtype)
            eye_t = torch.eye(nt, device=xt_i.device, dtype=xt_i.dtype)

            kcc = kcc + (self.noise_std**2 + self.jitter) * eye_c
            ktt = ktt + (self.noise_std**2 + self.jitter) * eye_t

            y_c = yc_i[..., 0]
            alpha = torch.linalg.solve(kcc, y_c)

            base_mean = kct.transpose(-1, -2) @ alpha
            solve_kct = torch.linalg.solve(kcc, kct)
            base_cov = ktt - kct.transpose(-1, -2) @ solve_kct
            base_cov = 0.5 * (base_cov + base_cov.transpose(-1, -2))
            base_var = base_cov.diagonal().clamp_min(self.jitter)

            offsets = self.branch_offsets(
                xt_i.unsqueeze(0),
                fork_locations[b : b + 1],
            )[:, 0, :, 0]  # [K, Nt]

            component_means = base_mean.unsqueeze(0) + offsets
            mixture_mean = component_means.mean(dim=0)

            # Var(Y) = E[var(Y|z)] + Var(E[Y|z])
            mixture_var = base_var + component_means.var(dim=0, unbiased=False)
            mixture_std = mixture_var.clamp_min(self.jitter).sqrt()

            mean_list.append(mixture_mean)
            std_list.append(mixture_std)

            if yt is not None:
                dist = torch.distributions.Normal(
                    loc=component_means,
                    scale=base_var.sqrt().unsqueeze(0).expand_as(component_means),
                )
                log_probs = dist.log_prob(yt[b, ..., 0].unsqueeze(0))
                gt_loglik = torch.logsumexp(log_probs, dim=0) - math.log(
                    self.num_regimes
                )
                gt_loglik_list.append(gt_loglik)

        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)
        gt_loglik = torch.stack(gt_loglik_list, dim=0) if gt_loglik_list else None

        if yt is not None:
            self._result_cache = {
                "mean": mean.detach(),
                "std": std.detach(),
                "gt_loglik": gt_loglik.detach() if gt_loglik is not None else None,
            }

        mean = mean.to(old_device)
        std = std.to(old_device)
        if gt_loglik is not None:
            gt_loglik = gt_loglik.to(old_device)

        return mean, std, gt_loglik


class LatentRegimeForkGenerator(SyntheticGenerator):
    """Hard multimodal synthetic generator with latent structural regimes."""

    def __init__(
        self,
        *,
        kernel: Union[
            RandomHyperparameterKernel,
            Tuple[RandomHyperparameterKernel, ...],
        ],
        noise_std: float = 0.1,
        delta: float = 1.5,
        transition_width: float = 0.4,
        long_period: float = 2.5,
        short_period: float = 0.8,
        step_sharpness: float = 6.0,
        fork_range: Tuple[float, float] = (-0.5, 0.75),
        domain_range: Tuple[float, float] = (-4.0, 4.0),
        context_window_1: Tuple[float, float] = (-3.5, -2.0),
        context_window_2: Tuple[float, float] = (-1.2, -0.2),
        target_post_prob: float = 0.8,
        target_pre_margin: float = 0.2,
        num_regimes: int = 4,
        jitter: float = 1e-5,
        return_gt_pred: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if self.dim != 1:
            raise ValueError("LatentRegimeForkGenerator currently supports dim=1 only.")
        if num_regimes != 4:
            raise ValueError("This first implementation expects num_regimes=4.")

        self.kernel = kernel
        if isinstance(self.kernel, Iterable):
            self.kernel = tuple(self.kernel)

        self.noise_std = float(noise_std)
        self.delta = float(delta)
        self.transition_width = float(transition_width)
        self.long_period = float(long_period)
        self.short_period = float(short_period)
        self.step_sharpness = float(step_sharpness)
        self.fork_range = tuple(float(v) for v in fork_range)
        self.domain_range = tuple(float(v) for v in domain_range)
        self.context_window_1 = tuple(float(v) for v in context_window_1)
        self.context_window_2 = tuple(float(v) for v in context_window_2)
        self.target_post_prob = float(target_post_prob)
        self.target_pre_margin = float(target_pre_margin)
        self.num_regimes = int(num_regimes)
        self.jitter = float(jitter)
        self.return_gt_pred = bool(return_gt_pred)

    def _sample_kernel(self) -> gpytorch.kernels.Kernel:
        if isinstance(self.kernel, tuple):
            kernel = random.choice(self.kernel)
        else:
            kernel = self.kernel

        kernel = kernel()
        kernel.sample_hyperparameters()
        kernel.eval()

        for param in kernel.parameters():
            param.requires_grad_(False)

        return kernel

    def _sample_fork_locations(self, batch_size: int) -> torch.Tensor:
        low, high = self.fork_range
        return torch.rand(batch_size) * (high - low) + low

    def _sample_uniform_relative_window(
        self,
        *,
        fork_locations: torch.Tensor,
        window: Tuple[float, float],
        num_points: int,
    ) -> torch.Tensor:
        """Sample x from [x0 + window[0], x0 + window[1]]."""
        batch_size = fork_locations.shape[0]
        low = fork_locations[:, None] + window[0]
        high = fork_locations[:, None] + window[1]

        domain_low, domain_high = self.domain_range
        low = low.clamp(min=domain_low, max=domain_high)
        high = high.clamp(min=domain_low, max=domain_high)

        u = torch.rand(batch_size, num_points)
        x = low + u * (high - low).clamp_min(1e-6)
        return x[..., None]

    def _sample_inputs_with_fork(
        self,
        *,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
        fork_locations: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = batch_shape[0]

        nc1 = int(nc) // 2
        nc2 = int(nc) - nc1

        xc1 = self._sample_uniform_relative_window(
            fork_locations=fork_locations,
            window=self.context_window_1,
            num_points=nc1,
        )
        xc2 = self._sample_uniform_relative_window(
            fork_locations=fork_locations,
            window=self.context_window_2,
            num_points=nc2,
        )
        xc = torch.cat([xc1, xc2], dim=1)

        # Targets are biased toward the post-fork future region, but include
        # some pre-/near-fork targets for diagnostics.
        nt_post = int(round(int(nt) * self.target_post_prob))
        nt_pre = int(nt) - nt_post

        domain_low, domain_high = self.domain_range

        post_low = fork_locations[:, None] + self.target_pre_margin
        post_high = torch.full_like(post_low, domain_high)

        u_post = torch.rand(batch_size, nt_post)
        xt_post = post_low + u_post * (post_high - post_low).clamp_min(1e-6)
        xt_post = xt_post.clamp(min=domain_low, max=domain_high)[..., None]

        pre_low = torch.full((batch_size, 1), domain_low)
        pre_high = fork_locations[:, None] + self.target_pre_margin

        u_pre = torch.rand(batch_size, nt_pre)
        xt_pre = pre_low + u_pre * (pre_high - pre_low).clamp_min(1e-6)
        xt_pre = xt_pre.clamp(min=domain_low, max=domain_high)[..., None]

        xt = torch.cat([xt_pre, xt_post], dim=1)

        # Shuffle context and target positions separately.
        xc_perm = torch.stack([torch.randperm(xc.shape[1]) for _ in range(batch_size)])
        xt_perm = torch.stack([torch.randperm(xt.shape[1]) for _ in range(batch_size)])

        xc = torch.gather(xc, dim=1, index=xc_perm[..., None].expand_as(xc))
        xt = torch.gather(xt, dim=1, index=xt_perm[..., None].expand_as(xt))

        return torch.cat([xc, xt], dim=1)

    def generate_batch(self) -> SyntheticBatch:
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nt = torch.randint(low=self.min_nt, high=self.max_nt + 1, size=())

        return self.sample_batch(
            nc=int(nc.item()),
            nt=int(nt.item()),
            batch_shape=torch.Size((self.batch_size,)),
        )

    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        batch_size = batch_shape[0]
        fork_locations = self._sample_fork_locations(batch_size=batch_size)

        x = self._sample_inputs_with_fork(
            nc=nc,
            nt=nt,
            batch_shape=batch_shape,
            fork_locations=fork_locations,
        )

        kernel = self._sample_kernel()
        gt_pred = LatentRegimeForkGroundTruthPredictor(
            kernel=kernel,
            noise_std=self.noise_std,
            fork_locations=fork_locations,
            delta=self.delta,
            transition_width=self.transition_width,
            long_period=self.long_period,
            short_period=self.short_period,
            step_sharpness=self.step_sharpness,
            num_regimes=self.num_regimes,
            jitter=self.jitter,
        )

        regimes = torch.randint(low=0, high=self.num_regimes, size=(batch_size,))
        gt_pred.set_sampled_regimes(regimes)

        with torch.no_grad():
            y = gt_pred.sample_outputs(x, regimes=regimes).detach()

        xc = x[:, :nc, :]
        yc = y[:, :nc, :]
        xt = x[:, nc:, :]
        yt = y[:, nc:, :]

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=gt_pred if self.return_gt_pred else None,
        )

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        if nt is None:
            raise ValueError("LatentRegimeForkGenerator requires nt.")
        fork_locations = self._sample_fork_locations(batch_size=batch_shape[0])
        return self._sample_inputs_with_fork(
            nc=nc,
            nt=nt,
            batch_shape=batch_shape,
            fork_locations=fork_locations,
        )

    def sample_outputs(self, x: torch.Tensor):
        raise NotImplementedError(
            "LatentRegimeForkGenerator overrides sample_batch because fork locations "
            "must be shared between input sampling and output generation."
        )