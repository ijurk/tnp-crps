from __future__ import annotations

import math
import random
from typing import Iterable, Optional, Tuple, Union

import gpytorch
import torch

from tnp.data.base import GroundTruthPredictor
from tnp.data.synthetic import SyntheticGeneratorUniformInput
from tnp.networks.gp import RandomHyperparameterKernel


def _kernel_to_dense(kernel_output):
    if hasattr(kernel_output, "to_dense"):
        return kernel_output.to_dense()
    return kernel_output.evaluate()


def _is_infinite_df(df: Optional[float]) -> bool:
    return df is None or math.isinf(float(df))


class StudentTProcessGroundTruthPredictor(GroundTruthPredictor):
    """Ground-truth predictor for a Student-t process over observed outputs.

    y = s z
    z ~ N(0, K + sigma^2 I)
    s = sqrt(df / chi2_df)

    df=None or df=inf gives the Gaussian-process limit.
    """

    def __init__(
        self,
        *,
        kernel: gpytorch.kernels.Kernel,
        noise_std: float,
        df: Optional[float],
        jitter: float = 1e-5,
    ):
        self.kernel = kernel
        self.noise_std = float(noise_std)
        self.df = None if df is None else float(df)
        self.jitter = float(jitter)
        self._result_cache = None

    @property
    def is_gaussian_limit(self) -> bool:
        return _is_infinite_df(self.df)

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

    def _observation_covariance(self, x: torch.Tensor) -> torch.Tensor:
        cov = self._covariance(x)
        n = cov.shape[-1]
        eye = torch.eye(n, device=x.device, dtype=x.dtype)
        return cov + (self.noise_std**2 + self.jitter) * eye

    def sample_outputs(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        if sample_shape != torch.Size():
            raise NotImplementedError(
                "StudentTProcessGroundTruthPredictor.sample_outputs currently "
                "supports only empty sample_shape."
            )

        squeezed = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeezed = True

        if x.ndim != 3:
            raise ValueError(f"Expected x [B, N, D] or [N, D]. Got {x.shape}.")

        batch_size, n, _ = x.shape

        cov = self._observation_covariance(x)
        chol = torch.linalg.cholesky(cov)

        eps = torch.randn(batch_size, n, 1, device=x.device, dtype=x.dtype)
        y = chol @ eps

        if not self.is_gaussian_limit:
            chi2 = torch.distributions.Chi2(
                torch.tensor(self.df, device=x.device, dtype=x.dtype)
            ).sample((batch_size,))
            scale = torch.sqrt(self.df / chi2).view(batch_size, 1, 1)
            y = scale * y

        if squeezed:
            y = y.squeeze(0)

        return y

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ):
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

        for i, (xc_i, yc_i, xt_i) in enumerate(zip(xc, yc, xt)):
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
            cond_scale_matrix = ktt - kct.transpose(-1, -2) @ solve_kct
            cond_scale_matrix = 0.5 * (
                cond_scale_matrix + cond_scale_matrix.transpose(-1, -2)
            )

            diag = cond_scale_matrix.diagonal().clamp_min(self.jitter)

            if self.is_gaussian_limit:
                std = diag.sqrt()

                if yt is not None:
                    dist = torch.distributions.Normal(loc=mean, scale=std)
                    gt_loglik = dist.log_prob(yt[i, ..., 0])
                    gt_loglik_list.append(gt_loglik)

            else:
                beta = y_c @ alpha
                df_post = self.df + nc

                scale_factor = (self.df + beta) / (self.df + nc)
                t_scale_diag = (scale_factor * diag).clamp_min(self.jitter)
                t_scale = t_scale_diag.sqrt()

                if df_post > 2.0:
                    std = torch.sqrt(t_scale_diag * df_post / (df_post - 2.0))
                else:
                    std = torch.full_like(t_scale, float("nan"))

                if yt is not None:
                    dist = torch.distributions.StudentT(
                        df=torch.tensor(df_post, device=xt_i.device, dtype=xt_i.dtype),
                        loc=mean,
                        scale=t_scale,
                    )
                    gt_loglik = dist.log_prob(yt[i, ..., 0])
                    gt_loglik_list.append(gt_loglik)

            mean_list.append(mean)
            std_list.append(std)

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


class StudentTProcessGeneratorBase:
    def __init__(
        self,
        *,
        kernel: Union[
            RandomHyperparameterKernel,
            Tuple[RandomHyperparameterKernel, ...],
        ],
        noise_std: float,
        df: Optional[float],
        jitter: float = 1e-5,
        return_gt_pred: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel = kernel
        if isinstance(self.kernel, Iterable):
            self.kernel = tuple(self.kernel)

        self.noise_std = float(noise_std)
        self.df = None if df is None else float(df)
        self.jitter = float(jitter)
        self.return_gt_pred = bool(return_gt_pred)

    def set_up_student_t_process(self) -> StudentTProcessGroundTruthPredictor:
        if isinstance(self.kernel, tuple):
            kernel = random.choice(self.kernel)
        else:
            kernel = self.kernel

        kernel = kernel()
        kernel.sample_hyperparameters()
        kernel.eval()

        # Important: generated batches must not carry autograd graphs.
        # Otherwise DataLoader workers can crash when serialising tensors.
        for param in kernel.parameters():
            param.requires_grad_(False)

        return StudentTProcessGroundTruthPredictor(
            kernel=kernel,
            noise_std=self.noise_std,
            df=self.df,
            jitter=self.jitter,
        )

    def sample_outputs(
        self,
        x: torch.Tensor,
    ):
        gt_pred = self.set_up_student_t_process()

        with torch.no_grad():
            y = gt_pred.sample_outputs(x).detach()

        if self.return_gt_pred:
            return y, gt_pred

        return y, None


class RandomScaleStudentTProcessGenerator(
    StudentTProcessGeneratorBase,
    SyntheticGeneratorUniformInput,
):
    pass