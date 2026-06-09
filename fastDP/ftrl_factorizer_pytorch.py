import os
import time
from typing import Callable, Generic, Optional, Tuple, TypeVar

import torch
from torch.optim import LBFGS
from torch.nn.modules.loss import _Loss

import numpy as np
from tqdm.autonotebook import tqdm

from pfl.exception import MatrixFactorizationError
from pfl.internal.bridge.factory import FrameworkBridgeFactory as bridges
from pfl.internal.ops import get_ops
from pfl.internal.platform.selector import get_platform
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.stats import TrainingStatistics

from pfl.internal.ops.selector import set_framework_module
from pfl.internal.ops import pytorch_ops

os.environ['PFL_PYTORCH_DEVICE'] = 'cuda'
set_framework_module(pytorch_ops)

Tensor = TypeVar('Tensor')


class MatrixFactorizationError(Exception):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss


class TELoss(_Loss):
    """
    Total error in Sec. 4 https://arxiv.org/abs/2306.08153
    """
    def __init__(self,
                 workload_matrix: torch.Tensor,
                 mask: torch.Tensor):
        super().__init__()
        self.workload_matrix = workload_matrix
        self.mask = mask

    @staticmethod
    def _solve_positive_definite(A: torch.Tensor,
                                 B: torch.Tensor) -> torch.Tensor:
        """
        Solve X for AX = B where A is a positive definite matrix.
        """
        C = torch.linalg.cholesky(A.detach())
        return torch.cholesky_solve(B, C)

    def _obj(self, X: torch.Tensor):
        """
        Objective function.
        """
        H = self._solve_positive_definite(X, self.workload_matrix.T)
        loss = torch.trace(H @ self.workload_matrix)
        return loss

    @torch.no_grad()
    def _project_update(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Project dX so that:
        (1) The diagonal of X will be unchanged by setting diagonal of dX
        to 0 to ensure the sensitivity of the mechanism.
        (2) dX[i,j] is set to zero if mask[i,j] = 0.
        """
        dX.fill_diagonal_(0)
        return dX * self.mask

    @torch.no_grad()
    def _derivative(self, X: torch.Tensor):
        """Calculate derivatives."""
        H = self._solve_positive_definite(X, self.workload_matrix.T)
        gradients = self._project_update(-H @ H.T)
        return gradients

    def forward(self, X: torch.Tensor):
        n = X.shape[0]
        mat_eye = torch.eye(n, device=X.device, dtype=X.dtype)
        try:
            H = self._solve_positive_definite(X, self.workload_matrix.T)
            loss = torch.trace(H @ self.workload_matrix)
            gradients = self._project_update(-H @ H.T)
            return loss, gradients
        except Exception:
            pcost = torch.max(torch.diag(X))
            X = X / pcost
            cov = torch.tril(X)
            X = cov + cov.T + mat_eye * 2.0
            X_inv = torch.linalg.inv(X)
            loss = 100 * torch.trace(self.workload_matrix.T @ self.workload_matrix @ X_inv)
            raise MatrixFactorizationError(loss)


class FTRLMatrixFactorizer(Generic[Tensor]):
    """
    Class for factorizing matrices for matrix mechanism based on solving the
    optimization problem in Equation 6 in https://arxiv.org/pdf/2306.08153.pdf.

    :param workload_matrix:
        The input workload, n x n lower triangular matrix.
    :param mask:
        A boolean matrix describing the constraints on the gram matrix
        X = C^T C.
    """

    def __init__(self,
                 workload_matrix: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 X_init: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones_like(workload_matrix, dtype=workload_matrix.dtype)
        self._n = workload_matrix.shape[1]
        self._A = get_ops().to_tensor(workload_matrix, dtype='float64')
        # Mask determine which entries of X are allowed to be non-zero.
        self._mask = get_ops().to_tensor(mask, dtype='float64')

        if X_init is None:
            self._X_init = torch.eye(self._n, dtype=torch.float64, requires_grad=True, device='cuda')
        else:
            self._X_init = get_ops().to_tensor(X_init).double().requires_grad_()

    def optimize(self, iters: int = 30) -> Tensor:
        """
        Optimize the strategy matrix with an iterative gradient-based method.
        """
        X = self._X_init

        def _lbfgs_closure():
            lbfgs_optimizer.zero_grad()
            try:
                loss, gradients = loss_fn(X)
                X.grad = gradients.clone()
            except MatrixFactorizationError as e:
                loss = e.loss
                loss.backward()
                X.grad = torch.zeros_like(X.grad)
            return loss

        loss_fn = TELoss(mask=self._mask, workload_matrix=self._A)
        lbfgs_optimizer = LBFGS([X],
                                lr=0.25,
                                max_iter=6000,
                                history_size=60,
                                # tolerance_grad=1e-5,
                                line_search_fn="strong_wolfe"
                                )

        for _ in tqdm(range(iters), desc="Iteration", unit="iter", leave=False):
            lbfgs_optimizer.step(_lbfgs_closure)

        return X


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    num_iterations = 2000
    min_separation = 64

    def get_banded_matrix_mask_ex(n: int, bandwidth: int) -> np.ndarray:
        # Create a 2D array where each element is its row index
        row_indices = np.arange(n)[:, None]

        # Create a 2D array where each element is its column index
        col_indices = np.arange(n)

        # Compute the absolute difference between each pair of row and column indices
        abs_diff = np.abs(row_indices - col_indices)

        # Use the absolute difference to determine if an element should be 1 (within bandwidth) or 0 (outside bandwidth)
        mask = (abs_diff < bandwidth).astype(int)
        return mask

    banded_mask = get_banded_matrix_mask_ex(num_iterations, min_separation-1)
    workload_matrix = np.tril(np.ones(num_iterations, dtype=np.float64))
    ftrl_solver = FTRLMatrixFactorizer(workload_matrix=workload_matrix, mask=None)
    X = get_ops().to_numpy(ftrl_solver.optimize())
    print("X: ", X)

