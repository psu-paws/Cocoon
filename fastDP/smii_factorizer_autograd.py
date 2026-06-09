"""
    Test torch.optim.LBFGS
"""
import copy

import math
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


class SMIILoss(_Loss):
    def __init__(self,
                 bound: int = 1,
                 mask: torch.Tensor = None,
                 workload_matrix: Optional[torch.Tensor] = None,) -> None:
        super().__init__()
        assert mask is not None
        num_iterations = len(mask)
        device = mask.device

        # bound is a two-digit number,
        # first digit means the bound value,
        # second digit = 1 means we linearly grow the bound,
        # = 0 means linearly decay the bound
        # = 3 means sqrt grow the bound
        is_grow = bound % 10
        if is_grow == 1:
            is_grow_printf = 'linearly grow'
        elif is_grow == 0:
            is_grow_printf = 'linearly decay'
        elif is_grow == 3:
            is_grow_printf = 'sqrt grow'
        elif is_grow == 7:
            is_grow_printf = 'linearly grow with coefficient 0.7'
        elif is_grow == 5:
            is_grow_printf = 'linearly grow with coefficient 0.5'
        elif is_grow == 9:
            assert workload_matrix is not None
            is_grow_printf = 'grow with workload encoding'
        else:
            raise ValueError(f'Invalid bound value: {is_grow}')
        bound_value = bound // 10
        print(f'{is_grow_printf}, bound={bound_value}')

        if is_grow == 1:    # linearly grow
            self.bound = torch.linspace(1, bound_value, num_iterations, device=device)  # linearly grow the bound from start to bound
        elif is_grow == 0:   # linearly decay
            self.bound = torch.linspace(bound_value, 1, num_iterations, device=device)  # linearly decay the bound from start to 1
        elif is_grow == 3:   # sqrt grow
            self.bound = torch.sqrt(torch.linspace(1, bound_value, num_iterations, device=device))
        elif is_grow == 7:   # linearly grow with coefficient 0.7
            self.bound = 0.7 * torch.linspace(1, bound_value, num_iterations, device=device)
        elif is_grow == 5:   # linearly grow with coefficient 0.5
            self.bound = 0.5 * torch.linspace(1, bound_value, num_iterations, device=device)
        elif is_grow == 9:   # grow with workload encoding
            prefix_sum_inv = get_ops().to_tensor(np.linalg.inv(np.tril(np.ones(num_iterations))), dtype='float64')
            self.bound = torch.linspace(1, bound_value, num_iterations, device=device, dtype=torch.float64)
            self.bound = workload_matrix @ prefix_sum_inv @ self.bound
        else:
            raise ValueError(f'Invalid bound value: {is_grow}')

        self.mask = mask

    @staticmethod
    def _solve_positive_definite(A: torch.Tensor,
                                 B: torch.Tensor) -> torch.Tensor:
        """
        Solve X for AX = B where A is a positive definite matrix.
        """
        C = torch.linalg.cholesky(A.detach())
        return torch.cholesky_solve(B, C)

    @staticmethod
    def _func_var(mat_index, inv_X, var_bound):
        """
        Inequality constraint function for variance.

        Parameters
        ----------
        var_bound : variance bound
        mat_index : the index matrix
        cov : the co-variance matrix
        """

        vec_d = ((mat_index @ inv_X) * mat_index).sum(axis=1)
        return vec_d / var_bound

    @staticmethod
    def _func_pcost(mat_basis, X):
        """
        Inequality constraint function for privacy cost.

        Parameters
        ----------
        mat_basis : the basis matrix
        inv_cov : the inverse of the co-variance matrix X
        """
        vec_d = torch.diag(X)
        return vec_d

    @staticmethod
    def _obj(param_t, f_var, f_pcost):
        """
        Objective function.

        Parameters
        ----------
        X : co-variance matrix
        param_t : privacy cost approximation parameter
        """
        const_k = param_t * torch.max(f_var)
        const_t = param_t * torch.max(f_pcost)
        log_sum_k = torch.log(torch.sum(torch.exp(param_t * f_var - const_k)))
        log_sum_t = torch.log(torch.sum(torch.exp(param_t * f_pcost - const_t)))
        f_obj = const_t + log_sum_t + const_k + log_sum_k
        return f_obj / param_t

    @torch.no_grad()
    def _project_update(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Project dX so that:
        (1) The diagonal of X will be unchanged by setting diagonal of dX
        to 0 to ensure the sensitivity of the mechanism.
        (2) dX[i,j] is set to zero if mask[i,j] = 0.
        """
        # dX.fill_diagonal_(0)
        return dX * self.mask

    @staticmethod
    @torch.no_grad()
    def _derivative(param_t, f_var, f_pcost, mat_index, mat_basis, inv_X, var_bound):
        """Calculate derivatives."""
        param_k = param_t
        const_k = param_k * torch.max(f_pcost)
        exp_k = torch.exp(param_k * f_pcost - const_k)
        g_var = torch.diag(exp_k)

        const_t = torch.max(param_t * f_var)
        mat_ix = inv_X @ mat_index.T
        exp_t = torch.exp(param_t * f_var - const_t)
        g_pcost = -(exp_t / var_bound * mat_ix) @ mat_ix.T

        coef_k = param_t / torch.sum(exp_k)
        coef_t = param_t / torch.sum(exp_t)
        grad_k = g_var * coef_k
        grad_t = g_pcost * coef_t

        grad = grad_k + grad_t
        return grad

    @staticmethod
    def is_pos_def(A):
        """
        Check positive definiteness.

        Return true if psd.
        """
        # first check symmetry
        if torch.allclose(A, A.T, 1e-5, 1e-8):
            # whether cholesky decomposition is successful
            try:
                torch.linalg.cholesky(A)
                return True
            except Exception:
                return False
        else:
            return False

    def forward(self,
                A: torch.Tensor,
                X: torch.Tensor,
                mask: torch.Tensor,
                param_t: float):
        n = X.shape[0]
        mat_eye = torch.eye(n, device=X.device, dtype=X.dtype)
        try:
            inv_X = self._solve_positive_definite(X, mat_eye)
            # Todo: simplify _func_var when mat_index is the identity matrix
            mat_index = A
            # Todo: simplify _func_pcost when mat_basis is the prefix_sum matrix
            mat_basis = mat_eye

            f_var = self._func_var(mat_index, inv_X, self.bound)
            f_pcost = self._func_pcost(mat_basis, X)
            loss = self._obj(param_t, f_var, f_pcost)

            gradients = self._derivative(param_t, f_var, f_pcost, mat_index, mat_basis, inv_X, self.bound)
            gradients = self._project_update(gradients)
            return loss, gradients
        except Exception:
            pcost = torch.max(torch.diag(X))
            X = X / pcost 
            cov = torch.tril(X)
            X = cov + cov.T + mat_eye * 2.0
            inv_X = torch.linalg.solve(X, mat_eye)
            # Todo: simplify _func_var when mat_index is the identity matrix
            mat_index = A
            # Todo: simplify _func_pcost when mat_basis is the prefix_sum matrix
            mat_basis = mat_eye

            f_var = self._func_var(mat_index, inv_X, self.bound)
            f_pcost = self._func_pcost(mat_basis, X)
            loss = 100 * self._obj(param_t, f_var, f_pcost)
            raise MatrixFactorizationError(loss)


class SMIIMatrixFactorizer(Generic[Tensor]):
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
                 bound: int = 1,
                 X_init: Optional[np.ndarray] = None):
        assert workload_matrix.shape == mask.shape
        if mask is None:
            mask = np.ones_like(workload_matrix, dtype=workload_matrix.dtype)
        self._n = workload_matrix.shape[1]
        self._A = get_ops().to_tensor(workload_matrix).double()
        # Mask determine which entries of X are allowed to be non-zero.
        self._mask = get_ops().to_tensor(mask)

        if X_init is None:
            self._X_init = torch.eye(self._n, dtype=torch.float64, requires_grad=True, device='cuda')
        else:
            self._X_init = get_ops().to_tensor(X_init).double().requires_grad_()

        self.param_t = 1.0

        self.bound = bound

    def optimize(self, iters: int = 30) -> Tensor:
        """
        Optimize the strategy matrix with an iterative gradient-based method.
        """
        X = self._X_init

        def _lbfgs_closure():
            lbfgs_optimizer.zero_grad()
            try:
                loss, gradients = loss_fn(self._A, X, self._mask, self.param_t)
                X.grad = gradients.clone()
            except MatrixFactorizationError as e:
                loss = e.loss
                loss.backward()
                X.grad = torch.zeros_like(X.grad)
            return loss

        loss_fn = SMIILoss(bound=self.bound, mask=self._mask, workload_matrix=self._A)
        lbfgs_optimizer = LBFGS([X],
                                lr=0.25,
                                max_iter=6000,
                                history_size=60,
                                # tolerance_grad=1e-5,
                                line_search_fn="strong_wolfe"
                                )

        for i in tqdm(range(iters), desc="Iteration", unit="iter", leave=False):
            lbfgs_optimizer.step(_lbfgs_closure)
            gap = 2 * self._n / self.param_t
            if gap < 1e-4:
                break
            else:
                self.param_t *= 2
                # print('update t: {0}'.format(self.param_t))
                # print(X)
        pcost = torch.max(torch.diag(X))
        X = X / pcost
        return X
