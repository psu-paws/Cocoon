# Copyright © 2023-2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/federated:
# Copyright 2022, Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
"""
Primal optimization algorithms for multi-epoch matrix factorization.
Reference: https://github.com/google-research/federated/blob/master/multi_epoch_dp_matrix_factorization/multiple_participations/primal_optimization.py.  # pylint: disable=line-too-long
"""

from typing import Tuple

import torch
from opt_einsum import contract


from pfl.exception import MatrixFactorizationError

from ..base import FTRLFrameworkBridge


class PyTorchFTRLBridge(FTRLFrameworkBridge[torch.Tensor]):

    @staticmethod
    @torch.no_grad()
    def loss_and_gradient(
            A: torch.Tensor,
            X: torch.Tensor,
            mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        def _solve_positive_definite(A: torch.Tensor,
                                     B: torch.Tensor) -> torch.Tensor:
            """
            Solve X for AX = B where A is a positive definite matrix.
            """
            C = torch.linalg.cholesky(A.detach())
            return torch.cholesky_solve(B, C)

        def _project_update(dX: torch.Tensor) -> torch.Tensor:
            """
            Project dX so that:
            (1) The diagonal of X will be unchanged by setting diagonal of dX
            to 0 to ensure the sensitivity of the mechanism.
            (2) dX[i,j] is set to zero if mask[i,j] = 0.
            """
            dX.fill_diagonal_(0)
            return dX * mask

        try:
            H = _solve_positive_definite(X, A.T)
            loss, gradients = torch.trace(H @ A), _project_update(-H @ H.T)
            assert not torch.any(torch.isnan(loss)).item()
            assert not torch.any(torch.isnan(loss)).item()
        except Exception as e:
            raise MatrixFactorizationError from e
        else:
            return loss, gradients

    @staticmethod
    @torch.no_grad()
    def lbfgs_direction(X: torch.Tensor,
                        dX: torch.Tensor,
                        prev_X: torch.Tensor,
                        prev_dX: torch.Tensor) -> torch.Tensor:
        S = X - prev_X
        Y = dX - prev_dX
        rho = 1.0 / torch.sum(Y * S)
        alpha = rho * torch.sum(S * dX)
        gamma = torch.sum(S * Y) / torch.sum(Y ** 2)
        Z = gamma * (dX - rho * torch.sum(S * dX) * Y)
        beta = rho * torch.sum(Y * Z)
        Z = Z + S * (alpha - beta)
        return Z

    @staticmethod
    def terminate_fn(dX: torch.Tensor) -> bool:
        return torch.abs(dX).amax().item() <= 1e-3


class PyTorchSMIIBridge(FTRLFrameworkBridge[torch.Tensor]):

    @staticmethod
    @torch.no_grad()
    def loss_and_gradient(
            A: torch.Tensor,
            X: torch.Tensor,
            mask: torch.Tensor,
            param_t: torch.Tensor, bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        def _solve_positive_definite(A: torch.Tensor,
                                     B: torch.Tensor) -> torch.Tensor:
            """
            Solve X for AX = B where A is a positive definite matrix.
            """
            C = torch.linalg.cholesky(A.detach())
            return torch.cholesky_solve(B, C)

        def _project_update(dX: torch.Tensor) -> torch.Tensor:
            """
            Project dX so that:
            (1) The diagonal of X will be unchanged by setting diagonal of dX
            to 0 to ensure the sensitivity of the mechanism.
            (2) dX[i,j] is set to zero if mask[i,j] = 0.
            """
            dX.fill_diagonal_(0)
            return dX * mask

        def _func_var(mat_index, cov, var_bound):
            """
            Inequality constraint function for variance.

            Parameters
            ----------
            var_bound : variance bound
            mat_index : the index matrix
            cov : the co-variance matrix
            """
            # d = torch.diag(self.mat_index @ self.cov @ self.mat_index.T)
            # vec_d = ((mat_index @ cov) * mat_index).sum(axis=1)
            vec_d = torch.diag(cov)
            return vec_d / var_bound

        def _func_pcost(mat_basis, inv_cov):
            """
            Inequality constraint function for privacy cost.

            Parameters
            ----------
            mat_basis : the basis matrix
            inv_cov : the inverse of the co-variance matrix X
            """
            # d = torch.diag(self.mat_basis.T @ self.inv_cov @ self.mat_basis)
            vec_d = ((mat_basis.T @ inv_cov) * mat_basis.T).sum(axis=1)
            # vec_d = torch.tril(self.inv_cov.cumsum(axis=1)).sum(axis=0)
            return vec_d
        
        
        def _obj(param_t, f_var, f_pcost):
            """
            Objective function.

            Parameters
            ----------
            X : co-variance matrix
            param_t : privacy cost approximation parameter
            """

            const_k = param_t*torch.max(f_var)
            const_t = param_t*torch.max(f_pcost)
            log_sum_k = torch.log(torch.sum(torch.exp(param_t*f_var - const_k)))
            log_sum_t = torch.log(torch.sum(torch.exp(param_t*f_pcost - const_t)))
            f_obj = const_t + log_sum_t + const_k + log_sum_k
            return f_obj

        def _derivative(param_t, f_var, f_pcost, mat_index, mat_basis, inv_cov, var_bound):
            """Calculate derivatives."""
            const_k = param_t * torch.max(f_var)
            exp_k = torch.exp(param_t*f_var - const_k)
            g_var = (exp_k/var_bound*mat_index.T) @ mat_index

            const_t =  torch.max(param_t * f_pcost)
            mat_bix = inv_cov.T @ mat_basis
            exp_t =  torch.exp(param_t*f_pcost-const_t)
            g_pcost = -(exp_t*mat_bix) @ mat_bix.T

            coef_k = param_t/ torch.sum(exp_k)
            coef_t = param_t/ torch.sum(exp_t)
            grad_k = g_var * coef_k
            grad_t = g_pcost * coef_t

            grad = grad_k + grad_t
            # vec_grad =  torch.reshape(grad, [-1], 'F')
            return grad

        try:
            n = X.shape[0]
            mat_eye = torch.eye(n)
            inv_cov = _solve_positive_definite(X, mat_eye)
            cov = X
            # Todo: simplify _func_var when mat_index is the identity matrix
            mat_index = mat_eye
            # Todo: simplify _func_pcost when mat_basis is the prefix_sum matrix
            mat_basis = A

            f_var = _func_var(mat_index, cov, bound)
            f_pcost = _func_pcost(mat_basis, inv_cov)
            loss = _obj(param_t, f_var, f_pcost)
            gradients = _derivative(param_t, f_var, f_pcost, mat_index, mat_basis, inv_cov, bound)
            gradients = _project_update(gradients)
            assert not torch.any(torch.isnan(loss)).item()
            assert not torch.any(torch.isnan(loss)).item()
        except Exception as e:
            raise MatrixFactorizationError from e
        else:
            return loss, gradients

    @staticmethod
    @torch.no_grad()
    def lbfgs_direction(X: torch.Tensor,
                        dX: torch.Tensor,
                        prev_X: torch.Tensor,
                        prev_dX: torch.Tensor) -> torch.Tensor:
        S = X - prev_X
        Y = dX - prev_dX
        rho = 1.0 / torch.sum(Y * S)
        alpha = rho * torch.sum(S * dX)
        gamma = torch.sum(S * Y) / torch.sum(Y ** 2)
        Z = gamma * (dX - rho * torch.sum(S * dX) * Y)
        beta = rho * torch.sum(Y * Z)
        Z = Z + S * (alpha - beta)
        return Z

    @staticmethod
    @torch.no_grad()
    def lbfgs_direction_ex(X: torch.Tensor,
                           dX: torch.Tensor,
                           prev_X: torch.Tensor,
                           prev_dX: torch.Tensor) -> torch.Tensor:

        S = X - prev_X
        Y = dX - prev_dX
        rho = 1.0 / torch.sum(Y * S)
        alpha = rho * torch.sum(S * dX)
        gamma = torch.sum(S * Y) / torch.sum(Y ** 2)
        Z = gamma * (dX - rho * torch.sum(S * dX) * Y)
        beta = rho * torch.sum(Y * Z)
        Z += S * (alpha - beta)
        return Z

    @staticmethod
    def terminate_fn(dX: torch.Tensor) -> bool:
        return torch.abs(dX).amax().item() <= 1e-3

