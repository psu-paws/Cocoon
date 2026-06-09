"""
Banded Matrix Factorization (BandMF) mechanism for correlated-noise DP training.

Attribution / prior work
------------------------
- Apple PFL-Research implementation:
    https://github.com/apple/pfl-research/blob/c9d1f32f6e84fec984f448e2bc19a3e3fd5a4807/pfl/privacy/ftrl_mechanism.py
- Paper describing the algorithm (Algorithm 9):
    "(Amplified) Banded Matrix Factorization: A unified approach to private training"
    https://arxiv.org/pdf/2306.08153.pdf

This file improves on the pfl with:
  - Per-step division eliminated at init time (pre-division trick in CorNoiseGenerator)
  - DeepSpeed partition support: solve multiple noise tensors per step
"""

import math
import os
import time
import ctypes
from typing import Callable, Generic, Optional, Tuple, TypeVar, List

import inspect

import torch
import numpy as np

from pfl.exception import MatrixFactorizationError
from pfl.internal.bridge.factory import FrameworkBridgeFactory as bridges
from pfl.internal.ops import get_ops
from pfl.internal.platform.selector import get_platform
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.stats import TrainingStatistics

from pfl.internal.ops.selector import set_framework_module
from pfl.internal.ops import pytorch_ops

set_framework_module(pytorch_ops)

from .ftrl_factorizer_pytorch import FTRLMatrixFactorizer
from .smii_factorizer_autograd import SMIIMatrixFactorizer

from train_utils import CooldownLR


Tensor = TypeVar('Tensor')


class CorNoiseGenerator(Generic[Tensor]):
    """
    :param matrix:
        The lower triangular matrix L.
    :param bandwidth:
        Optional bandwidth of L.
    """

    def __init__(self, matrix: np.ndarray, bandwidth: Optional[int] = None, device: Optional[torch.device] = 'cpu', partition: Optional[int] = 1):
        self._matrix = torch.as_tensor(matrix, dtype=torch.float32).to(device)
        self._bandwidth = bandwidth

        # Pre-divide lower-triangular entries by diagonal to avoid per-step division.
        diag = torch.diag(self._matrix)
        divided_mat = self._matrix / diag.view(-1, 1)
        mask = torch.tril(torch.ones_like(self._matrix, dtype=torch.bool), diagonal=-1)
        self._matrix = torch.where(mask, divided_mat, self._matrix)

        self.device = device
        self._buf = []  # circular buffer: (bandwidth-1) slots per partition
        self.partition = partition
        self._write_ptr = 0
        self._step = 0          # step counter; increment by advance()
        self._col_idx = []  # column indices active in the buffer

    def set_matrix(self, dtype):
        self._matrix = self._matrix.to(dtype)

    def set_partition(self, max_part):
        self.partition = max_part

    def reset(self):
        self._buf = []
        self._step = 0
        self._write_ptr=0
        self._col_idx = []

    def skip_to_steady_state(self):
        """Pre-fill col_idx as if bandwidth-1 warm-up steps have already run.
        buf is NOT pre-allocated here; it is allocated lazily on the first step()
        call via y_i.repeat() (one-time cost). Full steady-state GEMV therefore
        starts from the second step() call onwards.
        """
        n_rows = self._matrix.shape[0]
        self._buf = []
        self._col_idx = [i % n_rows for i in range(self._bandwidth - 1)]
        self._step = 0
        self._write_ptr = 0

    def step(self, y_i: Tensor, partition_index: Optional[int] = 0, noise_type: Optional[torch.device]= torch.float32) -> Tensor:
        """ 
        Call advance() once after all partitions for this step are processed.
        """
        assert self.partition > partition_index

        if len(self._buf) == self.partition:
            if len(self._col_idx) < self._bandwidth - 1:  # warm-up: partial history, skipped when skip_to_steady_state()
                y_i = y_i - (self._matrix[self._step:self._step + 1, self._col_idx] @ self._buf[partition_index][:len(self._col_idx)])[0]
            else:
                if self._bandwidth > 2:
                    y_i = y_i - (self._matrix[self._step:self._step + 1, self._col_idx] @ self._buf[partition_index])[0]
                else:  # bandwidth == 2: scalar mul
                    y_i = y_i - self._matrix[self._step, self._col_idx] * self._buf[partition_index][0]

        if len(self._buf) < self.partition:  # allocate slot on first fill
            self._buf.append(y_i.repeat(self._bandwidth - 1, 1))
        else:
            self._buf[partition_index][self._write_ptr] = y_i
        return y_i

    def advance(self):
        """Advance step counter and circular-buffer pointer. Call once per training step."""
        if len(self._col_idx) < self._bandwidth - 1:
            self._col_idx.append(self._step)
        else:
            self._col_idx[self._write_ptr] = self._step

        self._step += 1
        self._write_ptr = self._step % (self._bandwidth - 1) # write target in circular buffer

class BandedMatrixFactorizationMechanism:
    """Banded matrix factorization for correlated DP noise.

    Computes the factorization on first use and caches it under
    ``./banded_mf_dp_ftrl/`` keyed by hyperparameters.  Per training step:
    call step(z, partition_index) for each partition, then advance().

    Args:
        num_iterations: Total training steps (matrix size).
        min_separation: Bandwidth b (number of non-zero diagonals).
        objective: 'sum' (minimise total variance) or 'max' (minimise worst-case).
        workload_matrix_type: 'A' for prefix-sum (SGD), 'M' for lr/momentum-weighted.
        lrs: Per-step learning rates; required when workload_matrix_type='M'.
        momentum: Momentum β; required when workload_matrix_type='M'.
        partition: Number of noise partitions (sharding).
        noise_offload: Keep noise buffers on CPU to save GPU memory.
    """

    _QUERY_MATRIX_DIR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'banded_mf_dp_ftrl')

    def __init__(self,
                 num_iterations: int,
                 min_separation: int,
                 objective: str,
                 workload_matrix_type: Optional[str] = None,
                 lrs: Optional[List[float]] = None,
                 lr_scheduler: Optional[str] = None,
                 momentum: Optional[float] = None,
                 bound: Optional[int] = None,
                 device_num: Optional[int] = None,
                 partition: Optional[int] = 1,
                 noise_offload: bool = False,
                 speed_mode: bool = False):

        super().__init__()
        if objective not in ('sum', 'max', 'p1'):
            raise NotImplementedError(f'Unsupported objective: {objective}')
        if objective in ('sum', 'max') and workload_matrix_type not in ('A', 'M'):
            raise NotImplementedError(f'Unsupported workload_matrix_type: {workload_matrix_type}')

        # bandwidth = b (non-zero diagonals), pfl used b-1 
        self.bandwidth = min_separation
        self.partition = partition
        self.num_iterations = num_iterations
        self.device = 'cpu' if noise_offload else f'cuda:{device_num}'

        # Build cache filenames from parameter components.
        parts = [objective, num_iterations, min_separation]
        if objective == 'max':
            parts.append(bound)
        if workload_matrix_type == 'M':
            parts.extend([momentum, lr_scheduler, lrs[0], len(lrs)])
        elif objective == 'p1':
            parts.append(lrs[0])
        _QUERY_MATRIX_NPY_NAME = 'query_matrix_' + '_'.join(str(p) for p in parts) + '.npy'

        if objective in ('sum', 'max'):
            x_parts = [objective, num_iterations, min_separation]
            if objective == 'max':
                x_parts.append(bound)
            _X_NPY_NAME = 'X_' + '_'.join(str(p) for p in x_parts) + '.npy'

        if speed_mode:
            # Skip factorization: banded matrix with tiny off-diagonal for timing benchmarks.
            # Off-diagonal = 0.01 (negligible correction, prevents noise explosion);
            # diagonal = 1.0. Values don't matter for timing — only band structure does.
            C = np.zeros((num_iterations, num_iterations), dtype=np.float32)
            for i in range(num_iterations):
                lo = max(0, i - self.bandwidth + 1)
                if lo < i:
                    C[i, lo:i] = 0.01
            np.fill_diagonal(C, 1.0)
            self._C = C
            self._solver = CorNoiseGenerator(self._C, self.bandwidth, self.device, self.partition)
            return

        query_matrix_dir = get_platform().create_checkpoint_directories(
            [self._QUERY_MATRIX_DIR_PATH])[0]
        query_matrix_path = os.path.join(query_matrix_dir, _QUERY_MATRIX_NPY_NAME)

        banded_matrix_mask = self._banded_mask(num_iterations, self.bandwidth)
        prefix_sum_matrix = np.tril(np.ones(num_iterations))

        if not os.path.exists(query_matrix_path):
            X_init = None
            if workload_matrix_type == 'A':
                workload_matrix = prefix_sum_matrix
            
            elif workload_matrix_type == 'M':
                steps_one_epoch = num_iterations // len(lrs)
                M_eta = np.tril(np.repeat(lrs, steps_one_epoch))

                row_indices = np.arange(num_iterations)
                col_indices = np.arange(num_iterations)
                data = momentum ** np.abs(row_indices[:, None] - col_indices)
                M_beta = np.where(row_indices[:, None] >= col_indices, data, 0)

                workload_matrix = M_eta @ M_beta

                if objective == 'sum':
                    X_path = os.path.join(query_matrix_dir, _X_NPY_NAME)
                    if not os.path.exists(X_path):
                        print('\n===> Factorizing prefix sum...\n')
                        matrix_factorizer = FTRLMatrixFactorizer(workload_matrix, banded_matrix_mask)
                        X = get_ops().to_numpy(matrix_factorizer.optimize())
                        np.save(X_path, X)
                        X_init = np.load(X_path)
            else:
                raise NotImplementedError(f'Unsupported workload_matrix_type: {workload_matrix_type}')

            if objective == 'sum':
                matrix_factorizer = FTRLMatrixFactorizer(workload_matrix, banded_matrix_mask, X_init=X_init)
            elif objective == 'max':
                matrix_factorizer = SMIIMatrixFactorizer(workload_matrix, banded_matrix_mask, bound=bound, X_init=None)
            elif objective == 'p1':
                coefficient_vector = np.array([(1 - lrs[0]) ** (num_iterations - i) for i in range(num_iterations)])
                matrix_factorizer = VCMatrixFactorizer(coefficient_vector, banded_matrix_mask)
            else:
                raise NotImplementedError(f'Unsupported objective: {objective}')

            if get_ops().distributed.local_rank == 0:  # rank-0 factorizes; others wait
                X = get_ops().to_numpy(matrix_factorizer.optimize())
                self._C = np.array(np.linalg.cholesky(X))
                np.save(query_matrix_path, self._C)

            while not os.path.exists(query_matrix_path):
                time.sleep(2)
        self._C = np.load(query_matrix_path)
        self._solver: CorNoiseGenerator = CorNoiseGenerator(
            self._C, self.bandwidth, self.device, self.partition)

    @property
    def query_matrix(self) -> np.ndarray:
        return self._C

    @property
    def current_step(self) -> int:
        """Current step index inside the correlated-noise solver."""
        return self._solver._step

    @staticmethod
    def _banded_mask(n: int, bandwidth: int) -> np.ndarray:
        """Return n×n binary mask: 1 within `bandwidth` diagonals, 0 outside."""
        row_indices = np.arange(n)[:, None]
        col_indices = np.arange(n)
        abs_diff = np.abs(row_indices - col_indices)
        return (abs_diff < bandwidth).astype(int)
    
    def step(self, flattened_noise: Tensor, partition_index: Optional[int] = 0, noise_type: Optional[torch.dtype]= torch.float32) -> Tensor:
        return self._solver.step(flattened_noise, partition_index, noise_type)

    def advance(self):
        return self._solver.advance()

    def set_matrix(self, noise_type: Optional[torch.device]= torch.float32):
        self._solver.set_matrix(noise_type)

    def set_partition(self, partition: int):
        """Update the number of active noise partitions (must be called before training)."""
        self.partition = partition
        self._solver.set_partition(partition)
    
    def reset(self):
        self._solver.reset()

    def skip_to_steady_state(self):
        self._solver.skip_to_steady_state()

    def diag(self, step):
        return self._solver._matrix[step,step]