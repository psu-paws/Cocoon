# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import contextlib
import sys
import time
import nvtx

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# data generation
import dlrm_data_pytorch as dp
from dlrm_preprocessing import NoiseCache, show_memory

# For distributed run
import extend_distributed as ext_dist

# numpy
import numpy as np
import optim.rwsadagrad as RowWiseSparseAdagrad
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
import random
import os

from fastDP import PrivacyEngine
from fastDP.bandmf import BandedMatrixFactorizationMechanism
from fastDP.noise_worker import noise_worker

# dataloader
try:
    from internals import fbDataLoader, fbInputBatchFormatter

    has_internal_libs = True
except ImportError:
    has_internal_libs = False

from tqdm.auto import tqdm, trange
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# mixed-dimension trick
from tricks.md_embedding_bag import md_solver, PrEmbeddingBag

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import onnx
    except ImportError as error:
        print("Unable to import onnx. ", error)

exc = getattr(builtins, "IOError", "FileNotFoundError")

def safelen(x):
    """Return x.size(0) for non-scalar tensors, 1 for 0-dim tensors."""
    return x.size(0) if x.dim() > 0 else 1

def time_wrap(use_gpu, time_list=None):
    """Record a CUDA event (if time_list given) or synchronize, then return wall time."""
    if use_gpu:
        if time_list != None:
            time_list.append(torch.cuda.Event(enable_timing=True))
            time_list[-1].record()
        else:
            torch.cuda.synchronize()
    return time.time()


def forward_pass(X, lS_o, lS_i, use_gpu, device, ndevices=1):
    """Move inputs to device and run a single DLRM forward pass."""
    with record_function("DLRM forward"):
        if use_gpu:
            if ndevices == 1:
                lS_i = (
                    [S_i.to(device) for S_i in lS_i]
                    if isinstance(lS_i, list)
                    else lS_i.to(device)
                )
                lS_o = (
                    [S_o.to(device) for S_o in lS_o]
                    if isinstance(lS_o, list)
                    else lS_o.to(device)
                )
        return dlrm(X.to(device), lS_o, lS_i)


def compute_loss(Z, T, use_gpu, device):
    """Compute the configured loss (mse/bce/wbce) between predictions Z and labels T."""
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce":
            if ext_dist.my_size > 1:
                return dlrm.module.loss_fn(Z, T.to(device))
            else:
                return dlrm.loss_fn(Z, T.to(device))
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
            loss_fn_ = dlrm.loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()


def unpack_batch(b):
    """Unpack a dataloader batch into (X, lS_o, lS_i, T, W, CBPP) with unit weights."""
    if args.data_generation == "internal":
        return fbInputBatchFormatter(b, args.data_size)
    else:
        return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class LRPolicyScheduler(_LRScheduler):
    """LR schedule: linear warmup → flat → quadratic decay → flat at near-zero."""

    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


class DLRM_Net(nn.Module):
    """DLRM model: embedding tables for sparse features + bottom/top MLPs + dot-product interaction.

    Architecture: dense features → bot_l (MLP) → interact with sparse embedding outputs → top_l (MLP) → click prob.
    Supports optional QR (quotient-remainder) and MD (mixed-dimension) embedding tricks.
    """

    def create_mlp(self, ln, sigmoid_layer):
        """Build a Sequential MLP from layer-size array ln; sigmoid at sigmoid_layer, ReLU elsewhere."""
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers).to("cuda:0")

    def create_emb(self, m, ln, weighted_pooling=None):
        """Create embedding tables: standard EmbeddingBag, or QR/MD variants based on flags."""
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            if ext_dist.my_size > 1:
                if i not in self.local_emb_indices:
                    continue
            n = ln[i]

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=True,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False)
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
                # v_W_l.append(torch.ones(n, dtype=torch.double))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # If running distributed, get local slice of embedding tables
            # Code Modified, Always Data-Parallel (Whole models each GPU)
            if ext_dist.my_size > 1:
                n_emb = len(ln_emb)
                self.n_global_emb = n_emb
                self.n_local_emb = n_emb
                self.n_emb_per_rank = None
                self.local_emb_indices = list(range(n_emb))

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling)
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def apply_mlp(self, x, layers):
        """Pass x through Sequential MLP layers."""
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        """Perform batched embedding lookups for all sparse feature tables.

        lS_i[k]: sparse index tensor for table k (shape: total_lookups)
        lS_o[k]: offset tensor marking batch boundaries for table k
        Returns list of row vectors, one per table, each of shape (batch, m_spa).
        """

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                s1 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                s2 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                print("quantized emb sizes:", s1, s2)

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(QV)
            else:
                E = emb_l[k]
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                    per_sample_weights=per_sample_weights,
                )

                ly.append(V)

        return ly

    #  using quantizing functions from caffe2/aten/src/ATen/native/quantized/cpu
    def quantize_embedding(self, bits):
        n = len(self.emb_l)
        self.emb_l_q = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.emb_l[k].weight
                )
            else:
                return
        self.emb_l = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x, ly):
        """Combine dense vector x with sparse embedding outputs ly via dot or cat interaction."""
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            _, ni, nj = Z.shape
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def clean_warmup(self, non_private):
        """Remove per-sample private_grad attributes accumulated during DP warm-up."""
        for layer in self.modules():
            if not non_private:
                for param in layer.parameters():
                    if hasattr(param, 'private_grad'):
                        del param.private_grad

    def forward(self, dense_x, lS_o, lS_i):
        return self._forward(dense_x, lS_o, lS_i)

    def _forward(self, dense_x, lS_o, lS_i):
        """Full forward pass: bot_l → embeddings → interaction → top_l → click probability."""
        x = self.apply_mlp(dense_x, self.bot_l)        # (B, bot_out)
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)  # list of (B, m_spa)
        z = self.interact_features(x, ly)              # (B, num_interactions + bot_out)
        p = self.apply_mlp(z, self.top_l)              # (B, 1)
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p
        return z

def int_list(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def float_list(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value


def inference(
    args,
    dlrm,
    best_acc_test,
    test_ld,
    device,
    use_gpu,
    log_iter=-1,
):
    """Run evaluation over test_ld; return model_metrics_dict and is_best flag.

    Computes per-batch accuracy and AUC. Updates best_acc_test if current run improves it.
    """
    is_main = ext_dist.my_local_rank in (0, -1)
    test_accu = 0
    test_samp = 0

    scores = []
    targets = []

    for i, testBatch in enumerate(test_ld):
        if nbatches > 0 and i >= nbatches:
            break

        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )

        if ext_dist.my_size > 1 and X_test.size(0) % ext_dist.my_size != 0:
            if is_main:
                print(f"Warning: skipping test batch {i} with size {X_test.size(0)}")
            continue

        # forward pass
        Z_test = forward_pass(
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        ### gather the distributed results on each rank ###
        # For some reason it requires explicit sync before all_gather call if
        # tensor is on GPU memory
        if Z_test.is_cuda:
            torch.cuda.synchronize()
        (_, batch_split_lengths) = ext_dist.get_split_lengths(X_test.size(0))
        if ext_dist.my_size > 1:
            Z_test = ext_dist.all_gather(Z_test, batch_split_lengths)

        with record_function("DLRM accuracy compute"):
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array

            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

            test_accu += A_test
            test_samp += mbs_test
        scores.append(S_test)
        targets.append(T_test)

    acc_test = test_accu / test_samp
    writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    is_best = acc_test > best_acc_test
    if is_best:
        best_acc_test = acc_test
    scores  = np.concatenate(scores,  axis=0)
    targets = np.concatenate(targets, axis=0)
    test_auc = sklearn.metrics.roc_auc_score(targets, scores)
    if is_main:
        print(f"acc {acc_test*100:.3f}%  best {best_acc_test*100:.3f}%  auc {test_auc:.4f}",
              flush=True)

    return model_metrics_dict, is_best


def _init_dp(args, dlrm, nbatches, ndevices, train_data, train_ld,
             ln_emb, m_spa, optimizer, lr_scheduler, use_gpu, device):
    """Set up BandMF, PrivacyEngine, and optional noise offload process.

    Three execution modes depending on args:
      - noise_offload=True  : BandMF runs in a background CPU subprocess; noise is queued to GPU.
      - preprocessing=True  : BandMF noise for cold (low-freq) rows is pre-computed by NoiseCache.
      - default (on-the-fly): PrivacyEngine generates correlated noise inline each step on GPU.

    Returns (dlrm, bandmf_solver, privacy_engine, noise_process, stop_event, noise_cache).
    noise_process/stop_event are None when not offloading; noise_cache is None without preprocessing.
    Also moves dlrm to device.
    """
    verbose = args.verbose
    bandmf_solver = None
    noise_process = stop_event = noise_cache = None
    preprocessing = args.preprocessing

    if args.min_separation > 1:
        # BandMF mechanism: correlated noise with memory length = min_separation
        bandmf_solver = BandedMatrixFactorizationMechanism(
            num_iterations=nbatches * args.nepochs,
            min_separation=args.min_separation,
            bound=args.bound,
            objective=args.objective,
            lr_scheduler=lr_scheduler.__class__.__name__,
            momentum=args.momentum,
            workload_matrix_type=args.workload_matrix_type,
            device_num=ext_dist.my_local_rank,
            speed_mode=args.speed_mode,
        )
        if verbose:
            print(f"[setup] BandMF: min_sep={args.min_separation} iters={nbatches} "
                  f"workload={args.workload_matrix_type}")

    if ext_dist.my_size > 1:
        dlrm = ext_dist.DDP(dlrm, device_ids=[ext_dist.my_local_rank])
        module = dlrm.module
    else:
        if dlrm.weighted_pooling == "fixed":
            for k, w in enumerate(dlrm.v_W_l):
                dlrm.v_W_l[k] = w.cuda()
        module = dlrm

    noise_queue = raw_noise_queue = None
    if args.noise_offload:
        torch.multiprocessing.set_start_method('spawn', force=True)
        noise_queue = torch.multiprocessing.Queue(maxsize=7)  # 7-slot ring buffer
        raw_noise_queue = torch.multiprocessing.Queue(maxsize=2)
        stop_event = torch.multiprocessing.Event()
        resume_event = torch.multiprocessing.Event()

    num_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if verbose:
        print(f"[setup] trainable params: {num_trainable_params:,}")

    privacy_engine = PrivacyEngine(
        module=module,
        batch_size=args.batch_size,
        sample_size=len(train_data),
        epochs=args.nepochs,
        max_grad_norm=args.max_grad_norm,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        accounting_mode=args.accounting_mode,
        clipping_mode="MixGhostClip",
        clipping_fn="Abadi",
        clipping_style="all-layer",
        origin_params=None,
        bandmf_solver=bandmf_solver,
        num_GPUs=ndevices,
        noise_queue=noise_queue,
        preprocessing=preprocessing,
        torch_seed_is_fixed=True,
        noise_offload=args.noise_offload,
        verbose=verbose,
    )
    privacy_engine.attach(optimizer)
    if verbose:
        print(f"[setup] PrivacyEngine: ε={args.target_epsilon} δ={args.target_delta} "
              f"noise_mult={privacy_engine.noise_multiplier:.4f} "
              f"max_grad_norm={privacy_engine.max_grad_norm}")

    noise_std = privacy_engine.noise_multiplier * privacy_engine.max_grad_norm

    if args.noise_offload:
        bandmf_solver.set_matrix("cpu")  # keep BandMF matrix on CPU for subprocess
        noise_process = torch.multiprocessing.Process(
            target=noise_worker,
            args=(noise_queue, stop_event, num_trainable_params, noise_std, bandmf_solver),
            kwargs={'raw_noise_queue': raw_noise_queue, 'gpu_device': device, 'verbose': verbose,
                    'speed_mode': args.speed_mode, 'num_threads': args.noise_num_threads},
        )
        noise_process.start()
        dlrm = dlrm.to(device)
    elif use_gpu:
        if preprocessing:
            noise_cache = NoiseCache(
                dlrm, train_ld, train_data, bandmf_solver,
                args.thresholds, ln_emb.size, m_spa, args.nepochs,
                noise_std, args.chunk_size, device, verbose=verbose,
            )
            noise_cache.build()
            if args.preprocess_only:
                sys.exit(0)
        dlrm = dlrm.to(device)
        privacy_engine.noise_vec = torch.empty(num_trainable_params, device='cuda:0')

    return dlrm, bandmf_solver, privacy_engine, noise_process, stop_event, noise_cache, raw_noise_queue


def report_timing(args, use_gpu, dlrm, privacy_engine,
                      start_epoch, end_epoch, start_iter, end_iter,
                      start_forward, end_forward, end_backward, end_step):
    """Print per-component CUDA timing breakdown after a benchmarked training run."""
    torch.cuda.synchronize()

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    def format_edges(numbers, edge_count=5, decimals=2):
        """
        Returns a string of the first and last `edge_count` numbers in a list,
        formatted to `decimals` decimal places.
        """
        # If the list is short, just format the whole thing to avoid overlapping
        if len(numbers) <= edge_count * 2:
            formatted = [f"{num:.{decimals}f}" for num in numbers]
            return f"{formatted}"
            
        # Grab and format the front and back slices
        front = [f"{num:.{decimals}f}" for num in numbers[:edge_count]]
        back = [f"{num:.{decimals}f}" for num in numbers[-edge_count:]]
        
        return f"{front} ... {back}"
    
    times_epoch  = [s.elapsed_time(e) for s, e in zip(start_epoch,   end_epoch)]
    times_iter   = [s.elapsed_time(e) for s, e in zip(start_iter,    end_iter)]
    times_iter2  = [s.elapsed_time(e) for s, e in zip(start_iter[:-1], start_iter[1:])]
    times_fwd    = [s.elapsed_time(e) for s, e in zip(start_forward,  end_forward)]
    times_bwd    = [s.elapsed_time(e) for s, e in zip(end_forward,    end_backward)]
    times_step   = [s.elapsed_time(e) for s, e in zip(end_backward,   end_step)]
    print(f"epoch_to_first_iter  {start_epoch[0].elapsed_time(start_iter[0]):.2f}ms")
    print(f"avg_epoch            {_avg(times_epoch):.2f}ms")

    if args.verbose:
        print(f"Iteration E2E   ={format_edges(times_iter)}")
        print(f"Inter-iteration ={format_edges(times_iter2)}")
        print(f"Forward         ={format_edges(times_fwd)}")
        print(f"Backward        ={format_edges(times_bwd)}")
        print(f"Optimizer Step  ={format_edges(times_step)}")
    print(f"Avg_Iter {_avg(times_iter):.2f}ms  "
          f"Avg_Forward {_avg(times_fwd):.2f}ms  "
          f"Avg_Backward  {_avg(times_bwd):.2f}ms  "
          f"Avg_Optimizer_Step {_avg(times_step):.2f}ms")

    if args.non_private:
        return

    if args.noise_offload:
        times_clip  = [s.elapsed_time(e) for s, e in zip(dlrm.start_clip, dlrm.end_clip)]
        times_nadd  = [s.elapsed_time(e) for s, e in zip(
            privacy_engine.end_noise_comp, privacy_engine.end_noise_add)]
        if args.verbose:
            print(f"Grad Clip       ={format_edges(times_clip)}")
            print(f"Noise Addition  ={format_edges(times_nadd)}")
        print(f"Avg_Grad_Clip {_avg(times_clip):.2f}ms  Avg_Noise_Add {_avg(times_nadd):.2f}ms")
    else:
        times_clip  = [s.elapsed_time(e) for s, e in zip(dlrm.start_clip, dlrm.end_clip)]
        times_ngen  = [s.elapsed_time(e) for s, e in zip(
            privacy_engine.start_noise_gen, privacy_engine.end_noise_gen)]
        times_nadd  = [s.elapsed_time(e) for s, e in zip(
            privacy_engine.end_noise_gen, privacy_engine.end_noise_add)]
        if args.verbose:
            print(f"Grad Clip       ={format_edges(times_clip)}")
            print(f"Noise Generate  ={format_edges(times_ngen)}")
            print(f"Noise Addition  ={format_edges(times_nadd)}")
        print(f"Avg_Grad_Clip {_avg(times_clip):.2f}ms  "
              f"Avg_Noise_Gen {_avg(times_ngen):.2f}ms  "
              f"Avg_Noise_Add {_avg(times_nadd):.2f}ms")
        if args.preprocessing:
            times_ncostr = [s.elapsed_time(e) for s, e in zip(
                privacy_engine.start_noise_gen, privacy_engine.PPtransfer)]
            times_pptrans = [s.elapsed_time(e) for s, e in zip(
                privacy_engine.PPtransfer, privacy_engine.end_noise_gen)]
            if args.verbose:
                print(f"Noise Reassemble ={format_edges(times_ncostr)}")
                print(f"Noise Transfer  ={format_edges(times_pptrans)}")
            print(f"Avg_Noise_Assemble {_avg(times_ncostr):.2f}ms  "
                  f"Avg_Noise_Transfer {_avg(times_pptrans):.2f}ms")


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=int_list, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=int_list, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=int_list, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=float_list, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation",
        type=str,
        choices=["random", "dataset", "internal"],
        default="random",
    )  # synthetic, dataset or internal
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # quantize
    parser.add_argument("--quantize-mlp-with-bit", type=int, default=32)
    parser.add_argument("--quantize-emb-with-bit", type=int, default=32)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=0)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="print noise tensor shapes/stats each step (DP setup and training)")
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)

    # privacy-DP  (pass --non-private to disable DP; default is private training)
    parser.add_argument("--non-private", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=30)
    parser.add_argument("--noise-multiplier", type=float, default=0.1)
    parser.add_argument("--target-epsilon", type=float, default=3.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--accounting-mode", type=str, default="rdp")

    # privacy-FTRL
    parser.add_argument("--min-separation", type=int, default=4)
    parser.add_argument("--objective", type=str, default="sum")
    parser.add_argument("--bound", type=int, default=1001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--workload-matrix-type", type=str, default="A")
    
    # privacy-FTRL-Ours
    parser.add_argument("--preprocessing", action="store_true", default=False)
    parser.add_argument("--preprocess-only", action="store_true", default=False,
                        help="Exit after preprocessing (noise memory layout printed); skip training.")
    parser.add_argument("--thresholds", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4_300_000)
    parser.add_argument("--synthetic", type=str, default="")
    
    
    parser.add_argument("--noise_offload", action="store_true", default=False)
    parser.add_argument("--speed-mode", action="store_true", default=False,
                        help="Skip BandMF factorization; use random banded matrix for timing benchmarks.")
    parser.add_argument("--noise-num-threads", type=int, default=None,
                        help="OMP/MKL thread count for the CPU noise_worker subprocess.")


    global args
    global nbatches
    global nbatches_test
    global writer
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.memory._set_allocator_settings(
        "expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
    )

    if args.dataset_multiprocessing:
        assert sys.version_info[0] >= 3 and sys.version_info[1] > 7, (
            "The dataset_multiprocessing flag is susceptible to a bug in Python 3.7 "
            "and under. https://github.com/facebookresearch/dlrm/issues/172"
        )

    if args.weighted_pooling is not None:
        if args.qr_flag:
            sys.exit("ERROR: quotient remainder with weighted pooling is not supported")
        if args.md_flag:
            sys.exit("ERROR: mixed dimensions with weighted pooling is not supported")
    if args.quantize_emb_with_bit in [4, 8]:
        if args.qr_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with quotient remainder is not supported"
            )
        if args.md_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with mixed dimensions is not supported"
            )
        if args.use_gpu:
            sys.exit("ERROR: 4 and 8-bit quantization on GPU is not supported")

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()

    if not args.debug_mode:
        ext_dist.init_distributed(
            local_rank=args.local_rank, use_gpu=use_gpu, backend=args.dist_backend
        )

    is_main = ext_dist.my_local_rank in (0, -1)

    if is_main:
        print(args)

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        if ext_dist.my_size > 1:
            ngpus = 1
            device = torch.device("cuda", ext_dist.my_local_rank)
        else:
            ngpus = torch.cuda.device_count()
            device = torch.device("cuda", 0)
        if is_main:
            print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        if is_main:
            print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    n_acc_steps = args.batch_size // (args.mini_batch_size * torch.cuda.device_count()) # gradient accumulation steps
    # input data

    if args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = 0
        if is_main:
            print(f"Train batches: {nbatches}")

        time1 = time.time()
        _ = iter(train_ld)  # dataloader warmup
        if is_main:
            print(f"DataLoader warmup: {time.time() - time1:.2f}s")

        ln_emb = train_data.counts
        if args.synthetic == "78M":
            ln_emb = [1472, 577, 82741, 18940, 305, 23, 1172,
                    633, 3, 9090, 5918, 64300, 3207, 27, 1550,
                    44262, 10, 5485, 2161, 3, 56473, 17, 15,
                    27360, 104, 12934]
        elif args.synthetic == "u520M":
            ln_emb = [1298561, 1298561, 1298561, 1298561, 1298561,
                    1298561, 1298561, 1298561, 1298561, 1298561,
                    1298561, 1298561, 1298561, 1298561, 1298561,
                    1298561, 1298561, 1298561, 1298561, 1298561,
                    1298561, 1298561, 1298561, 1298561, 1298561,
                    1298552]
        elif args.synthetic == "double":
            ln_emb = [2920, 1166, 20262454, 4405216, 610, 48, 25034, 1266, 6, 186290, 11366,
                    16703186, 6388, 54, 29984, 10922612, 20, 11304, 4346, 8, 14093094,
                    36, 30, 572362, 210, 285144]
        elif args.synthetic == "half":
            ln_emb = [730, 292, 5065614, 1101304, 153, 12, 6259, 317, 2, 46573, 2842, 4175797,
                    1597, 14, 7496, 2730653, 5, 2826, 1087, 2, 3523274, 9, 8, 143091, 53,
                    71286]
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        else:
            ln_emb = np.array(ln_emb)
        m_den = train_data.m_den
        ln_bot[0] = m_den
    elif args.data_generation == "internal":
        if not has_internal_libs:
            raise Exception("Internal libraries are not available.")
        NUM_BATCHES = 5000
        nbatches = args.num_batches if args.num_batches > 0 else NUM_BATCHES
        train_ld, feature_to_num_embeddings = fbDataLoader(args.data_size, nbatches)
        ln_emb = np.array(list(feature_to_num_embeddings.values()))
        m_den = ln_bot[0]
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        args.data_size = args.num_batches * args.mini_batch_size
        train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader(
            args, ln_emb, m_den
        )
        nbatches = args.num_batches * n_acc_steps if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

    args.ln_emb = ln_emb.tolist()

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims,
        ).tolist()

    if args.debug_mode and is_main:
        print(f"mlp_top: {ln_top.size - 1} layers {ln_top}")
        print(f"mlp_bot: {ln_bot.size - 1} layers {ln_bot}")
        print(f"interactions: {num_int}  features: {num_fea}  dense: {m_den}  sparse: {m_spa}")
        print(f"embeddings ({ln_emb.size}) x {m_spa}: {ln_emb}")
        for j, inputBatch in enumerate(train_ld):
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
            if nbatches > 0 and j >= nbatches:
                break
            print(f"mini-batch {j}: X={X.shape} lS_i=[{lS_i[0].shape},...] T={T.shape}")

    global ndevices
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    global dlrm
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        weighted_pooling=args.weighted_pooling,
        loss_function=args.loss_function,
    )
    num_trainable_params = sum(p.numel() for p in dlrm.parameters() if p.requires_grad)
    num_total_params     = sum(p.numel() for p in dlrm.parameters())
    if is_main:
        print(dlrm)
        print(f"device={device}  ndevices={ndevices}")
        print(f"trainable params: {num_trainable_params:,}  total: {num_total_params:,}")
        print(f"grad_accum_steps={n_acc_steps}  logical_batch={args.batch_size}")

    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = (
            dlrm.parameters()
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            args.lr_num_warmup_steps,
            args.lr_decay_start_step,
            args.lr_num_decay_steps,
        )
        bandmf_solver = noise_process = stop_event = noise_cache = raw_noise_queue = None
        privacy_engine = None
        if not args.non_private:
            dlrm, bandmf_solver, privacy_engine, noise_process, stop_event, noise_cache, raw_noise_queue = \
                _init_dp(args, dlrm, nbatches, ndevices, train_data, train_ld,
                         ln_emb, m_spa, optimizer, lr_scheduler, use_gpu, device)
        else:
            if use_gpu:
                dlrm = dlrm.to(device)

    ### main loop ###

    # training or inference
    best_acc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0

    # Load model if specified
    if not (args.load_model == ""):
        if is_main:
            print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device("cuda"),
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_acc_test = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        if is_main:
            print("Saved at: epoch={:d}/{:d} batch={:d}/{:d} ntbatch={:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test))
            print("Training state: loss={:.6f}  test_acc={:.3f}%".format(
                ld_train_loss, ld_acc_test * 100))

    if args.inference_only:
        # Currently only dynamic quantization with INT8 and FP16 weights are
        # supported for MLPs and INT4 and INT8 weights for EmbeddingBag
        # post-training quantization during the inference.
        # By default we don't do the quantization: quantize_{mlp,emb}_with_bit == 32 (FP32)
        assert args.quantize_mlp_with_bit in [
            8,
            16,
            32,
        ], "only support 8/16/32-bit but got {}".format(args.quantize_mlp_with_bit)
        assert args.quantize_emb_with_bit in [
            4,
            8,
            32,
        ], "only support 4/8/32-bit but got {}".format(args.quantize_emb_with_bit)
        if args.quantize_mlp_with_bit != 32:
            if args.quantize_mlp_with_bit in [8]:
                quantize_dtype = torch.qint8
            else:
                quantize_dtype = torch.float16
            dlrm = torch.quantization.quantize_dynamic(
                dlrm, {torch.nn.Linear}, quantize_dtype
            )
        if args.quantize_emb_with_bit != 32:
            dlrm.quantize_embedding(args.quantize_emb_with_bit)

    tb_file = "./" + args.tensor_board_filename
    writer = SummaryWriter(tb_file)
    if not args.noise_offload and not args.non_private and noise_cache is not None:
        privacy_engine.noise_cache = noise_cache

    start_epoch = []
    end_epoch = []
    start_iter = []
    end_iter = []
    start_forward = []
    end_forward = []
    end_backward = []
    end_step = []

    if not args.non_private and args.min_separation > 1:
        bandmf_solver.reset()
    if not args.non_private:
        privacy_engine.benchmark = args.enable_profiling
        if args.enable_profiling:
            dlrm.start_clip = []
            dlrm.end_clip = []
            privacy_engine.start_noise_gen = []
            privacy_engine.end_noise_gen = []
            privacy_engine.PPtransfer = []
            privacy_engine.end_noise_comp = []
            privacy_engine.end_noise_add = []

    ext_dist.barrier()
    if not args.inference_only:
        k = 0
        while k < args.nepochs:
            if k < skip_upto_epoch:
                k += 1
                continue

            if args.enable_profiling:
                t1 = time_wrap(use_gpu, start_epoch)

            loader_iter = iter(train_ld)
            j = 0
            for inputBatch in loader_iter:
                if j < skip_upto_batch:
                    # epoch_pbar.update(1)
                    continue

                t1 = time_wrap(use_gpu, start_iter)
                X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

                # early exit if nbatches was set by the user and has been exceeded
                if nbatches > 0 and j >= nbatches:
                    break

                # Skip the batch if batch size not multiple of total ranks
                if ext_dist.my_size > 1 and X.size(0) % ext_dist.my_size != 0:
                    if is_main:
                        print(f"Warning: skipping batch {j} with size {X.size(0)}")
                    j += 1
                    continue

                mbs = T.shape[0]  # = args.mini_batch_size except for last batch

                # Optimizer step when gradient accumulation is complete
                if (j + 1) % n_acc_steps == 0:
                    if args.enable_profiling:
                        time_wrap(use_gpu, start_forward)
                    Z = forward_pass(X, lS_o, lS_i, use_gpu, device, ndevices=ndevices)
                    E = compute_loss(Z, T, use_gpu, device)
                    L = E.detach().cpu().numpy()
                    if args.enable_profiling:
                        time_wrap(use_gpu, end_forward)
                    E.backward()
                    if args.enable_profiling:
                        time_wrap(use_gpu, end_backward)
                    optimizer.step()
                    if args.enable_profiling:
                        time_wrap(use_gpu, end_step)
                    if args.verbose and is_main and privacy_engine is not None:
                        # inspect the noise vector added this step
                        n = privacy_engine.noise_vec
                        print(f"[verbose] step {j} noise shape={tuple(n.shape)} "
                              f"std={n.std().item():.4f} mean={n.mean().item():.4f} "
                              f"norm={n.norm().item():.4f}")
                    lr_scheduler.step()
                    dlrm.zero_grad(set_to_none=True)
                    optimizer.zero_grad()
                else:
                    # Accumulation step — suppress gradient sync in DDP
                    ctx = dlrm.no_sync() if ext_dist.my_size > 1 else contextlib.nullcontext()
                    with ctx:
                        Z = forward_pass(X, lS_o, lS_i, use_gpu, device, ndevices=ndevices)
                        E = compute_loss(Z, T, use_gpu, device)
                        L = E.detach().cpu().numpy()
                        rng = nvtx.start_range(message="backward_accum", color="green")
                        with record_function("DLRM backward"):
                            E.backward()
                        nvtx.end_range(rng)

                t2 = time_wrap(use_gpu, end_iter)
                total_time += t2 - t1
                total_loss += L * mbs
                total_iter += 1
                total_samp += mbs

                should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
                should_test  = (
                    args.test_freq > 0
                    and args.data_generation in ["dataset", "random"]
                    and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )

                if (should_print or should_test) and is_main:
                    train_loss = total_loss / total_samp
                    wall_time = f" ({time.strftime('%H:%M')})" if args.print_wall_time else ""
                    if args.print_time:
                        gT = 1000.0 * total_time / total_iter
                        print(
                            f"Epoch {k} it {j+1}/{nbatches}  "
                            f"loss {train_loss:.6f}  {gT:.2f} ms/it{wall_time}",
                            flush=True,
                        )
                    else:
                        print(
                            f"Epoch {k} it {j+1}/{nbatches}  "
                            f"loss {train_loss:.6f}{wall_time}",
                            flush=True,
                        )
                    log_iter = nbatches * k + j + 1
                    writer.add_scalar("Train/Loss", train_loss, log_iter)
                    total_time = total_loss = total_iter = total_samp = 0

                if should_test:
                    if is_main:
                        print(f"Testing at {j+1}/{nbatches} of epoch {k}")
                    log_iter = nbatches * k + j + 1
                    model_metrics_dict, is_best = inference(
                        args, dlrm, best_acc_test,
                        test_ld, device, use_gpu, log_iter,
                    )
                    if is_best and args.save_model and not args.inference_only:
                        model_metrics_dict.update({
                            "epoch": k, "iter": j + 1,
                            "train_loss": train_loss, "total_loss": total_loss,
                            "opt_state_dict": optimizer.state_dict(),
                        })
                        if is_main:
                            print(f"Saving model to {args.save_model}")
                        torch.save(model_metrics_dict, args.save_model)

                j += 1

            if args.enable_profiling:
                time_wrap(use_gpu, end_epoch)
            k += 1

        if is_main:
            show_memory()
    else:
        if is_main:
            print("Testing for inference only")
        inference(args, dlrm, best_acc_test, test_ld, device, use_gpu)

    if args.enable_profiling:
        report_timing(args, use_gpu, dlrm, privacy_engine,
                      start_epoch, end_epoch, start_iter, end_iter,
                      start_forward, end_forward, end_backward, end_step)

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    total_time_end = time_wrap(use_gpu)
    if noise_process is not None:
        stop_event.set()
        noise_process.join()


if __name__ == "__main__":
    run()
