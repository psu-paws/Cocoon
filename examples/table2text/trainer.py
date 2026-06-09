import inspect
import json
import os
import re
import shutil
import warnings
import math
import psutil, resource
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from ml_swissknife import utils
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import is_datasets_available, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import distributed_broadcast_scalars
from transformers.trainer_utils import (EvalPrediction, EvaluationStrategy, IntervalStrategy, PREFIX_CHECKPOINT_DIR,
                                        PredictionOutput, TrainOutput, set_seed)
from transformers.utils import logging
from transformers.deepspeed import deepspeed_init
import deepspeed
from deepspeed import comm as dist
import threading
from fastDP.supported_layers_grad_samplers import _supported_layers_norm_sample_AND_clipping
import torch.autograd.profiler as profiler
import decoding_utils
from compiled_args import (DataTrainingArguments, ModelArguments, PrivacyArguments,
                            TrainingArguments)
import time, copy
from fastDP.bandmf import BandedMatrixFactorizationMechanism
from fastDP.noise_worker import noise_worker
from fastDP.deepspeed_patch import apply_dp_patch, setup_dp_optimizer

logger = logging.get_logger(__name__)

CXL_DEVICES    = 1
BW_CXLGPU      = 22_400_000_000 / 4 * CXL_DEVICES  # bytes/s (FP32)
BW_CXLCPU      = 22_800_000_000 / 4 * CXL_DEVICES
BW_CPUGPU      = 23_300_000_000 / 4 * CXL_DEVICES
THROUGHPUT_NMP = 47_900_000_000 / 4 * CXL_DEVICES  # NMP GEMV throughput (bytes/s)

# Compute-location toggles — comment in/out to switch mode.
# Transfer cost is assumed hidden by computation (optimistic/favorable baseline).
CPU_MODE = "GEMV_at_CPU" #   "GEMV_at_CPU" | "GEMV_at_GPU"
NMP_MODE = "GEMV_at_NMP" #   "GEMV_at_NMP" | "GEMV_at_CPU" | "GEMV_at_GPU"

def show_memory(dev):
    print("torch.cuda.memory_allocated: %f MiB"%(torch.cuda.memory_allocated(dev)/1024/1024))
    print("torch.cuda.memory_reserved: %f MiB"%(torch.cuda.memory_reserved(dev)/1024/1024))
    print("torch.cuda.max_memory_allocated: %f MiB"%(torch.cuda.max_memory_allocated(dev)/1024/1024))
    print("torch.cuda.max_memory_reserved: %f MiB"%(torch.cuda.max_memory_reserved(dev)/1024/1024))
        
def divide_with_last_residue(total, divisor):
    if divisor <= 0:
        raise ValueError("Divisor must be a positive number.")

    part_value = math.ceil(total / divisor)
    parts = [part_value] * (divisor - 1)
    last_part = total - sum(parts)  # residue goes to last partition
    parts.append(last_part)

    prefix_sum_list = [0]
    current_sum = 0
    for part in parts:
        current_sum += part
        prefix_sum_list.append(current_sum)

    return prefix_sum_list

def get_max_memory_mb():
    peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_mb = peak_memory_kb / 1024.0
    print(f"Max Main Memory - CPU Noise Table: {peak_memory_mb:.2f} MB", flush=True)


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~transformers.TrainingArguments`
            with the ``output_dir`` set to a directory named `tmp_trainer` in the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or
            :obj:`eval_dataset`. Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is
            provided, an instance of :func:`~transformers.DataCollatorWithPadding` otherwise.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~transformers.Trainer.train` will start from a new instance of the model as given by this function.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.
        kwargs:
            Deprecated keyword arguments.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        args: Optional[TrainingArguments] = None,
        model_args: Optional[ModelArguments] = None,
        data_args: Optional[DataTrainingArguments] = None,
        privacy_args: Optional[PrivacyArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),

        val_dataset: Optional[Dataset] = None,
        generation_stuff: Optional[Dict] = None,
        noise_queue : Optional[torch.multiprocessing.Queue] = None,
        **kwargs,
    ):
        if args is None:
            logger.info("No `TrainingArguments` passed, using the current path as `output_dir`.")
            args = TrainingArguments("tmp_trainer")
        self.args = args
        self.model_args = model_args
        self.data_args = data_args
        self.privacy_args = privacy_args

        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        assert (
            model is not None or model_init is not None
        ), "You must provide a model to use `Trainer`, either by using the `model` argument or the `model_init` " \
           "argument."
        assert model_init is None
        self.model = model if model is not None else None
        self.num_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        from transformers.modeling_utils import Conv1D
        self.num_non_embedding_params = sum(
            param.numel()
            for module in self.model.modules() if isinstance(module, (nn.LayerNorm, Conv1D))
            for param in module.parameters() if param.requires_grad
        )
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.val_dataset = val_dataset
        self.generation_stuff = generation_stuff
        self.tokenizer = tokenizer
        self.curr_best_eval = 10000000.
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        self.log_history = []
        if "prediction_loss_only" in kwargs:
            warnings.warn(
                "Passing `prediction_loss_only` as a keyword argument is deprecated and won't be possible in a future "
                "version. Use `args.prediction_loss_only` instead.",
                FutureWarning,
            )
            self.args.prediction_loss_only = kwargs.pop("prediction_loss_only")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                    "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                    + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

        self.global_step = None
        self.epoch = None
        self.total_flos = None
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        if self.args.label_names is None:
            self.args.label_names = (
                ["start_positions, end_positions"]
                if type(self.model) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values()
                else ["labels"]
            )

        if self.args.deepspeed_config:
            deepspeed.init_distributed()
            self.spared_noise_length = self.num_params // torch.distributed.get_world_size()
            if torch.distributed.get_rank() == 0:
                print(f"spared: {self.spared_noise_length} world size: {torch.distributed.get_world_size()}")

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(self.model.forward)
        signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        signature_columns += ["label", "label_ids"]
        columns = [k for k in signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        dset_description = "" if description is None else f"in the {description} set "
        logger.info(
            f"The following columns {dset_description}don't have a corresponding argument in `"
            f"{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
        )
        dataset.set_format(type=dataset.format["type"], columns=columns)

    def _get_train_sampler(self, shuffle=True) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            # Sometimes we don't want to shuffle!
            if shuffle:
                return (
                    RandomSampler(self.train_dataset)
                    if self.args.local_rank == -1
                    else DistributedSampler(self.train_dataset)
                )
            else:
                return SequentialSampler(self.train_dataset)

    def get_train_dataloader(self, train_sampler=None) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if train_sampler is None:
            train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.eval_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential
        sampler (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def num_examples(self, dataloader: DataLoader) -> int:
        """Return number of samples in a DataLoader."""
        return len(dataloader.dataset)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
                if self.args.fp16:
                    inputs[k] = inputs[k].half()
                if self.args.bf16:
                    inputs[k] = inputs[k].bfloat16()

        # GPT-2 don't use these; these are mostly for encoder-decoder architectures.
        inputs.pop('src_attn', None)
        inputs.pop('tgt_attn', None)
        inputs.pop('src', None)
        return inputs

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               (not any(nd in n for nd in no_decay)) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = optim.SGD(net.parameters(), lr=self.args.learning_rate)
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def clean_warmup(self, model, non_private):
        for layer in model.modules():
            if hasattr(layer, 'activations'):
                del layer.activations
            if hasattr(layer, 'backprops'):
                del layer.backprops
            if not non_private:
                for param in layer.parameters():
                    if hasattr(param, 'private_grad'):
                        del param.private_grad

    # ------------------------------------------------------------------
    # Private helpers (called from train())
    # ------------------------------------------------------------------

    def _setup_deepspeed_training(self, bandmf_solver, t_total):
        """Initialize DeepSpeed engine, wire DP optimizer, and launch noise-offload processes.

        Returns:
            (part_GPU, part_CPU, part_NMP, noise_process, stop_event)
            Process handles and stop_event are None when not used.
        """
        apply_dp_patch()
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            args=self.args, model=self.model, optimizer=self.optimizer,
            model_parameters=self.model.parameters(),
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.memory._set_allocator_settings(
            "expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
        )
        self.bandmf_solver = bandmf_solver

        part_GPU = self.privacy_args.GPU_partition
        part_CPU = part_NMP = 0
        noise_process = stop_event = None

        if not self.privacy_args.non_private:
            self.partition_size = int(self.num_params / self.args.num_GPUs)
            setup_dp_optimizer(
                self.optimizer,
                bandmf_solver=bandmf_solver,
                noise_multiplier=self.privacy_args.noise_multiplier,
                per_example_max_grad_norm=self.privacy_args.per_example_max_grad_norm,
                train_batch_size=self.args.train_batch_size,
                partition=self.privacy_args.noise_partition,
                noise_offload=self.privacy_args.noise_offload,
                partition_size=self.partition_size,
                local_rank=self.args.local_rank,
                part_GPU=self.privacy_args.GPU_partition,
            )

            if self.privacy_args.noise_offload and self.privacy_args.min_separation > 1:
                part_CPU = self.privacy_args.CPU_partition
                part_NMP = self.privacy_args.noise_partition - part_GPU - part_CPU
                logger.info(f"Noise partitions: GPU={part_GPU} CPU={part_CPU} NMP={part_NMP}")

                torch.multiprocessing.set_start_method('spawn', force=True)
                self.noise_queue = torch.multiprocessing.Queue(maxsize=1)
                self.raw_noise_queue = torch.multiprocessing.Queue(maxsize=1)
                stop_event = torch.multiprocessing.Event()
                resume_event = torch.multiprocessing.Event()
                self.optimizer.part_GPU = part_GPU

                if part_CPU > 0 and CPU_MODE == "GEMV_at_CPU":
                    self.bandmf_solver_cpu = BandedMatrixFactorizationMechanism(
                        num_iterations=t_total,
                        min_separation=self.privacy_args.min_separation,
                        bound=1001, objective='sum',
                        lr_scheduler=self.lr_scheduler.__class__.__name__,
                        momentum=0.9, workload_matrix_type='A',
                        noise_offload=True, partition=part_CPU,
                        speed_mode=self.privacy_args.speed_mode,
                    )
                    if getattr(self.model, 'benchmark', False):
                        self.bandmf_solver_cpu.skip_to_steady_state()
                    if self.args.local_rank == 0:
                        _chunk_size = self.partition_size // self.privacy_args.noise_partition
                        _buf_gb_per_slot = (self.privacy_args.min_separation - 1) * _chunk_size * 4 / 1e9
                        _buf_gb_per_rank = part_CPU * _buf_gb_per_slot
                        print(f"[MEM] Expected buf per rank: {_buf_gb_per_rank:.2f} GB  "
                              f"(part_CPU={part_CPU} × bw-1={self.privacy_args.min_separation - 1} × chunk={_chunk_size} × 4B = {_buf_gb_per_slot:.2f} GB/slot)", flush=True)
                        _numa_free_gb = 0
                        try:
                            with open('/proc/self/status') as _sf:
                                for _sl in _sf:
                                    if _sl.startswith('Mems_allowed:'):
                                        _mask = int(_sl.split()[1].replace(',', ''), 16)
                                        break
                            # use the first (lowest) allowed NUMA node — that is the node
                            # where both GPU training and noise workers allocate.
                            _best_node, _best_free = -1, 0.0
                            for _n in range(8):
                                if not (_mask & (1 << _n)):
                                    continue
                                try:
                                    _free = 0
                                    with open(f'/sys/devices/system/node/node{_n}/meminfo') as _mf:
                                        for _ml in _mf:
                                            if 'MemFree' in _ml or 'FilePages' in _ml or 'SReclaimable' in _ml:
                                                _free += int(_ml.split()[3]) / 1024 / 1024
                                    _best_node, _best_free = _n, _free
                                    break  # first allowed node is the GPU's NUMA node
                                except FileNotFoundError:
                                    pass
                            if _best_node >= 0:
                                _numa_free_gb = _best_free
                            print(f"[MEM] NUMA node{_best_node} available: {_numa_free_gb:.2f} GB (MemFree+FilePages+SReclaimable)", flush=True)
                        except Exception:
                            pass
                        if _numa_free_gb > 0:
                            _buf_gb_all_ranks = _buf_gb_per_slot * self.args.num_GPUs
                            _max_cpu = max(1, int(_numa_free_gb * 0.9 / _buf_gb_all_ranks))
                            print(f"[MEM] buf/part across ranks: {_buf_gb_all_ranks:.2f} GB | "
                                  f"=> advised max part_CPU={_max_cpu}", flush=True)
                    _cpu_affinity = len(os.sched_getaffinity(0))
                    _num_threads = max(1, _cpu_affinity // self.args.num_GPUs)
                    noise_process = torch.multiprocessing.Process(
                        target=noise_worker,
                        args=(self.noise_queue, stop_event, 0,
                              self.privacy_args.noise_multiplier, self.bandmf_solver_cpu),
                        kwargs={
                            'raw_noise_queue': self.raw_noise_queue,
                            'partition': part_CPU,
                            'transfer_to_gpu': False,
                            'dev_rank': self.args.local_rank,
                            'num_threads': _num_threads,
                        },
                    )
                    print(f"[DEBUG rank={self.args.local_rank}] noise_worker num_threads={_num_threads} (affinity={_cpu_affinity} / {self.args.num_GPUs} GPUs)", flush=True)
                    noise_process.start()
                    print(f"[DEBUG rank={self.args.local_rank}] noise_process started pid={noise_process.pid}", flush=True)
                    time.sleep(0.5)  # let subprocess initialize before reading its RSS
                    try:
                        with open(f'/proc/{noise_process.pid}/status') as _pf:
                            _child_rss = _child_peak = 0
                            for _pl in _pf:
                                if _pl.startswith('VmRSS:'):
                                    _child_rss = int(_pl.split()[1]) / 1024 / 1024
                                elif _pl.startswith('VmPeak:'):
                                    _child_peak = int(_pl.split()[1]) / 1024 / 1024
                        print(f"[MEM rank={self.args.local_rank}] child pid={noise_process.pid} RSS={_child_rss:.2f}GB peak={_child_peak:.2f}GB", flush=True)
                    except Exception:
                        pass
            else:
                part_GPU = self.privacy_args.GPU_partition

        fp16, bf16 = self.model.fp16_enabled(), self.model.bfloat16_enabled()
        logger.info(f'deepspeed=True fp16={fp16} bf16={bf16}')
        return part_GPU, part_CPU, part_NMP, noise_process, stop_event

    def _resume_from_checkpoint(self, model_path, model, num_update_steps_per_epoch):
        """Load optimizer/scheduler states from checkpoint and compute resume position.

        Returns:
            (epochs_trained, steps_trained_in_current_epoch)
        """
        epochs_trained = steps_trained_in_current_epoch = 0
        if model_path is None:
            return epochs_trained, steps_trained_in_current_epoch

        if (os.path.isfile(os.path.join(model_path, "optimizer.pt")) and
                os.path.isfile(os.path.join(model_path, "scheduler.pt"))):
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        try:
            self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
            flos_cfg = model.module.config if self.args.n_gpu > 1 else model.config
            self.total_flos = getattr(flos_cfg, "total_flos", 0)
            epochs_trained = self.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.global_step % num_update_steps_per_epoch

            def _log():
                logger.info("  Resuming from checkpoint: epoch=%d global_step=%d skip_steps=%d",
                            epochs_trained, self.global_step, steps_trained_in_current_epoch)
            if self.args.deepspeed:
                if torch.distributed.get_rank() == 0:
                    _log()
            else:
                _log()
        except ValueError:
            self.global_step = self.total_flos = 0
            logger.info("  Starting fine-tuning.")

        return epochs_trained, steps_trained_in_current_epoch

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def time_wrap(self, time_list=None):
        if time_list is not None:
            time_list.append(torch.cuda.Event(enable_timing=True))
            time_list[-1].record()

    def print_time_list(self, text, start_list=None, end_list=None, init_index=0):
        times_list = [s.elapsed_time(e) for s, e in zip(start_list, end_list)]
        print(f"Device{self.args.local_rank}: {text} Raw Time: {times_list}")
        avg_time = sum(times_list[init_index:]) / len(times_list[init_index:])
        return avg_time

    def _report_benchmark(self, model):
        """Print per-component timing breakdown after a benchmarked training run."""
        torch.cuda.synchronize()
        # skip_microbatch / skip_step: warmup iterations to exclude from averaging.
        # e.g.) skip_microbatch (min_separation * grad_accum_steps) & skip_step (min_separation).
        skip_microbatch = 0
        skip_step = 0

        # Per-microbatch timings.
        iteration    = self.print_time_list("iter",        model.start_iter,        model.end_iter,          skip_microbatch)
        before_fwd   = self.print_time_list("before_fwd",  model.start_iter,        model.start_forward,     skip_microbatch)
        forward      = self.print_time_list("forward",     model.start_forward,     model.end_forward,       skip_microbatch)
        loss         = self.print_time_list("loss",        model.start_loss,        model.end_loss,          skip_microbatch)
        backward     = self.print_time_list("backward",    model.end_loss,          model.end_backward,      skip_microbatch)
        deep_step    = self.print_time_list("deep_step",   model.end_backward,      model.deep_end,          skip_microbatch)
        # Per-optimizer-step timings.
        step_by_step = self.print_time_list("sbs",         model.end_step[:-1],     model.end_step[1:],      skip_step)
        optim_step   = self.print_time_list("optim_step",  model.start_step,        model.end_step,          skip_step)

        if not self.privacy_args.non_private:
            clip = self.print_time_list("clip", model.start_clip, model.end_clip, skip_microbatch)
            if self.args.deepspeed_config:
                ngen    = self.print_time_list("ngen",    self.optimizer.start_noise_gen,  self.optimizer.end_noise_gen,    skip_step)
                nupdate = self.print_time_list("nupdate", self.optimizer.end_noise_gen,    self.optimizer.end_grad_update,  skip_step)
                gather  = self.print_time_list("gather",  self.optimizer.end_grad_update,  self.optimizer.end_gather,       skip_step)
                if self.CPU_time:
                    _avg_cpu = sum(self.CPU_time) / len(self.CPU_time) * 1000
                    _n = self.privacy_args.CPU_partition
                    print(f"CPU×{_n}: total/step={_avg_cpu:.1f}ms  unit/part={_avg_cpu/_n if _n else 0:.1f}ms", self.CPU_time)
                if self.CXL_time:
                    _avg_cxl = sum(self.CXL_time) / len(self.CXL_time) * 1000
                    _n = self.privacy_args.noise_partition - self.privacy_args.GPU_partition - self.privacy_args.CPU_partition
                    print(f"NMP×{_n}: total/step={_avg_cxl:.1f}ms  unit/part={_avg_cxl/_n if _n else 0:.1f}ms", self.CXL_time)
                if self.GPU_time:
                    _avg_gpu = sum(self.GPU_time) / len(self.GPU_time) * 1000
                    print(f"GPU_TIME (compute+transfer) avg={_avg_gpu:.1f}ms", self.GPU_time)
                if self.GPU_transfer_time:
                    print("GPU_TRANSFER_TIME", sum(self.GPU_transfer_time) / len(self.GPU_transfer_time), self.GPU_transfer_time)
                for _label, _val in [
                    ("step_by_step",                          step_by_step),
                    ("iter",                                  iteration),
                    ("forward",                               forward),
                    ("loss",                                  loss),
                    ("backward",                              backward - clip),
                    ("clip",                                  clip),
                    ("optim_step",                            optim_step),
                    (f"ngen(GPU×{self.privacy_args.GPU_partition})", ngen),
                    ("nupdate",                               nupdate),
                    ("gather",                                gather),
                    ("deep_step",                             deep_step),
                    ("before_fwd",                            before_fwd),
                ]:
                    print(f"{_label:<16}: {_val}")
            else:
                ngen  = self.print_time_list("ngen",  self.optimizer.privacy_engine.start_noise_gen,  self.optimizer.privacy_engine.end_noise_gen,   skip_step)
                nadd  = self.print_time_list("nadd",  self.optimizer.privacy_engine.end_noise_gen,    self.optimizer.privacy_engine.end_grad_update, skip_step)
                for _label, _val in [
                    ("step_by_step",  step_by_step),
                    ("iter",          iteration),
                    ("forward",       forward),
                    ("loss",          loss),
                    ("backward",      backward - clip),
                    ("clip",          clip),
                    ("optim_step",    optim_step),
                    ("ngen",          ngen),
                    ("nadd",          nadd),
                ]:
                    print(f"{_label:<16}: {_val}")
        else:
            for _label, _val in [
                ("step_by_step",  step_by_step),
                ("iter",          iteration),
                ("forward",       forward),
                ("loss",          loss),
                ("backward",      backward),
                ("optim_step",    optim_step),
            ]:
                print(f"{_label:<16}: {_val}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, model_path: Optional[str] = None, benchmark=None, bandmf_solver=None, **kwargs):
        """
        Main training entry point.

        Args:
            model_path: Local checkpoint path to resume from.
        """

        if self.model_init is not None:
            set_seed(self.args.seed)
            self.model = self.model_init().to(self.args.device)
            self.optimizer, self.lr_scheduler = None, None
        else:
            self.model.to(self.args.device)

        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = max(
            len(train_dataloader) // self.args.gradient_accumulation_steps, 1
        )
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        part_GPU = part_CPU = part_NMP = 0
        noise_process = stop_event = None
        if self.args.deepspeed_config:
            part_GPU, part_CPU, part_NMP, noise_process, stop_event = \
                self._setup_deepspeed_training(bandmf_solver, t_total)
        else:
            logger.info('deepspeed = False')

        model = self.model
        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained, steps_trained_in_current_epoch = self._resume_from_checkpoint(
            model_path, model, num_update_steps_per_epoch
        )

        if self.args.evaluate_before_training:
            self.evaluate(epoch=0)

        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.warning("***** Running training *****")
        if self.args.deepspeed_config:
            logger.warning("  torch.distributed.get_world_size() = %d", torch.distributed.get_world_size())
        logger.warning("  Num examples = %d", self.num_examples(train_dataloader))
        logger.warning("  Num Epochs = %d", num_train_epochs)
        logger.warning("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.warning("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.warning("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.warning("  Total optimization steps = %d", t_total)

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        start_one_epoch = torch.cuda.Event(enable_timing=True)
        start_one_epoch.record()
        part_size = int(self.num_params/ (self.args.num_GPUs * self.optimizer.partition))
        unit_size = int(self.partition_size / self.optimizer.partition) 
        self.CPU_time = []
        self.CXL_time = []
        self.GPU_time = []          # total GPU blocking time (compute + transfer) for GEMV_at_GPU
        self.GPU_transfer_time = [] # transfer-only portion; populated only when CPU/NMP_MODE==GEMV_at_GPU

        def cpu_noise_cpu(CPU_time, local_rank: int, gpu_slices: int, cpu_slices: int):
            """CPU-side noise thread: handles cpu_slices via the noise_worker subprocess queue."""
            torch.cuda.set_device(local_rank)
            stime1 = time.perf_counter()
            print(f"[DEBUG cpu_thread rank={local_rank}] started gpu_slices={gpu_slices} cpu_slices={cpu_slices}", flush=True)
            with torch.cuda.stream(cpu_stream):
                for j in range(gpu_slices, gpu_slices + cpu_slices):
                    torch.cuda.memory._set_allocator_settings("expandable_segments:False")
                    new_noise = torch.normal(mean=0,
                        std=self.optimizer.noise_multiplier * self.optimizer.per_example_max_grad_norm,
                        size=(unit_size,), device=f"cuda:{local_rank}",
                    )
                    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
                    print(f"[DEBUG cpu_thread rank={local_rank}] putting raw_noise j={j} to raw_noise_queue", flush=True)
                    # Use timeout on put() so a stale item from a previous aborted thread doesn't block forever.
                    self.raw_noise_queue.put(new_noise.to(f'cpu', non_blocking=True))
                    while self.raw_noise_queue.full():
                        time.sleep(0.01)
                    print(f"[DEBUG cpu_thread rank={local_rank}] waiting noise_queue.get() j={j}", flush=True)
                    while True:
                        try:
                            noise = self.noise_queue.get(timeout=5.0)
                            break
                        except Exception:
                            if not noise_process.is_alive():
                                ec = noise_process.exitcode
                                raise RuntimeError(
                                    f"[cpu_thread rank={local_rank}] noise_worker died "
                                    f"(exitcode={ec}, likely OOM-killed) while waiting for j={j}"
                                )
                    noise = noise.to(f"cuda:{local_rank}", non_blocking=True)
                    print(f"[DEBUG cpu_thread rank={local_rank}] got noise j={j}", flush=True)
                    self.optimizer.noise_placeholder[j * unit_size:(j + 1) * unit_size] = noise
            print(f"[DEBUG cpu_thread rank={local_rank}] done, elapsed={time.perf_counter()-stime1:.2f}s", flush=True)
            CPU_time.append(time.perf_counter() - stime1)

        def cxl_noise_cxl(CXL_time, local_rank: int, start_idx: int, slices: int, band: int):
            """Emulate NMP-side GEMV: analytically sleeps for CXL↔GPU transfer + NMP compute."""
            torch.cuda.set_device(local_rank)
            stime1 = time.perf_counter()
            total_transfer = 0
            with torch.cuda.stream(cxl_stream):
                for j in range(slices):
                    new_noise = torch.normal(mean=0,
                        std=self.optimizer.noise_multiplier * self.optimizer.per_example_max_grad_norm,
                        size=(unit_size,), device=f"cuda:{local_rank}",
                    )
                    # NMP computes GEMV: band history vectors × unit_size (band is power-of-2)
                    t_CXL_compute = unit_size * band * 4 / THROUGHPUT_NMP
                    # NMP sends GEMV result to GPU
                    t_CXL2GPU    = unit_size * 4 / BW_CXLGPU
                    # GPU: correlated_noise = raw_gaussian - GEMV_result  (BandMF recurrence)
                    new_noise = new_noise - new_noise
                    # GPU sends correlated noise back to NMP to update history buffer
                    t_GPU2CXL    = unit_size * 4 / BW_CXLGPU
                    total_transfer += t_CXL2GPU + t_GPU2CXL
                    time.sleep(t_CXL2GPU + t_CXL_compute + t_GPU2CXL)
                    s = (start_idx + j) * unit_size
                    self.optimizer.noise_placeholder[s:s + unit_size] = new_noise
            CXL_time.append(time.perf_counter() - stime1)

        def gpu_noise_sleep(GPU_time, GPU_transfer_time, local_rank: int, cpu_on_gpu: int, nmp_on_gpu: int):
            """Block the GPU critical path for GEMV_at_GPU mode.
            Compute portion: interpolated from last step's GPU noise-gen CUDA events (per-slice).
            Transfer portion: NOT hidden (GPU compute is fast), modeled via hardware constants.
              - CPU->GPU: BW_CPUGPU per cpu_on_gpu slice
              - CXL->GPU: BW_CXLGPU per nmp_on_gpu slice
            Both components are recorded separately for benchmark reporting.
            """
            if not self.optimizer.start_noise_gen:
                return
            t_gpu_step_ms = self.optimizer.start_noise_gen[-1].elapsed_time(self.optimizer.end_noise_gen[-1])
            t_compute = t_gpu_step_ms / 1000.0 / part_GPU * (cpu_on_gpu + nmp_on_gpu)
            t_transfer = (cpu_on_gpu * unit_size * 4 / BW_CPUGPU +
                          nmp_on_gpu * unit_size * 4 / BW_CXLGPU)
            stime = time.perf_counter()
            time.sleep(t_compute + t_transfer)
            GPU_time.append(time.perf_counter() - stime)
            GPU_transfer_time.append(t_transfer)
        
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            # This extra step is crucial. The problem is that the total number of steps in one epoch might
            # not divide the number of accumulation steps, thus the accumulated .summed_grad (.grad) might overflow to
            # the next epoch, causing more gradient signal than there truly is.
            if self.args.deepspeed_config:
                model.zero_grad()
            else:
                model.zero_grad(set_to_none=True)

            if model.benchmark:
                for attr in ('start_iter', 'end_iter', 'start_forward', 'end_forward',
                             'start_loss', 'end_loss', 'start_backward', 'end_backward',
                             'start_step', 'end_step', 'deep_start', 'deep_end'):
                    setattr(model, attr, [])
                if not self.args.deepspeed_config and not self.privacy_args.non_private:
                    pe = self.optimizer.privacy_engine
                    pe.benchmark = True
                    for attr in ('start_noise_gen', 'end_noise_gen', 'end_noise_add'):
                        setattr(pe, attr, [])

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            main_stream = torch.cuda.current_stream()
            cpu_stream = torch.cuda.Stream()
            cxl_stream = torch.cuda.Stream()
            cpu_thread = nmp_thread = None
            if part_CPU > 0 and CPU_MODE == "GEMV_at_CPU":
                cpu_thread = threading.Thread(target=cpu_noise_cpu,
                    args=(self.CPU_time, self.args.local_rank, part_GPU, part_CPU))
                cpu_thread.start()
            if part_NMP > 0 and NMP_MODE == "GEMV_at_NMP":
                nmp_thread = threading.Thread(target=cxl_noise_cxl,
                    args=(self.CXL_time, self.args.local_rank, part_GPU + part_CPU, part_NMP, self.privacy_args.min_separation))
                nmp_thread.start()
            # get_current_memory_mb()
            for step, inputs in enumerate(epoch_iterator):
                if model.benchmark:
                    self.time_wrap(model.start_iter)
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue
                model.train()
                inputs = self._prepare_inputs(inputs)
                ### sum of loss dividing micro-batch size, not batch size
                labels = inputs.pop('labels')
                if model.benchmark:
                    self.time_wrap(model.start_forward)
                outputs = model(**inputs)
                if model.benchmark:
                    self.time_wrap(model.end_forward)
                # Save past state if it exists
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index]
                if model.benchmark:
                    self.time_wrap(model.start_loss)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                seq_lens = (shift_labels != -50).sum(dim=1)
                loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels)
                if model.benchmark:
                    self.time_wrap(model.end_loss)

                if self.args.deepspeed_config:
                    model.backward(loss)
                else:
                    loss.backward()
                if model.benchmark:
                    self.time_wrap(model.end_backward)
                tr_loss += loss.detach()
                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    self.args.gradient_accumulation_steps >= len(epoch_iterator) == (step + 1)):
                    if self.privacy_args.min_separation > 1:
                        if cpu_thread:
                            print(f"[DEBUG main rank={self.args.local_rank}] step={step} joining cpu_thread", flush=True)
                            cpu_thread.join()
                            print(f"[DEBUG main rank={self.args.local_rank}] step={step} cpu_thread joined", flush=True)
                        if nmp_thread:
                            nmp_thread.join()
                        # NMP-at-CPU: extend CPU time proportionally for NMP slices.
                        # Uses last measured per-CPU-slice time; transfer assumed hidden.
                        if NMP_MODE == "GEMV_at_CPU" and part_NMP > 0 and self.CPU_time:
                            time.sleep(self.CPU_time[-1] / part_CPU * part_NMP)

                if self.args.deepspeed_config:
                    print(f"[DEBUG main rank={self.args.local_rank}] step={step} calling model.step()", flush=True)
                    model.step()
                    print(f"[DEBUG main rank={self.args.local_rank}] step={step} model.step() done", flush=True)
                # GPU-at-GPU: block critical path using last step's GPU noise-gen CUDA event timing.
                if self.privacy_args.min_separation > 1:
                    cpu_on_gpu = part_CPU if CPU_MODE == "GEMV_at_GPU" else 0
                    nmp_on_gpu = part_NMP if NMP_MODE == "GEMV_at_GPU" else 0
                    if cpu_on_gpu + nmp_on_gpu > 0:
                        gpu_noise_sleep(self.GPU_time, self.GPU_transfer_time, self.args.local_rank, cpu_on_gpu, nmp_on_gpu)
                if model.benchmark:
                    self.time_wrap(model.deep_end)

                # https://github.com/microsoft/DeepSpeed/issues/758
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    self.args.gradient_accumulation_steps >= len(epoch_iterator) == (step + 1)
                ):
                    if self.privacy_args.noise_offload and self.privacy_args.min_separation > 1:
                        cpu_thread = nmp_thread = None
                        # Don't prefetch noise for a step that won't exist (last optimizer step).
                        _next_step_exists = not (self.args.max_steps > 0 and self.global_step + 1 >= self.args.max_steps)
                        print(f"[DEBUG main rank={self.args.local_rank}] global_step={self.global_step} next_step_exists={_next_step_exists}", flush=True)
                        if _next_step_exists and part_CPU > 0 and CPU_MODE == "GEMV_at_CPU":
                            cpu_thread = threading.Thread(target=cpu_noise_cpu,
                                args=(self.CPU_time, self.args.local_rank, part_GPU, part_CPU))
                            cpu_thread.start()
                        if _next_step_exists and part_NMP > 0 and NMP_MODE == "GEMV_at_NMP":
                            nmp_thread = threading.Thread(target=cxl_noise_cxl,
                                args=(self.CXL_time, self.args.local_rank, part_GPU + part_CPU, part_NMP, self.privacy_args.min_separation))
                            nmp_thread.start()
                    if model.benchmark:
                        self.time_wrap(model.start_step)
                    if self.privacy_args.non_private:
                        # Don't double clip in private learning.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    if self.args.deepspeed_config:
                        pass
                    else:
                        self.optimizer.step()
                        model.zero_grad(set_to_none=True)

                    self.lr_scheduler.step()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)
                    if (
                        self.args.evaluation_strategy in (EvaluationStrategy.STEPS, IntervalStrategy.STEPS)
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        self.evaluate(epoch=epoch)

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            self.model = model.module
                        else:
                            self.model = model

                        # Save model checkpoint
                        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                        self.store_flos()
                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints(use_mtime=True)
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    if model.benchmark:
                        self.time_wrap(model.end_step)
                if model.benchmark:
                    self.time_wrap(model.end_iter)
                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            show_memory(self.args.local_rank)

            if epoch == 0:
                break
            epoch_pbar.close()
            train_pbar.update(1)

            if (
                self.args.evaluation_strategy in (EvaluationStrategy.EPOCH, IntervalStrategy.EPOCH) and
                (epoch + 1) % self.args.eval_epochs == 0
            ):
                metrics = self.evaluate(epoch=epoch)

            if self.args.max_steps is not None and 0 < self.args.max_steps <= self.global_step:
                break
            
        if model.benchmark:
            self._report_benchmark(model)
        train_pbar.close()

        if stop_event is not None:
            stop_event.set()
        if noise_process is not None:
            noise_process.terminate()
            noise_process.join()
        # Safety join: if a prefetch thread is still alive (e.g. epoch-end exit path), clean it up.
        for _t in (cpu_thread, nmp_thread):
            if _t is not None and _t.is_alive():
                print(f"[DEBUG cleanup] joining leftover thread {_t.name}", flush=True)
                _t.join(timeout=5)

        if self.args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.total_flos is not None:
            if self.args.local_rank != -1:
                total_flos = distributed_broadcast_scalars([self.total_flos]).sum().item()
            else:
                total_flos = self.total_flos
            if total_flos > 0:
                logs["total_flos"] = self.total_flos
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        output = {
            **logs,
            **{
                "step": self.global_step,
                'num_params': self.num_params,
                'num_non_embedding_params': self.num_non_embedding_params
            }
        }
        if self.is_world_process_zero():
            self.log_history.append(output)
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def is_local_process_zero(self) -> bool:
        """Whether or not this process is the local main process."""
        return self.args.local_rank in [-1, 0]

    def is_world_process_zero(self) -> bool:
        """Whether or not this process is the global main process."""
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    # Alias kept for call sites in other scripts.
    is_world_master = is_world_process_zero

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        if self.total_flos is not None:
            if self.args.local_rank != -1:
                total_flos = distributed_broadcast_scalars([self.total_flos]).sum().item()
            else:
                total_flos = self.total_flos
            if total_flos > 0:
                self.model.config.total_flos = total_flos

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_process_zero():
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)  # Find the models in `train_dir/checkpoint-k/pytorch_model.bin`
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        json.dump(
            self.log_history, open(os.path.join(output_dir, "log_history.json"), "w"), indent=4, ensure_ascii=False
        )

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        output_dir_name = os.path.basename(self.args.output_dir)
        checkpoint_prefix = f"{output_dir_name}-{PREFIX_CHECKPOINT_DIR}"

        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    # ------------------------------------------------------------------
    # Evaluation & Generation
    # ------------------------------------------------------------------

    def evaluate(self, log_results=True, epoch=None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            log_results:
                Store the results in `self.log_history` and print to stdout.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        eval_output = self.prediction_loop(eval_dataloader, description="Evaluate eval split")

        val_dataloader = self.get_eval_dataloader(self.val_dataset)
        val_output = self.prediction_loop(val_dataloader, description="Evaluate val split")

        train_sampler = self._get_train_sampler(shuffle=False)  # Don't shuffle during evaluation!
        train_dataloader = self.get_train_dataloader(train_sampler=train_sampler)
        train_output = self.prediction_loop(train_dataloader, description="Evaluate train split")

        metrics = {
            "train": train_output.metrics,
            "eval": eval_output.metrics,
            "val": val_output.metrics,
            "epoch": epoch,
            "lr": [pg["lr"] for pg in self.optimizer.param_groups],
        }

        if hasattr(self.optimizer, 'privacy_engine'):
            pe = self.optimizer.privacy_engine
            privacy_metrics = pe.get_privacy_spent(accounting_mode="all", lenient=True)
            privacy_stats = pe.get_training_stats()
            metrics = {**metrics, **privacy_metrics, **privacy_stats}

        # Generate with beam search.
        if not self.args.skip_generation:
            self.generate_and_write_to_file()

        if log_results:
            self.log(metrics)

            # Save log history always! This must appear after the `log_history` is updated.
            json.dump(
                self.log_history,
                open(os.path.join(self.args.output_dir, "log_history.json"), "w"),
                indent=2,
                ensure_ascii=False
            )

        return metrics

    def _get_loader_by_split(self, split):
        if split == "train":
            loader = self.get_train_dataloader()
        else:
            if split == "val":
                loader = self.get_eval_dataloader(self.val_dataset)
            elif split == "eval":
                loader = self.get_eval_dataloader(self.eval_dataset)
            else:
                raise ValueError(f"Unknown split: {split}")
        return loader

    def _get_prompt_dataset_by_split(self, split):
        return {
            "train": self.generation_stuff["train_prompts"],
            "val": self.generation_stuff["val_prompts"],
            "eval": self.generation_stuff["eval_prompts"],
        }[split]

    def generate_and_write_to_file(self, num_generations_to_print=6, **decoding_kwargs):
        # Pass in the additional decoding stuff from `decoding_kwargs`.

        models = (self.model,)
        model_tags = ("model",)
        all_generations = {model_tag: {} for model_tag in model_tags}

        for this_model, this_model_tag in utils.zip_(models, model_tags):
            kwargs = dict(model=this_model, tokenizer=self.tokenizer, device=self.args.device)
            this_generations = all_generations[this_model_tag]

            for split in ("train", "val", "eval"):
                # Don't use the loader to avoid duplicated prompts!
                prompt_dataset = self._get_prompt_dataset_by_split(split)
                if split == "train":  # Don't waste compute on sanity checks.
                    max_generations = self.args.max_generations_train
                    continue
                elif split in ('val', 'valid'):  # Use val and valid interchangeably.
                    max_generations = self.args.max_generations_valid
                    continue
                else:
                    max_generations = self.args.max_generations

                full_generations, unstripped_generations, generations, references = decoding_utils.generate(
                    prompt_dataset=prompt_dataset, max_generations=max_generations,
                    **kwargs, **decoding_kwargs
                )
                this_generations[split] = dict(
                    full_generations=full_generations,
                    unstripped_generations=unstripped_generations,
                    generations=generations,
                    references=references,
                )

                def pretty_format(lines):
                    """A useful helper to make printted generationed look nice."""
                    return '\n'.join([repr(line) for line in lines[:num_generations_to_print]])

                # Various visuals.
                print(f" --- split {split} --- ")
                print(f" *** full generations *** ")
                print(pretty_format(full_generations))
                print(f" *** unstripped generations *** ")
                print(pretty_format(unstripped_generations))
                print(f" *** generations *** ")
                print(pretty_format(generations))
                print(f" *** references *** ")
                print(pretty_format(references))
                print(f" *** num generations: {len(generations)}, num references: {len(references)} *** ")

                # Store generations for BLEU.
                counter = self.global_step if self.global_step is not None else -1
                generations_path = os.path.join(
                    self.args.output_dir,
                    f'generations_{this_model_tag}', f'{split}', f'global_step_{counter:08d}.txt'
                )
                os.makedirs(os.path.dirname(generations_path), exist_ok=True)
                with open(generations_path, 'w') as f:
                    f.writelines([line + '\n' for line in generations])
                logger.warning(f"Wrote generations to {generations_path}")

    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        self.model.eval()
        models = (self.model,)
        model_tags = ("model",)

        def create_record():
            return dict(
                eval_losses=[], entropy_losses=[], tok_logprobs=[], lin_logprobs=[],
            )

        records = {model_tag: create_record() for model_tag in model_tags}
        preds = label_ids = None

        if self.args.past_index >= 0:
            self._past = None

        def eval_stats(inputs, loss, logits, labels):
            if loss is not None:
                batch_size = inputs['input_ids'].size(0)
                eval_loss = [loss] * batch_size
            else:
                eval_loss = [-1]

            if logits is not None:
                logits = logits[..., :-1, :]
                labels = labels[..., 1:]

                valid_locations = (labels != -100)
                all_log_probs = logits.log_softmax(dim=-1)  # (B, L, V).
                entropy = -(all_log_probs.exp() * all_log_probs).sum(dim=-1)  # (B, L).
                entropy = entropy[valid_locations]

                logprob = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none")  # (B, L).
            else:
                entropy, logprob = [-1], [-1]

            return eval_loss, entropy, logprob

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        for batch_idx, inputs in tqdm(enumerate(dataloader), desc=description, disable=disable_tqdm):
            for this_model, this_model_tag in utils.zip_(models, model_tags):
                this_record = records[this_model_tag]
                loss, logits, labels = self.prediction_step(this_model, inputs, prediction_loss_only)
                eval_loss, entropy, logprob = eval_stats(inputs, loss, logits, labels)
                this_record["eval_losses"].extend(eval_loss)
                this_record["entropy_losses"].extend(entropy.tolist())
                this_record["tok_logprobs"].extend(logprob.view(-1).tolist())
                this_record["lin_logprobs"].extend(logprob.sum(dim=-1).view(-1).tolist())

            if 0 < self.args.max_eval_batches <= batch_idx + 1:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        for record_key, record_value in records.items():
            this_record = records[record_key]
            for key, value in this_record.items():
                if isinstance(value, (list, tuple)):
                    this_record[key] = np.mean(value)

        metrics = records

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            if has_labels:  # The .mean() is to reduce in case of distributed training
                loss = loss.mean().item()
            logits = outputs.logits

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return loss, logits, labels

