"""
DP noise injection for DeepSpeed ZeRO stage 1.

Usage
-----
Call `apply_dp_patch()` once before `deepspeed.initialize()`, then call
`setup_dp_optimizer()` after `deepspeed.initialize()` to wire the required
attributes onto the optimizer instance.
"""

import torch


def apply_dp_patch():
    """Patch DeepSpeedZeroOptimizer.reduce_gradients and .step for DP."""
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        from deepspeed import comm as dist
        from deepspeed.runtime.utils import see_memory_usage, all_gather_dp_groups
    except ImportError:
        return

    def _dp_reduce_gradients(self, pipeline_parallel=False):
        """Replace DeepSpeed reduce_gradients: move private_grad -> grad before all-reduce."""
        if pipeline_parallel and self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=torch.cuda.current_device())
            self.ipg_buffer.append(buf_0)
            self.ipg_index = 0
        if not self.overlap_comm:
            for i, group in enumerate(self.bit16_groups):
                for param in group:
                    if hasattr(param, 'private_grad'):
                        param.grad = torch.nan_to_num(param.private_grad).contiguous()
                        del param.private_grad
                        param.grad = param.grad / param.batch_size * self.loss_scale
                    else:
                        param.grad.zero_()
                    self.reduce_ready_partitions_and_remove_grads(param, i)
        self.overlapping_partition_gradients_reduce_epilogue()

    def _dp_step(self, closure=None):
        """Replacement for DeepSpeedZeroOptimizer.step — injects BandMF correlated noise."""
        self.micro_step_id = -1

        see_memory_usage(f"In step before checking overflow")

        self.check_overflow()
        OPTIMIZER_ALLGATHER = 'optimizer_allgather'
        OPTIMIZER_GRADIENTS = 'optimizer_gradients'
        OPTIMIZER_STEP = 'optimizer_step'
        timer_names = [OPTIMIZER_ALLGATHER, OPTIMIZER_GRADIENTS, OPTIMIZER_STEP]

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}
            see_memory_usage('After overflow after clearing gradients')
            self.start_timers(timer_names)
            self.stop_timers(timer_names)
            return

        # Step 1: Calculate gradient norm using fp-16 grads
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale

        see_memory_usage('After norm before optimizer')
        # Step 2: Run optimizer and upscaling simultaneously
        for i, group in enumerate(self.bit16_groups):
            self.start_timers([OPTIMIZER_GRADIENTS])
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
                self.stop_timers([OPTIMIZER_GRADIENTS])
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)

                from deepspeed.ops.adam import DeepSpeedCPUAdam
                if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
                    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                    fp32_partition = self.single_partition_of_fp32_groups[i]
                    bit16_partitions[partition_id].data.copy_(fp32_partition.data)

                self.stop_timers([OPTIMIZER_STEP])
            else:
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])
                
                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    self.noise_placeholder += self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])).to(self.single_partition_of_fp32_groups[i].dtype)
                else:
                    self.noise_placeholder += self.flatten(self.averaged_gradients[i]).to(
                        self.single_partition_of_fp32_groups[i].dtype)
                assert self.noise_placeholder.numel() == self.partition_size[i], \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        self.noise_placeholder.numel(), self.partition_size[i], i, partition_id)

                if self.noise_offload and self.bandmf_solver.bandwidth > 1:
                    if hasattr(self, 'start_noise_gen'):
                        self.start_noise_gen.append(torch.cuda.Event(enable_timing=True))
                        self.start_noise_gen[-1].record()
                    if self.part_GPU > 0:
                        part_size = int(self.partition_size[i] / self.partition)
                        noise_type = self.single_partition_of_fp32_groups[0].dtype
                        div = self.bandmf_solver.diag(self.bandmf_solver.current_step)
                        for j in range(self.part_GPU):
                            new_noise = torch.normal(mean=0,
                                    std=self.noise_multiplier * self.per_example_max_grad_norm / div,
                                    size=(part_size,),
                                    device=f"cuda:{partition_id}",
                                    dtype=noise_type,
                            )
                            # = not +=: offload path; GPU slices hold pure noise (CPU noise pre-assembled in noise_queue)
                            self.noise_placeholder[j * part_size:(j+1) * part_size] = self.bandmf_solver.step(new_noise, j)
                        self.bandmf_solver.advance()

                    if hasattr(self, 'end_noise_gen'):
                        self.end_noise_gen.append(torch.cuda.Event(enable_timing=True))
                        self.end_noise_gen[-1].record()
                else:
                    if hasattr(self, 'start_noise_gen'):
                        self.start_noise_gen.append(torch.cuda.Event(enable_timing=True))
                        self.start_noise_gen[-1].record()
                    noise_type = self.single_partition_of_fp32_groups[0].dtype
                    if self.bandmf_solver is not None and self.bandmf_solver.bandwidth > 1:
                        div = self.bandmf_solver.diag(self.bandmf_solver.current_step)
                    else:
                        div = 1
                    part_size = int(self.partition_size[i] / self.partition)
                    if self.bandmf_solver.bandwidth > 1:
                        for j in range(self.partition):
                            new_noise = torch.normal(mean=0,
                                    std=self.noise_multiplier * self.per_example_max_grad_norm / div,
                                    size=(part_size,),
                                    device=f"cuda:{partition_id}",
                                    dtype=noise_type,
                            )
                            # += : noise is added on top of accumulated gradient in noise_placeholder
                            self.noise_placeholder[j * part_size:(j+1) * part_size] += self.bandmf_solver.step(new_noise, j) / (self.train_batch_size * self.loss_scale)
                        self.bandmf_solver.advance()
                    else:
                        for j in range(self.partition):
                            new_noise = torch.normal(mean=0,
                                    std=self.noise_multiplier * self.per_example_max_grad_norm / div,
                                    size=(part_size,),
                                    device=f"cuda:{partition_id}",
                                    dtype=noise_type,
                            )
                            self.noise_placeholder[j * part_size:(j+1) * part_size] += new_noise / (self.train_batch_size * self.loss_scale)
                    if hasattr(self, 'end_noise_gen'):
                        self.end_noise_gen.append(torch.cuda.Event(enable_timing=True))
                        self.end_noise_gen[-1].record()

                self.single_partition_of_fp32_groups[i].grad = self.noise_placeholder
                # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])
                self.averaged_gradients[i] = None

                self.unscale_and_clip_grads([self.noise_placeholder], scaled_global_grad_norm)
                self.stop_timers([OPTIMIZER_GRADIENTS])

                # Step 3:- run the optimizer if no offloading
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)
                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                self.single_partition_of_fp32_groups[i].grad = None
                self.noise_placeholder.zero_()
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.stop_timers([OPTIMIZER_STEP])

        if hasattr(self, 'end_grad_update'):
            self.end_grad_update.append(torch.cuda.Event(enable_timing=True))
            self.end_grad_update[-1].record()
        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.start_timers([OPTIMIZER_ALLGATHER])
        all_gather_dp_groups(
            partitioned_param_groups=self.parallel_partitioned_bit16_groups,
            dp_process_group=self.real_dp_process_group,
            start_alignment_factor=self.nccl_start_alignment_factor,
            allgather_bucket_size=self.allgather_bucket_size)
        self.stop_timers([OPTIMIZER_ALLGATHER])

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.log_timers(timer_names)
        see_memory_usage('After zero_optimizer step')
        if hasattr(self, 'end_gather'):
            self.end_gather.append(torch.cuda.Event(enable_timing=True))
            self.end_gather[-1].record()

        return

    DeepSpeedZeroOptimizer.reduce_gradients = _dp_reduce_gradients
    DeepSpeedZeroOptimizer.step = _dp_step


def setup_dp_optimizer(optimizer, *, bandmf_solver, noise_multiplier,
                        per_example_max_grad_norm, train_batch_size, partition,
                        noise_offload, partition_size, local_rank, part_GPU=0):
    """Attach DP state to a DeepSpeedZeroOptimizer after deepspeed.initialize().

    part_GPU: GPU-side BandMF partitions; solver partition is capped to this when offloading,
    to avoid unbounded _buf growth (one new (bandwidth-1, unit_size) tensor per step otherwise).
    """
    gpu_solver_partition = part_GPU if (noise_offload and part_GPU > 0) else partition
    bandmf_solver.set_partition(gpu_solver_partition)
    print(f"[DEBUG setup_dp_optimizer rank={local_rank}] "
          f"noise_offload={noise_offload} part_GPU={part_GPU} total_partition={partition} "
          f"gpu_solver_partition={gpu_solver_partition}", flush=True)
    for attr, val in (
        ('bandmf_solver',              bandmf_solver),
        ('noise_multiplier',           noise_multiplier),
        ('per_example_max_grad_norm',  per_example_max_grad_norm),
        ('train_batch_size',           train_batch_size),
        ('partition',                  partition),
        ('noise_offload',              noise_offload),
        ('part_GPU',                   part_GPU),
        ('noise_placeholder',          torch.zeros(partition_size, device=f'cuda:{local_rank}')),
    ):
        setattr(optimizer, attr, val)
    # Benchmark timing lists; events are recorded when these lists exist (see _dp_step).
    for attr in ('start_noise_gen', 'end_noise_gen', 'end_noise_comp', 'end_grad_update', 'end_gather'):
        setattr(optimizer, attr, [])
