'''Train with ImageNet-sized dummy data (scale: (224,224,3), 50k samples, 1000 classes).'''
def _tw(lst):
    """Append a new CUDA timing event to lst and record it."""
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    lst.append(e)

def _avg_ms(starts, ends, skip=0):
    ts = [s.elapsed_time(e) for s, e in zip(starts[skip:], ends[skip:])]
    return sum(ts) / len(ts) if ts else 0.0

def main(args):
    config = json.load(open(args.deepspeed_config))

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Data: FakeData with CIFAR10 scale but ImageNet resolution
    print('==> Preparing data..')
    trainset = torchvision.datasets.FakeData(50000, (3, 224, 224), 1000, transform=transformation)

    # Model
    print('==> Building and fixing model..', args.model, '. Mode: ', args.clipping_mode)
    net = timm.create_model(args.model, pretrained=False, num_classes=1000)
    net = ModuleValidator.fix(net)

    criterion = nn.CrossEntropyLoss()

    if 'BiTFiT' in args.clipping_mode:
        for name, param in net.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))

    # BandMF solver
    bandmf_solver = None
    n_gpu = torch.distributed.get_world_size()
    nbatches = args.num_batches if args.num_batches > 0 else (len(trainset) // config['train_batch_size'])
    if args.min_separation > 1:
        from fastDP.bandmf import BandedMatrixFactorizationMechanism
        bandmf_solver = BandedMatrixFactorizationMechanism(
            num_iterations=nbatches * args.epochs,
            min_separation=args.min_separation,
            objective='sum',
            workload_matrix_type='A',
            noise_offload=args.noise_offload,
            device_num=torch.distributed.get_rank() % torch.cuda.device_count(),
        )

    # Privacy engine
    non_private = 'nonDP' in args.clipping_mode
    privacy_engine = None
    if not non_private:
        privacy_engine = PrivacyEngine(
            net,
            batch_size=config['train_batch_size'],
            sample_size=len(trainset),
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            clipping_mode='MixOpt',
            clipping_style=args.clipping_style,
            num_GPUs=n_gpu,
            torch_seed_is_fixed=True,
            bandmf_solver=bandmf_solver,
            noise_offload=args.noise_offload,
        )

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if bandmf_solver is not None:
        from fastDP.deepspeed_patch import apply_dp_patch, setup_dp_optimizer
        apply_dp_patch()

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, optimizer=optimizer,
        model_parameters=net.parameters(), training_data=trainset)

    fp16 = model_engine.fp16_enabled(); bf16 = model_engine.bfloat16_enabled()
    print(f'fp16={fp16},bf16={bf16}')

    # gradient_accumulation_steps = logical_batch / (micro_batch * n_gpu)
    grad_accum = config['train_batch_size'] // (config['train_micro_batch_size_per_gpu'] * n_gpu)
    WARMUP = 5  # logical steps to skip before timing

    # Wire BandMF state onto the DeepSpeed ZeRO optimizer
    num_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if bandmf_solver is not None and privacy_engine is not None:
        setup_dp_optimizer(
            optimizer,
            bandmf_solver=bandmf_solver,
            noise_multiplier=privacy_engine.noise_multiplier,
            per_example_max_grad_norm=privacy_engine.max_grad_norm,
            train_batch_size=config['train_batch_size'],
            partition=1,
            noise_offload=args.noise_offload,
            partition_size=num_trainable // n_gpu,
            local_rank=torch.distributed.get_rank(),
            part_GPU=0,
        )

    # Start noise_worker subprocess and set up per-step cpu_thread for CPU-GEMV offload
    noise_process = stop_event = noise_queue = None
    if args.noise_offload and bandmf_solver is not None and privacy_engine is not None:
        from fastDP.noise_worker import noise_worker as _noise_worker
        torch.multiprocessing.set_start_method('spawn', force=True)
        noise_queue = torch.multiprocessing.Queue(maxsize=1)
        stop_event  = torch.multiprocessing.Event()
        noise_std   = privacy_engine.noise_multiplier * privacy_engine.max_grad_norm
        dev_rank    = torch.distributed.get_rank()
        gpu_device  = f'cuda:{model_engine.local_rank}'
        noise_process = torch.multiprocessing.Process(
            target=_noise_worker,
            args=(noise_queue, stop_event, num_trainable // n_gpu, noise_std, bandmf_solver),
            kwargs={
                'gpu_device':  gpu_device,
                'dev_rank':    dev_rank,
                'verbose':     True,
                'speed_mode':  True,
                'num_threads': args.noise_num_threads,
            },
        )
        noise_process.start()
        print(f'[rank{dev_rank}] noise_worker started (speed_mode=True, gpu={gpu_device})', flush=True)

    unit_size = (num_trainable // n_gpu)  # params per GPU partition

    h2d_bench_ms = 0.0
    if noise_queue is not None:
        import time as _time
        _BENCH_RUNS = 10
        _BENCH_WARMUP = 3
        _dummy_cpu = torch.empty(unit_size, dtype=torch.float32)
        _samples = []
        for _ in range(_BENCH_RUNS + _BENCH_WARMUP):
            _t0 = _time.perf_counter()
            _dummy_cpu.to(gpu_device)
            torch.cuda.synchronize()
            _samples.append((_time.perf_counter() - _t0) * 1000)
        del _dummy_cpu
        h2d_bench_ms = sum(_samples[_BENCH_WARMUP:]) / _BENCH_RUNS
        print(f'[H2D bench] {h2d_bench_ms:.2f} ms avg over {_BENCH_RUNS} runs '
              f'(min={min(_samples[_BENCH_WARMUP:]):.2f} max={max(_samples[_BENCH_WARMUP:]):.2f}) '
              f'for {unit_size * 4 / 1e9:.2f} GB', flush=True)

    def _make_cpu_thread():
        """Fetch CUDA GEMV result (already on GPU via IPC) and write to noise_placeholder."""
        import threading
        def _fn():
            noise_gpu = noise_queue.get(timeout=60.0)   # CUDA tensor via IPC
            optimizer.noise_placeholder[:unit_size] = noise_gpu
        return threading.Thread(target=_fn, daemon=True)

    # Enable benchmark events on privacy_engine and net (for clip timing)
    if privacy_engine is not None:
        privacy_engine.benchmark = True
        privacy_engine.start_noise_gen = []
        privacy_engine.end_noise_gen   = []
        privacy_engine.end_noise_add   = []
        net.start_clip = []
        net.end_clip   = []

    # Per-micro-batch CUDA event lists
    ev_start_iter  = []; ev_end_forward = []
    ev_end_loss    = []; ev_end_backward = []
    # Per-logical-step CUDA event lists
    ev_start_step  = []; ev_end_step    = []

    def train(epoch):
        net.train()
        train_loss = 0; correct = 0; total = 0
        logical_step = 0

        # Prefetch noise for step 0 before the loop starts
        cpu_thread = _make_cpu_thread() if noise_queue is not None else None
        if cpu_thread is not None:
            cpu_thread.start()

        for batch_idx, data in enumerate(tqdm(trainloader)):
            is_step_boundary = (batch_idx + 1) % grad_accum == 0

            inputs, targets = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
            if fp16:  inputs = inputs.half()
            if bf16:  inputs = inputs.bfloat16()

            _tw(ev_start_iter)
            if batch_idx % grad_accum == 0:
                _tw(ev_start_step)

            outputs = model_engine(inputs)
            _tw(ev_end_forward)

            loss = criterion(outputs, targets)
            _tw(ev_end_loss)

            model_engine.backward(loss)
            _tw(ev_end_backward)

            if is_step_boundary and cpu_thread is not None:
                cpu_thread.join()  # ensure noise is in noise_placeholder before step()

            model_engine.step()

            if is_step_boundary:
                _tw(ev_end_step)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if is_step_boundary:
                logical_step += 1
                _done = args.num_batches > 0 and logical_step >= args.num_batches
                if noise_queue is not None and not _done:
                    cpu_thread = _make_cpu_thread()
                    cpu_thread.start()
                if _done:
                    break

        torch.cuda.synchronize()

        if torch.distributed.get_rank() == 0:
            skip_micro = WARMUP * grad_accum
            skip_step  = WARMUP

            iter_ms    = _avg_ms(ev_start_iter,   ev_end_backward, skip_micro)  # forward+loss+backward per micro
            forward_ms = _avg_ms(ev_start_iter,   ev_end_forward,  skip_micro)
            loss_ms    = _avg_ms(ev_end_forward,  ev_end_loss,     skip_micro)
            backward_ms= _avg_ms(ev_end_loss,     ev_end_backward, skip_micro)
            step_ms    = _avg_ms(ev_start_step,   ev_end_step,     skip_step)   # optimizer step per logical step
            sbs_ms     = _avg_ms(ev_end_step[:-1],ev_end_step[1:], skip_step)   # step-to-step
            total_ms   = sbs_ms  # logical step time ≈ sbs

            clip_ms = ngen_ms = nadd_ms = 0.0
            if privacy_engine is not None:
                if net.start_clip:
                    clip_ms = _avg_ms(net.start_clip, net.end_clip, skip_micro)
                if privacy_engine.start_noise_gen:
                    ngen_ms = _avg_ms(privacy_engine.start_noise_gen, privacy_engine.end_noise_gen, skip_step)
                if privacy_engine.end_noise_gen and privacy_engine.end_noise_add:
                    nadd_ms = _avg_ms(privacy_engine.end_noise_gen, privacy_engine.end_noise_add, skip_step)

            h2d_ms = h2d_bench_ms  # one-time H2D benchmark (CPU→GPU, blocking)

            print(f'\n--- Timing Breakdown (avg over post-warmup steps, micro-batch={config["train_micro_batch_size_per_gpu"]}, grad_accum={grad_accum}) ---')
            print(f'{"step_by_step":<16}: {sbs_ms:.2f} ms')
            print(f'{"iter":<16}: {iter_ms * grad_accum:.2f} ms  (micro×{grad_accum})')
            print(f'{"forward":<16}: {forward_ms:.2f} ms  (per micro)')
            print(f'{"loss":<16}: {loss_ms:.2f} ms  (per micro)')
            print(f'{"backward":<16}: {backward_ms - clip_ms:.2f} ms  (per micro, excl. clip)')
            print(f'{"clip":<16}: {clip_ms:.2f} ms  (per micro)')
            print(f'{"optim_step":<16}: {step_ms:.2f} ms  (per logical step)')
            print(f'{"ngen":<16}: {ngen_ms:.2f} ms  (per logical step)')
            print(f'{"nadd":<16}: {nadd_ms:.2f} ms  (per logical step)')
            print(f'{"h2d_transfer":<16}: {h2d_ms:.2f} ms  (one-time benchmark, CPU→GPU blocking)')
            print(f'Avg_Iter {sbs_ms:.2f}ms  (over {logical_step - WARMUP} steps after {WARMUP}-step warmup)')

        print('Epoch: ', epoch, logical_step,
              'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / max(logical_step * grad_accum, 1),
                 100. * correct / max(total, 1), correct, total))

    for epoch in range(args.epochs):
        train(epoch)

    if noise_process is not None:
        stop_event.set()
        noise_process.join()


if __name__ == '__main__':
    import deepspeed
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch ViT-Large Benchmark')
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--epsilon', default=2, type=float)
    parser.add_argument('--clipping_mode', default='MixOpt', type=str)
    parser.add_argument('--model', default='vit_large_patch16_224', type=str)
    parser.add_argument('--clipping_style', type=str, default='layer-wise')
    parser.add_argument('--min_separation', type=int, default=1,
                        help='BandMF bandwidth b (1 = standard DP-SGD)')
    parser.add_argument('--noise_offload', action='store_true', default=False,
                        help='Offload BandMF GEMV to CPU noise_worker subprocess')
    parser.add_argument('--noise-num-threads', type=int, default=None, dest='noise_num_threads',
                        help='OMP/MKL thread count for the CPU noise_worker subprocess')
    parser.add_argument('--num_batches', type=int, default=0,
                        help='Limit training to N logical steps per epoch (0 = full epoch)')

    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    from fastDP import PrivacyEngine

    import torch
    import torchvision
    torch.manual_seed(3)
    import torch.nn as nn
    import torch.optim as optim
    import timm
    from opacus.validators import ModuleValidator
    from tqdm import tqdm
    import warnings; warnings.filterwarnings("ignore")

    import json

    deepspeed.init_distributed()

    main(args)
