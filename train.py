import os
import time
import numpy as np
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(__name__)
from utils import logging_utils
from utils.YParams import YParams
from utils import get_data_loader
from utils.loss import l2_loss, l2_loss_opt
from utils.metrics import weighted_rmse
from utils.plots import generate_images
from model import transformer

import matplotlib
matplotlib.use('Agg') 

def _create_adam_optimizer(model, params):
    adam_kwargs = {"lr": params.lr, "betas": (0.9, 0.95)}
    try:
        optimizer = optim.Adam(model.parameters(), fused=True, **adam_kwargs)
        logging.info("Using Adam optimizer implementation: fused")
        return optimizer
    except (TypeError, RuntimeError, ValueError) as exc:
        logging.info("Fused Adam unavailable (%s), falling back.", exc)

    try:
        optimizer = optim.Adam(model.parameters(), foreach=True, **adam_kwargs)
        logging.info("Using Adam optimizer implementation: foreach")
        return optimizer
    except (TypeError, RuntimeError, ValueError) as exc:
        logging.info("Foreach Adam unavailable (%s), falling back to default Adam.", exc)

    logging.info("Using Adam optimizer implementation: default")
    return optim.Adam(model.parameters(), **adam_kwargs)


def _amp_dtype_from_arg(amp_mode):
    if amp_mode == "none":
        return None
    if amp_mode == "bf16":
        return torch.bfloat16
    if amp_mode == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported AMP mode: {amp_mode}")


def _autocast_context(amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _build_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _maybe_compile_model(model, args):
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        logging.warning("torch.compile is unavailable; running in eager mode.")
        return model
    disable_cudagraphs = bool(getattr(args, "disable_compile_cudagraphs", False))
    if not disable_cudagraphs:
        # Compile + DDP + gradient accumulation can trigger CUDAGraph tensor
        # lifetime issues on some PyTorch builds.
        disable_cudagraphs = bool(getattr(args, "distributed", False)) and int(
            getattr(args, "_effective_grad_accum_steps", 1)
        ) > 1
    if disable_cudagraphs:
        try:
            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraphs = False
            logging.info("Disabled torch.compile cudagraphs for stability.")
        except Exception:
            os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
            logging.info("Set TORCHINDUCTOR_CUDAGRAPHS=0 for stability.")
    try:
        compiled_model = torch.compile(model, mode=args.compile_mode)
        logging.info("Enabled torch.compile with mode=%s", args.compile_mode)
        return compiled_model
    except Exception as exc:  # fallback on backend/runtime compile issues
        logging.warning("torch.compile failed (%s); running in eager mode.", exc)
        return model


def _maybe_cudagraph_step_begin(args):
    if not args.compile:
        return
    compiler_mod = getattr(torch, "compiler", None)
    step_begin = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
    if step_begin is not None:
        step_begin()


def _setup_distributed(params):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    torch.cuda.set_device(local_rank)

    if distributed:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")
        init_kwargs = {
            "backend": params.get("ddp_backend", "nccl"),
            "init_method": "env://",
        }
        try:
            dist.init_process_group(device_id=local_rank, **init_kwargs)
        except TypeError:
            dist.init_process_group(**init_kwargs)

    return distributed, rank, world_size, local_rank


def _cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _is_rank0(args):
    return args.rank == 0


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def train(params, args):
    device = torch.device(f"cuda:{args.local_rank}")
    amp_dtype = _amp_dtype_from_arg(args.amp)
    use_grad_scaler = args.amp == "fp16"
    grad_accum_steps = int(params.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    args._effective_grad_accum_steps = grad_accum_steps
    scaler = _build_grad_scaler(enabled=use_grad_scaler)
    if amp_dtype is None:
        logging.info("AMP disabled")
    else:
        logging.info("AMP enabled with mode=%s", args.amp)

    # get data loader
    logging.info('begin data loader initialisation')
    train_data_loader, train_dataset, train_sampler = get_data_loader(
        params,
        params.train_data_path,
        train=True,
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
    )
    val_data_loader, valid_dataset = get_data_loader(
        params,
        params.valid_data_path,
        train=False,
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
    )
    logging.info('data loader initialised')

    # create the model and copy it to the gpu
    model = transformer.transformer(params).to(device)
    model = _maybe_compile_model(model, args)
    if args.distributed:
        ddp_kwargs = {
            "device_ids": [args.local_rank],
            "output_device": args.local_rank,
            "find_unused_parameters": bool(params.get("ddp_find_unused_parameters", False)),
            "broadcast_buffers": bool(params.get("ddp_broadcast_buffers", False)),
            "bucket_cap_mb": float(params.get("ddp_bucket_cap_mb", 25)),
            "gradient_as_bucket_view": bool(params.get("ddp_gradient_as_bucket_view", False)),
            "static_graph": bool(params.get("ddp_static_graph", False)),
        }
        try:
            model = DDP(model, **ddp_kwargs)
        except TypeError as exc:
            logging.warning(
                "Optional DDP kwargs unsupported by this torch build (%s); "
                "falling back to basic DDP settings.",
                exc,
            )
            ddp_kwargs.pop("bucket_cap_mb", None)
            ddp_kwargs.pop("gradient_as_bucket_view", None)
            ddp_kwargs.pop("static_graph", None)
            model = DDP(model, **ddp_kwargs)

    # create the optimiser
    optimizer = _create_adam_optimizer(model, params)

    logging.info(_unwrap_model(model))

    iters = 0
    startEpoch = 0
 
    # setup a loss rate scheduler if there is one specified
    if params.lr_schedule == 'cosine':
        if params.warmup > 0:
            lr_scale = lambda x: min((x+1)/params.warmup, 0.5*(1 + np.cos(np.pi*x/params.num_iters)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_iters)
    else:
        scheduler = None

    loss_func = l2_loss

    logging.info("Beginning Training Loop...")

    # log initial loss on train and validation to tensorboard
    # this does not train the network, just runs it on two data sets to see the current quality
    with torch.no_grad():
        inp, tar = [x.to(device, non_blocking=True) for x in next(iter(train_data_loader))]
        with _autocast_context(amp_dtype):
            _maybe_cudagraph_step_begin(args)
            gen = model(inp)
            tr_loss = loss_func(gen, tar)
        inp, tar = [x.to(device, non_blocking=True) for x in next(iter(val_data_loader))]
        with _autocast_context(amp_dtype):
            _maybe_cudagraph_step_begin(args)
            gen = model(inp)
            val_loss = loss_func(gen, tar)
            val_rmse = weighted_rmse(gen, tar)
        if args.distributed:
            dist.all_reduce(tr_loss, op=dist.ReduceOp.SUM)
            tr_loss /= args.world_size
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= args.world_size
            dist.all_reduce(val_rmse, op=dist.ReduceOp.SUM)
            val_rmse /= args.world_size
        if args.tboard_writer is not None:
            args.tboard_writer.add_scalar('Loss/train', tr_loss.item(), 0)
            args.tboard_writer.add_scalar('Loss/valid', val_loss.item(), 0)
            args.tboard_writer.add_scalar('RMSE(u10m)/valid', val_rmse.cpu().numpy()[0], 0)

    params.num_epochs = params.num_iters//len(train_data_loader)
    logging.info('number of epochs: '+str(params.num_epochs)+'(' + str(params.num_iters) + ',' + str(len(train_data_loader)) + ')')
    iters = 0
    t1 = time.time()
    for epoch in range(startEpoch, startEpoch + params.num_epochs):
        torch.cuda.synchronize() # barrier on the gpu(s) to ensure accurarte timings
        start = time.time()
        tr_loss_sum = torch.zeros(1, dtype=torch.float32, device=device)
        tr_loss_count = 0
        tr_time = 0.
        dat_time = 0.
        log_time = 0.

        # enabling training mode for the model
        model.train()
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum_step_in_window = 0
        accum_target = grad_accum_steps
        optimizer_step_count = 0
        num_train_steps = len(train_data_loader)
        step_count = 0
        for i, data in enumerate(train_data_loader, 0):
            if (epoch == 3 and i == 0):
                torch.cuda.profiler.start()
            if (epoch == 3 and i == len(train_data_loader) - 1):
                torch.cuda.profiler.stop()

            torch.cuda.nvtx.range_push(f"step {i}")
            iters += 1
            dat_start = time.time()
            torch.cuda.nvtx.range_push(f"data copy in {i}")

            inp, tar = [x.to(device, non_blocking=True) for x in data]
            torch.cuda.nvtx.range_pop() # copy in

            tr_start = time.time()

            if accum_step_in_window == 0:
                remaining_steps = num_train_steps - i
                accum_target = min(grad_accum_steps, remaining_steps)
            should_sync_and_step = (accum_step_in_window + 1) == accum_target
            ddp_sync_context = nullcontext()
            if args.distributed and not should_sync_and_step:
                ddp_sync_context = model.no_sync()

            torch.cuda.nvtx.range_push(f"forward")
            with ddp_sync_context:
                with _autocast_context(amp_dtype):
                    _maybe_cudagraph_step_begin(args)
                    gen = model(inp)
                    loss = loss_func(gen, tar)
                loss_for_backward = loss / float(accum_target)
                if use_grad_scaler:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()
            torch.cuda.nvtx.range_pop() #forward

            if should_sync_and_step:
                torch.cuda.nvtx.range_push(f"optimizer")
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                torch.cuda.nvtx.range_pop() # optimizer
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1
                accum_step_in_window = 0
            else:
                accum_step_in_window += 1

            tr_loss_sum += loss.detach().float()
            tr_loss_count += 1

            torch.cuda.nvtx.range_pop() # step
            # lr step
            if scheduler is not None:
                scheduler.step()

            tr_end = time.time()
            tr_time += tr_end - tr_start
            dat_time += tr_start - dat_start
            step_count += 1

        torch.cuda.synchronize() # device sync to ensure accurate epoch timings
        end = time.time()

        epoch_time = torch.tensor([end - start], device=device, dtype=torch.float32)
        if args.distributed:
            dist.all_reduce(epoch_time, op=dist.ReduceOp.MAX)
        iters_per_sec = step_count / epoch_time.item()
        samples_per_sec = params["global_batch_size"] * iters_per_sec
        tr_loss_totals = torch.cat(
            (
                tr_loss_sum,
                torch.tensor([float(tr_loss_count)], device=device, dtype=torch.float32),
            )
        )
        if args.distributed:
            dist.all_reduce(tr_loss_totals, op=dist.ReduceOp.SUM)
        mean_tr_loss = (tr_loss_totals[0] / tr_loss_totals[1].clamp_min(1.0)).item()
        logging.info('Time taken for epoch %i is %f sec, avg %f samples/sec',
                     epoch + 1, epoch_time.item(), samples_per_sec)
        logging.info('  Avg train loss=%f' % mean_tr_loss)
        logging.info(
            '  Optimizer steps=%d (grad_accum_steps=%d)',
            optimizer_step_count,
            grad_accum_steps,
        )
        if args.tboard_writer is not None:
            args.tboard_writer.add_scalar('Loss/train', mean_tr_loss, iters)
            args.tboard_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iters)
            args.tboard_writer.add_scalar('Avg iters per sec', iters_per_sec, iters)
            args.tboard_writer.add_scalar('Avg samples per sec', samples_per_sec, iters)
            fig = generate_images([inp, tar, gen])
            args.tboard_writer.add_figure('Visualization, t2m', fig, iters, close=True)

        val_start = time.time()
        val_loss = torch.zeros(1, dtype=torch.float32, device=device)
        val_rmse = torch.zeros((params.n_out_channels), dtype=torch.float32, device=device)
        valid_steps = 0
        model.eval()

        with torch.inference_mode():
            with torch.no_grad():
                for i, data in enumerate(val_data_loader, 0):
                    inp, tar = [x.to(device, non_blocking=True) for x in data]
                    with _autocast_context(amp_dtype):
                        _maybe_cudagraph_step_begin(args)
                        gen = model(inp)
                        val_loss += loss_func(gen, tar).float()
                        val_rmse += weighted_rmse(gen, tar).float()
                    valid_steps += 1

        valid_steps_tensor = torch.tensor([valid_steps], dtype=torch.float32, device=device)
        if args.distributed:
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_rmse, op=dist.ReduceOp.SUM)
            dist.all_reduce(valid_steps_tensor, op=dist.ReduceOp.SUM)
        total_valid_steps = max(valid_steps_tensor.item(), 1.0)
        val_rmse /= total_valid_steps # Avg validation rmse
        val_loss /= total_valid_steps
        val_end = time.time()
        logging.info('  Avg val loss={}'.format(val_loss.item()))
        logging.info('  Total validation time: {} sec'.format(val_end - val_start)) 
        if args.tboard_writer is not None:
            args.tboard_writer.add_scalar('Loss/valid', val_loss, iters)
            args.tboard_writer.add_scalar('RMSE(u10m)/valid', val_rmse.cpu().numpy()[0], iters)
            args.tboard_writer.flush()

    t2 = time.time()
    tottime = t2 - t1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str, help='index of thecurrent experiment')
    parser.add_argument("--yaml_config", default='./config/coursework_transformer.yaml', type=str, help='path to yaml file containing training configuration')
    parser.add_argument("--config", default='base', type=str, help='name of desired config in yaml file (base, short, smoke, or medium)')
    parser.add_argument("--num_iters", default=None, type=int, help='number of iters to run')
    parser.add_argument("--grad_accum_steps", default=None, type=int, help='number of micro-batches to accumulate before optimizer step')
    parser.add_argument("--global_batch_size", default=None, type=int, help='override global batch size across all ranks')
    parser.add_argument("--local_batch_size", default=None, type=int, help='override per-rank batch size (global becomes local_batch_size * WORLD_SIZE)')
    parser.add_argument("--num_data_workers", default=None, type=int, help='number of data workers for data loader')
    parser.add_argument(
        "--experiment_num_data_workers",
        default=None,
        type=int,
        help='single worker setting for one experiment run (alias for --num_data_workers)'
    )
    parser.add_argument(
        "--amp",
        default="none",
        choices=["none", "bf16", "fp16"],
        type=str,
        help='mixed precision mode for autocast ("none", "bf16", or "fp16")'
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help='enable torch.compile for the model'
    )
    parser.add_argument(
        "--compile_mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        type=str,
        help='mode passed to torch.compile when --compile is enabled'
    )
    parser.add_argument("--ddp_bucket_cap_mb", default=None, type=float, help='DDP bucket size in MB')
    parser.add_argument("--ddp_gradient_as_bucket_view", action="store_true", help='enable DDP gradient_as_bucket_view')
    parser.add_argument("--ddp_static_graph", action="store_true", help='enable DDP static_graph optimization')
    args = parser.parse_args()
 
    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    if args.experiment_num_data_workers is not None:
        if args.experiment_num_data_workers < 0:
            raise SystemExit("--experiment_num_data_workers must be >= 0")
        if (
            args.num_data_workers is not None
            and args.num_data_workers != args.experiment_num_data_workers
        ):
            raise SystemExit(
                "Conflicting worker settings: use either --num_data_workers "
                "or --experiment_num_data_workers, not different values for both."
            )
        args.num_data_workers = args.experiment_num_data_workers

    args.distributed, args.rank, args.world_size, args.local_rank = _setup_distributed(params)

    # Update config with modified args
    if args.num_iters is not None:
        params.update({"num_iters" : args.num_iters})
    if args.grad_accum_steps is not None:
        if args.grad_accum_steps <= 0:
            _cleanup_distributed(args.distributed)
            raise SystemExit("--grad_accum_steps must be > 0")
        params.update({"grad_accum_steps": args.grad_accum_steps})

    if args.global_batch_size is not None and args.local_batch_size is not None:
        _cleanup_distributed(args.distributed)
        raise SystemExit("Use either --global_batch_size or --local_batch_size, not both.")
    if args.global_batch_size is not None:
        if args.global_batch_size <= 0:
            _cleanup_distributed(args.distributed)
            raise SystemExit("--global_batch_size must be > 0")
        params.update({"global_batch_size": args.global_batch_size})
    if args.local_batch_size is not None:
        if args.local_batch_size <= 0:
            _cleanup_distributed(args.distributed)
            raise SystemExit("--local_batch_size must be > 0")
        params.update({"global_batch_size": args.local_batch_size * args.world_size})

    if args.num_data_workers is not None:
        params.update({"num_data_workers" : args.num_data_workers})
    if args.ddp_bucket_cap_mb is not None:
        if args.ddp_bucket_cap_mb <= 0:
            _cleanup_distributed(args.distributed)
            raise SystemExit("--ddp_bucket_cap_mb must be > 0")
        params.update({"ddp_bucket_cap_mb": args.ddp_bucket_cap_mb})
    if args.ddp_gradient_as_bucket_view:
        params.update({"ddp_gradient_as_bucket_view": True})
    if args.ddp_static_graph:
        params.update({"ddp_static_graph": True})

    if params.global_batch_size % args.world_size != 0:
        _cleanup_distributed(args.distributed)
        raise SystemExit(
            "global_batch_size must be divisible by WORLD_SIZE: "
            f"{params.global_batch_size} % {args.world_size} != 0"
        )
    params.local_batch_size = params.global_batch_size // args.world_size

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(baseDir, args.config +  str(run_num) + '/')
    if _is_rank0(args):
        os.makedirs(expDir, exist_ok=True)
    if args.distributed:
        dist.barrier()

    if _is_rank0(args):
        logging.basicConfig(filename=os.path.join(expDir, 'out.log'), level=logging.INFO)
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))
    else:
        logging.basicConfig(level=logging.ERROR)
        args.tboard_writer = None

    params.experiment_dir = os.path.abspath(expDir)

    try:
        train(params, args)
    finally:
        if args.tboard_writer is not None:
            args.tboard_writer.close()
        _cleanup_distributed(args.distributed)

    if _is_rank0(args):
        logging.info('Finished')
        logging.shutdown()
