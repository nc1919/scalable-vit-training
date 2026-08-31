# Scalable Vision Transformer training optimisation

An experimental PyTorch training workload for a Vision Transformer over 20-channel, 360×720 gridded tensors. The work explores input-pipeline tuning, mixed precision, `torch.compile`, fused Adam, gradient accumulation, and two-GPU DistributedDataParallel (DDP).

## What is included

- `train.py` — single-process and `torchrun` training entry point.
- `model/` — Vision Transformer implementation.
- `utils/` — data loading, losses, metrics, plots, and configuration helpers.
- `config/coursework_transformer.yaml` — safe relative-path example with smoke, short, medium, and full profiles.
- Kubernetes job examples for the original GPU environment.
- [RESULTS.md](RESULTS.md) — curated measurements and limitations.

Training data, TensorBoard event files, scheduler logs, Python bytecode, and Nsight reports are not included.

## Environment

Python 3.10+ and a CUDA-capable PyTorch installation are recommended. Install the Python-level dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA driver separately if the generic resolver does not select one.

## Data configuration

The example configuration expects:

```text
data/
├── train/
├── valid/
├── test/
└── stats/
    ├── time_means.npy
    ├── global_means.npy
    └── global_stds.npy
```

The original dataset is not redistributed. Update the paths in `config/coursework_transformer.yaml` if your layout differs.

## Run

Smoke test on one GPU:

```bash
python train.py --config smoke --run_num smoke --amp bf16
```

Two-GPU DDP example:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config medium \
  --num_iters 1024 \
  --run_num ddp \
  --local_batch_size 8 \
  --grad_accum_steps 4 \
  --experiment_num_data_workers 4 \
  --amp bf16 \
  --compile \
  --compile_mode max-autotune
```

The Kubernetes manifests contain placeholders and site-specific storage, registry, queue, and GPU settings. Review every value before applying them.

## Reproducibility and scope

The checked-in logs were curated into [RESULTS.md](RESULTS.md) and then removed because they were generated, environment-specific, and responsible for most of the repository size. The results are exploratory single-environment measurements, not a general hardware claim.

This began as coursework/experimental ML code. Confirm dataset rights, institutional policy, and contributor permissions before changing visibility.

## License

No project-wide licence has been selected. Until one is added by the copyright holder, the code is available for review but no reuse rights are granted.
