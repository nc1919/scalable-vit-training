# Results

The source logs recorded training throughput in global samples/second on the original A100 environment. For completed runs, the value below is the arithmetic mean of the second half of the logged epoch-throughput values.

| Run | Configuration | Steady throughput (samples/s) | Last validation loss |
|---|---|---:|---:|
| Short baseline | single GPU, 0 workers, no compile optimisation | 2.42 | 0.34284 |
| Short worker/compile study | single GPU, 12 workers, max-autotune | 33.14 | 0.34327 |
| Medium 1,024 iterations | single GPU, 12 workers, max-autotune | 36.23 | 0.27451 |
| Medium DDP | 2 GPUs, pre-tuning DDP run | 70.90 | 0.27464 |
| Medium DDP | 2 GPUs, 4 workers/rank, max-autotune | 68.33 | 0.27469 |

A short worker-count sweep peaked at 12 workers (26.42 samples/s for that short comparison workload); 14 and 16 workers were slower, showing that worker count should be measured rather than maximised.

## Limitations

- The logs came from one cluster and GPU type; portability was not measured.
- Warm-up, compile time, and cache state can materially affect short runs.
- The repository did not preserve independent benchmark repetitions with uncertainty estimates.
- Several attempted accumulation/prefetch runs did not finish and are intentionally excluded from the table.
- Validation loss is shown only as a guardrail; these short runs do not establish model quality or convergence.

Re-run representative cases with fixed seeds, environment metadata, at least three repetitions, and a documented warm-up policy before using the numbers in a report.
