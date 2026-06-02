# Running on HPC

Advisory notes for running the paper sweep on a multi-GPU cluster. Nothing here
is required on a single workstation; the [reproduction guide](REPRODUCING.md)
covers the portable path.

## GPU dispatch

`scripts/run_sweep.py` is a self-contained dispatcher: pass `--gpus 0 1 2 3` and
it runs one cell per GPU, refilling each GPU as cells finish. Every phase runs
in its own subprocess pinned to a single GPU via `CUDA_VISIBLE_DEVICES`, so no
`accelerate`/`torchrun` launcher is needed — one sweep process per node owns all
the node's GPUs.

For multiple nodes, run one dispatcher per node over disjoint axis slices (for
example, split `--models` across nodes) and point every node at the same
`--root` on shared storage. The Phase 1 organism is shared per
`(model, misalignment, seed, scale)`, so keeping all cells of one organism on
the same node avoids redundant organism training across nodes.

## Storage

Set `--root` (or `CONSISTENCY_EM_RUNS_DIR`) to a path on a high-capacity
filesystem. Per organism the sweep keeps the LoRA adapter plus one checkpoint
per Phase 1 epoch; per cell, the final adapter plus one checkpoint per Phase 3
epoch. At paper scale that is roughly four checkpoints per cell across the
matrix — size for it before launching.

## Judge token lifetime

The misalignment eval calls a LiteLLM judge, which needs a provider key in the
environment. If your key is short-lived (for example, minted by a secrets
broker with a sliding TTL), pass a refresh command instead of a static key:

```bash
python scripts/run_sweep.py ... \
    --judge-key-command 'your-broker mint-key'
```

The command is run before each phase, so a multi-hour sweep survives token
expiry. `OPENAI_BASE_URL` (or the relevant provider base-URL env var) routes the
judge through a proxy if you use one.

## GPT-OSS-20B: CUDA forward-compatibility

GPT-OSS-20B loads in vLLM only with a CUDA toolkit newer than some datacenter
drivers ship against. On a node whose driver is capped below the toolkit vLLM
needs, use NVIDIA's CUDA forward-compatibility layer: the `cuda-compat` package
provides a newer `libcuda` that a capped driver can load. Prepend its library
directory to `LD_LIBRARY_PATH` so the newer `libcuda` wins:

```bash
export LD_LIBRARY_PATH=/path/to/cuda-compat:$LD_LIBRARY_PATH
```

Forward-compatibility is supported only on datacenter GPUs (the GH200 used for
the paper qualifies). The other five models are unaffected — prepending an
existing directory is a no-op for them.

To let the sweep apply this per phase without exporting it globally, set
`CONSISTENCY_EM_CUDA_COMPAT_DIR` to the `cuda-compat` directory; the cell runner
prepends it to each phase subprocess's `LD_LIBRARY_PATH` when the directory
exists, and ignores it otherwise.

```bash
export CONSISTENCY_EM_CUDA_COMPAT_DIR=/path/to/cuda-compat
python scripts/run_sweep.py ... --models openai/gpt-oss-20b
```

## Resuming

The sweep is fully resumable: every phase and trajectory is skip-if-exists, and
result rows stream incrementally to the table. After a preemption or timeout,
re-run the identical command against the same `--root` and `--table`; finished
work is skipped and the sweep continues.
