# CoFi Multi-Model Launcher

`cofi.py` is a self-contained launcher for running CoFi pruning workflows across
five model families:

- `bertbase`
- `tinybert`
- `distilbert`
- `mobilebert`
- `albert`

Each model workflow is embedded inside `cofi.py` and runs in its own namespace,
so the model-specific patching logic stays separate. This matters because each
model uses its own patched clone of `CoFiPruning`.

## Environment Setup

Use `cofi_environment.yml` to create the Python environment before running the
launcher.

```bash
conda env create -f cofi_environment.yml
conda activate cofi
```

If the environment already exists and you changed the YAML file, update it with:

```bash
conda env update -f cofi_environment.yml --prune
conda activate cofi
```

The environment includes Python, PyTorch, Transformers, Datasets, Evaluate,
Safetensors, and the other packages used by the pruning/evaluation workflows.

After activating it, check that the launcher starts:

```bash
python cofi.py --list-models
```

## Quick Start

List supported models and aliases:

```bash
python cofi.py --list-models
```

Show status for every model:

```bash
python cofi.py --status
```

Show status for one model:

```bash
python cofi.py --model bertbase --status
python cofi.py --model tinybert --status
python cofi.py --model distilbert --status
python cofi.py --model mobilebert --status
python cofi.py --model albert --status
```

Run one block for one model:

```bash
python cofi.py --model bertbase --block 0
python cofi.py --model bertbase --block 1
python cofi.py --model bertbase --block 2 --task sst2
python cofi.py --model bertbase --block 3 --task sst2
python cofi.py --model bertbase --block 4
```

Run one block for all models:

```bash
python cofi.py --block 1
python cofi.py --block 4
```

## Supported Tasks

All embedded workflows use these GLUE tasks:

- `sst2`
- `qnli`
- `mnli`
- `qqp`
- `rte`

If `--task` is omitted, the selected block runs for all tasks supported by that
model.

Example:

```bash
python cofi.py --model mobilebert --block 2 --task rte
```

## Models And Aliases

You can use the canonical model names or their aliases.

| Canonical | Aliases |
| --- | --- |
| `bertbase` | `bert`, `base`, `bert-base`, `bert_base` |
| `tinybert` | `tiny`, `tiny-bert`, `tiny_bert` |
| `distilbert` | `distil`, `distil-bert`, `distil_bert` |
| `mobilebert` | `mobile`, `mobile-bert`, `mobile_bert` |
| `albert` | `albertbase`, `albert-base`, `albert_base` |

These are equivalent:

```bash
python cofi.py --model bertbase --status
python cofi.py --model bert --status
python cofi.py --model bert-base --status
```

## Blocks

### Block 0: Clone And Patch CoFiPruning

```bash
python cofi.py --model bertbase --block 0
```

Block 0 creates a fresh model-specific clone of the upstream CoFiPruning repo
and applies patches needed for the selected model.

Important: Block 0 refreshes the model-specific repo directory. If that clone
already exists, it may be removed and recreated. This is intentional so patches
do not leak across models.

Typical repo directories:

- `CoFiPruning_Base`
- `CoFiPruning_Tiny`
- `CoFiPruning_Distil`
- `CoFiPruning_Mobile`
- `CoFiPruning_Albert`

Run Block 0 before pruning, especially after changing machines, dependencies,
or model scripts.

### Block 1: Download Fine-Tuned Models

```bash
python cofi.py --model bertbase --block 1
python cofi.py --model bertbase --block 1 --task rte
```

Block 1 downloads task-specific fine-tuned checkpoints and tokenizers from
Hugging Face, then saves them locally.

The output directories are model/task-specific, for example:

- `ft_base_sst2`
- `ft_tinybert_sst2`
- `ft_distilbert_sst2`
- `ft_mobilebert_sst2`
- `ft_albert_sst2`

If a model is already downloaded, the block skips it.

### Block 2: Prune

```bash
python cofi.py --model bertbase --block 2 --task sst2
```

Block 2 runs CoFi pruning for the selected model and task. It uses the
fine-tuned checkpoint from Block 1 as the distillation source and writes pruning
outputs to a model/task-specific directory.

Example output directories:

- `pr_base_sst2_s60`
- `pr_tinybert_sst2_s60`
- `pr_distilbert_sst2_s60`
- `pr_mobilebert_sst2_s60`
- `pr_albert_sst2_s60`

The target sparsity is `60%` in the current scripts.

Each pruning run writes a log file named `pruning_log.txt` inside the pruning
output directory. If a `best` checkpoint already exists, the task is skipped.

### Block 3: Evaluate

```bash
python cofi.py --model bertbase --block 3 --task sst2
```

Block 3 evaluates both:

- the unpruned fine-tuned model from Block 1
- the pruned model from Block 2, if available

It reports metrics such as:

- task score, usually accuracy or F1
- parameter count
- memory estimate
- latency
- throughput
- sparsity percentage

Evaluation results are saved as `eval_results.json` inside the relevant
fine-tuned or pruned output directory.

### Block 4: Results Table

```bash
python cofi.py --model bertbase --block 4
```

Block 4 reads saved evaluation results and prints a summary table for all tasks.
It does not run training or evaluation. Run Block 3 first if the table shows
`N/A`.

## Common Workflows

### Full Workflow For One Model

```bash
python cofi.py --model bertbase --block 0
python cofi.py --model bertbase --block 1
python cofi.py --model bertbase --block 2
python cofi.py --model bertbase --block 3
python cofi.py --model bertbase --block 4
```

### One Task Only

```bash
python cofi.py --model distilbert --block 0
python cofi.py --model distilbert --block 1 --task rte
python cofi.py --model distilbert --block 2 --task rte
python cofi.py --model distilbert --block 3 --task rte
python cofi.py --model distilbert --block 4
```

### Check What Is Done

```bash
python cofi.py --model mobilebert --status
```

Status output uses:

- `v` for done
- `.` for not done yet

### Run The Same Block Across All Models

```bash
python cofi.py --block 1
python cofi.py --block 4
```

When `--model` is omitted and `--block` or `--status` is provided, `cofi.py`
runs that command across every supported model.

## Standalone Source Files

`cofi.py` embeds the standalone model scripts:

- `bertbase` comes from `cofi_base_new.py`
- `tinybert` comes from `cofi_tinybert.py`
- `distilbert` comes from `cofi_distilbert.py`
- `mobilebert` comes from `cofi_mobilebert.py`
- `albert` comes from `cofi_albert.py`

For normal use, run `cofi.py`. Edit the standalone files only when you want to
change a model-specific workflow, then re-embed the updated source into
`cofi.py`.

## Notes

- Block 0 needs internet access because it clones CoFiPruning.
- Block 1 needs internet access because it downloads Hugging Face checkpoints.
- Blocks 2 and 3 can be slow and GPU-heavy.
- Re-running blocks is generally safe: existing downloaded, pruned, or evaluated
  outputs are skipped when detected.
- `mnli` uses the matched validation split for evaluation.
- `qqp` reports F1; the other listed tasks generally report accuracy.
