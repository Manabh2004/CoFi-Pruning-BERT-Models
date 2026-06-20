# COFI Evaluation Results

**Date:** 2026-06-20  
**Target sparsity:** 60%  
**Tasks:** SST-2, QNLI, MNLI, QQP, RTE  
**Score metric:** Accuracy for SST-2, QNLI, MNLI, and RTE; F1 for QQP

---

## Executive Summary

This report summarizes the evaluation results from applying COFI-style structured pruning across five transformer families. The goal was to compare quality retention, sparsity, latency, memory use, and throughput after targeting roughly 60% pruning.

The strongest overall result is **BERT-base**, which keeps quality nearly intact while approaching a 2x speedup. **TinyBERT** is also very strong, especially given that it starts from a compact model. **ALBERT** delivers the highest acceleration, reaching up to **2.86x speedup**, but shows more task-specific risk on RTE. **MobileBERT** is stable but does not fully reach the 60% sparsity target. **DistilBERT** speeds up, but quality drops too sharply under this pruning recipe.

---

## At-a-Glance Ranking

| Rank | Model | Avg Speedup | Retention Range | Main Takeaway |
|---:|---|---:|---:|---|
| 1 | **BERT-base** | 1.98x | **97.9% to 102.7%** | Best quality-speed balance |
| 2 | **ALBERT** | **2.82x** | 85.0% to 100.5% | Fastest, but RTE is risky |
| 3 | **TinyBERT** | 1.86x | 97.8% to 101.3% | Best lightweight result |
| 4 | **MobileBERT** | 1.62x | 94.9% to 99.4% | Stable, moderate compression |
| 5 | **DistilBERT** | 1.88x | 41.8% to 86.3% | Likely COFI compatibility issue |

---

## What Happened by Task

Each GLUE task is shown as its own table.

### SST-2

| Model     |Base Score|Pruned Score|Speedup|Retention|Sparsity|
|-----------|----------|------------|-------|---------|--------|
| BERT-base | 0.9323   | 0.9300     | 1.92x | 99.8%   | 58.95% |
| TinyBERT  | 0.8922   | 0.8899     | 2.00x | 99.7%   | 56.70% |
| DistilBERT| 0.9106   | 0.6537     | 1.87x | 71.8%   | 57.13% |
| MobileBERT| 0.9037   | 0.8578     | 1.62x | 94.9%   | 46.28% |
| ALBERT    | 0.9266   | 0.9128     | 2.83x | 98.5%   | 60.42% |

### QNLI

| Model     |Base Score|Pruned Score|Speedup|Retention|Sparsity|
|-----------|----------|------------|-------|---------|--------|
| BERT-base | 0.9050   | 0.9060     | 1.99x | 100.1%  | 59.25% |
| TinyBERT  | 0.8220   | 0.8330     | 2.09x | 101.3%  | 56.38% |
| DistilBERT| 0.8910   | 0.5420     | 1.88x | 60.8%   | 57.33% |
| MobileBERT| 0.9020   | 0.8970     | 1.52x | 99.4%   | 47.22% |
| ALBERT    | 0.9180   | 0.8720     | 2.86x | 95.0%   | 60.46% |

### MNLI

| Model    |Base Score|Pruned Score|Speedup|Retention|Sparsity|
|----------|----------|------------|-------|---------|--------|
|BERT-base | 0.8500   | 0.8460     | 2.05x | 99.5%   | 58.56% |
|TinyBERT  | 0.8020   | 0.7840     | 1.94x | 97.8%   | 56.60% |
|DistilBERT| 0.8070   | 0.3370     | 1.88x | 41.8%   | 57.35% |
|MobileBERT| 0.8230   | N/A        | N/A   | N/A     | N/A    |
|ALBERT    | 0.8370   | 0.8410     | 2.76x | 100.5%  | 60.33% |

### QQP

| Model    |Base Score|Pruned Score|Speedup|Retention|Sparsity|
|----------|----------|------------|-------|---------|--------|
|BERT-base | 0.8720   | 0.8956     | 2.06x | 102.7%  | 59.04% |
|TinyBERT  | 0.8485   | 0.8476     | 1.09x | 99.9%   | 56.56% |
|DistilBERT| 0.8783   | 0.4083     | 1.90x | 46.5%   | 57.38% |
|MobileBERT| 0.8596   | 0.8512     | 1.65x | 99.0%   | 48.18% |
|ALBERT    | 0.8667   | 0.8644     | 2.85x | 99.7%   | 60.35% |

### RTE

| Model     |Base Score|Pruned Score|Speedup|Retention|Sparsity|
|-----------|----------|------------|-------|---------|--------|
|BERT-base  | 0.6895   | 0.6751     | 1.86x | 97.9%   | 58.64% |
|TinyBERT   | 0.6679   | 0.6751     | 2.19x | 101.1%  | 58.69% |
|DistilBERT | 0.6570   | 0.5668     | 1.89x | 86.3%   | 58.63% |
|MobileBERT | 0.6679   | 0.6390     | 1.70x | 95.7%   | 53.97% |
|ALBERT     | 0.7726   | 0.6570     | 2.79x | 85.0%   | 58.47% |
---

## Model Notes

### BERT-base

BERT-base is the cleanest result in the evaluation. It lands close to the target sparsity, nearly doubles inference speed, and retains quality across every task.

- Speedup range: 1.86x to 2.06x
- Retention range: 97.9% to 102.7%
- Best use: safest deployment candidate when quality matters

### TinyBERT

TinyBERT performs surprisingly well for an already compact model. It preserves quality across tasks and gets strong acceleration everywhere except QQP, where the latency gain is modest.

Due to the small size of TinyBERT, we have applied 50 epochs for large tasks, deviating from the paper's 20. This was an experimental setup to see the difference in results for ourselves. However, there was no significance difference to be seen. Yet we stuck to 50 epochs to maintain consistency.

- Speedup range: 1.09x to 2.19x
- Retention range: 97.8% to 101.3%
- Best use: compact deployment with strong retention

### DistilBERT

DistilBERT is the main warning sign. The pruning recipe produces consistent speedups, but the score drops are too large for a production-quality compressed model.

Multiple pruning parameter settings were evaluated, but the behavior remained largely unchanged. This suggests the failure mode is not simply a poor hyperparameter choice; it appears to be a deeper compatibility issue between DistilBERT and this COFI setup. In particular, DistilBERT seems to adhere strongly to a poor local optimum during pruning and recovery, making it difficult to regain task performance even when the parameter settings are adjusted.

- Speedup range: 1.87x to 1.90x
- Retention range: 41.8% to 86.3%
- Best use: compatibility investigation, not a final compressed model

### MobileBERT

MobileBERT stays stable, but the realized sparsity is much lower than the target. Multiple pruning parameter settings were tested, yet the model consistently failed to approach the intended 60% sparsity level. Even so, the result is still useful: MobileBERT retains reasonable task performance despite landing closer to 50% sparsity. MNLI also has no pruned result, so its evaluation is incomplete.

- Speedup range: 1.52x to 1.70x, excluding MNLI
- Retention range: 94.9% to 99.4%
- Best use: low-risk compression where moderate sparsity is acceptable

### ALBERT

ALBERT is the speed leader. It reaches the target sparsity cleanly and has the best latency reductions, but RTE shows that the gains come with some task-specific quality risk.

- Speedup range: 2.76x to 2.86x
- Retention range: 85.0% to 100.5%
- Best use: maximum throughput when some tail-task risk is acceptable

## MNLI /best issue:

In every Model, one pattern was seen constantly: MNLI never created a /best.

CoFi only writes a best/ checkpoint once a flag flips on, triggered the first time the model's estimated sparsity crosses the 60% target at one of the periodic eval checks. For MNLI, that estimate seems to have hovered just under the threshold without ever crossing it at the right moment, so the flag never flipped and nothing was saved — even though the model's actual final sparsity (~56-57%) was in line with every other task. 

Fix we used: CoFi still dumps the model's latest state (model.safetensors, zs.pt, l0_module.pt, config.json) to the main output folder after every eval, even when it's not flagged as "best." We just manually copied those files into a best/ folder ourselves. The eval script doesn't care how best/ got created — it just needs the files there.

---

## Key Findings

1. **BERT-base is the best overall tradeoff.** It combines near-2x speedup with extremely strong retention.
2. **ALBERT is the acceleration winner.** It consistently gives the largest latency reduction, but RTE needs caution.
3. **TinyBERT proves pruning is still useful on compact models.** It keeps quality while improving throughput substantially.
4. **DistilBERT appears incompatible with this COFI setup.** Parameter changes did not resolve the quality collapse, suggesting the model may be getting trapped in a poor local optimum during pruning/recovery.
5. **MobileBERT needs a separate sparsity story.** It remains stable, but does not reach the target as cleanly as the other models.

---

You can find the pruned models here:
https://huggingface.co/collections/Manabh/bert-models-glue-finetunes-cofi-prunes

## Deployment Recommendation

For the strongest showcase result, lead with **BERT-base pruned**: it is fast, stable, and easy to defend. For maximum throughput, present **ALBERT pruned** as the speed-focused option. Treat **TinyBERT pruned** as the lightweight win. Keep **MobileBERT** as a moderate-gain baseline, and mark **DistilBERT** as evidence of a likely compatibility issue between the architecture and this COFI pruning setup.

---

## Measurement Notes

- Latency and throughput were measured on an NVIDIA RTX 4060 Ti GPU with batch size 32 and 1000 validation examples.
- Sparsity references:
  - BERT-base: 85M non-embedding parameters.
  - TinyBERT-4L: approximately 14.35M non-embedding parameters.
  - DistilBERT: approximately 52M non-embedding parameters.
  - MobileBERT and ALBERT: auto-computed against baseline non-embedding parameters.


