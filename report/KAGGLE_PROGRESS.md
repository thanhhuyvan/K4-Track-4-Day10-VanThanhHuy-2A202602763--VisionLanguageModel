# Kaggle progress — Văn Thành Huy

## Completed setup and benchmarks

Dataset version: `d10-21e4070301`. Official split: train 1422, val 276, test 311.
Kaggle runtime: two Tesla T4 GPUs, approximately 14.56 GiB each; code uses GPU 0.

| Benchmark | Updates | Trainable parameters | Peak allocated VRAM (GiB) | Minutes |
|---|---:|---:|---:|---:|
| E1 | 20 | 33030144 | 5.9155 | 3.1224 |
| E3 | 20 | 148426752 | 10.7172 | 3.7799 |

E3 completed 20 successful updates, 20 attempts, zero overflows. E1 uses an earlier log format without overflow counters; no overflow count is inferred. Benchmark loss decreased, but these runs do not establish detection F1. Raw logs are in `report/benchmarks/`.

The training label audit excludes one zero-height crack target in `vlcsnap-00058`. Positive-area boxes that collapse after rounding are expanded to one coordinate unit if encountered. Original images, labels, splits and official scoring scripts remain unchanged.

## Notebook usage

`day10_kaggle.ipynb`: original setup notebook; later fixes are consolidated in the E1 notebook below.

`day10_kaggle_e1.ipynb`: independent E1 notebook for a restarted Kaggle kernel. Enable Internet, select T4 GPUs and attach the road-damage dataset. It removes unused incompatible torchao, verifies ViT/lm_head are not quantized, masks prompt loss, trains LLM LoRA for 300 successful updates, reloads the adapter, previews validation outputs and evaluates the complete test split using the official scorer.

Export `day10_results.zip` and `day10_e1_checkpoint.zip` to local storage. Checkpoints do not contain optimizer state and cannot resume interrupted training exactly. A rerun starts from the original model. Save Kaggle outputs before ending the session.

## Remaining work

The new E1 notebook was validated for notebook structure and Python syntax locally, but its full training/test sequence has not yet been verified on Kaggle. Official E0–E4 evidence, prediction illustrations and the final REPORT.md are still outstanding. Benchmark files are not submission evidence. Do not treat this repository as ready to submit.

Implementation and debugging were assisted by Codex; benchmark numbers are from actual user-provided Kaggle outputs.
