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

## Official E1 completed

Imported user-provided checkpoint and test results. E1: 300 successful updates, zero overflows, 49.3745 minutes, 5.0253 GiB peak allocated VRAM. LoRA adapter has 504 tensors and 33030144 parameters, all LLM LoRA. The adapter remains local in ignored work/checkpoints/e1.

Official test: 311 predictions, zero missing, four invalid outputs. F1 0.2810707, CI95 [0.2430427, 0.3208255], precision 0.3174946, recall 0.2521441, FP/image 1.0160772. Per-class F1: pothole 0.1878453, crack 0.2120718, manhole 0.5158730. These are official scorer results, not a claim of high detection quality.

E0 is still missing; E1 improvement >=0.10 cannot yet be assessed. Added day10_kaggle_e0.ipynb for zero-shot evaluation with the same inference settings. E2/E3/E4, illustrations and final report remain outstanding.

## Official E0 completed

E0: 311 unique test predictions, 81 invalid JSON outputs, zero missing. F1 0.0614887, CI95 [0.0445469, 0.0802568], parse-valid rate 0.7395498, FP/image 1.977492. Dataset version matches E1. E1 absolute F1 improvement is 0.2195821, exceeding the rubric threshold 0.10; E1 parse-valid rate 0.9871383 also exceeds 0.95. E2/E3/E4 remain outstanding.

## Official E2 completed

E2: 300 successful updates, zero overflows, batch 2 x accumulation 4. Early stopping was enabled but did not trigger. Training 41.5096 minutes, peak allocated VRAM 11.1956 GiB, trainable parameters 142135296. Test: 311 unique predictions, all parse-valid, F1 0.3494293, CI95 [0.3095624, 0.3924731], FP/image 1.147910. Per-class F1: pothole 0.212766, crack 0.280315, manhole 0.631970. E2 increases F1 over E1 by 0.0683586; individual CI intervals overlap, so no significance claim is made. Timing is not a controlled speedup comparison because batch shape and trainable architecture both differ. E2 checkpoint ZIP has not yet been provided to local.

## E3 early-stopped result imported

E3 checkpoint includes adapter and separate projector.pt. Training stopped at 275 successful updates out of requested 300, zero overflows, batch 2 x accumulation 4, 44.2212 minutes. It is an early-stopped variant, not a completed 300-update run; the official configuration requirement remains unmet for E3. Optimizer state was not saved, so exact continuation is unavailable.

Test: 311 unique predictions, all parse-valid, F1 0.4581882, CI95 [0.4152192, 0.5013429], FP/image 0.9710611. Per-class F1: pothole 0.3109244, crack 0.3987034, manhole 0.7030717. Log reports 7.9236 GiB peak allocated VRAM; this is lower than the E3 batch-1 benchmark and E2, so a controlled resource comparison is not established. Do not infer a VRAM saving from architecture alone.

## Latest E3 and E4 results

Latest user ZIP contains a new E3 300-update run: 46.1739 minutes, zero overflows, F1 0.4598738, CI95 [0.4175824, 0.5004634]. This supersedes the primary E3 evidence. Earlier E3 275-update evidence/predictions were preserved under ignored work/archive/e3_275 and remain tied to the local E3 checkpoint, which has NOT been replaced by a 300-update checkpoint.

E4 10% uses 142 unique train images, 275 successful updates, zero overflows, 46.9843 minutes. F1 0.2635542, CI95 [0.2311631, 0.2959108], FP/image 1.8327974, all 311 outputs parse-valid. E4 100% reuses the earlier E3 275-update result (F1 0.4581882), not the newer 300-update result. Source hash matches the archived evidence. Both ablation sides use 275 actual updates with a 300-update LR schedule. Absolute F1 difference 0.1946339. Training loss on 10% falls to 0.00477 despite poor test F1, consistent with overfitting but not proof of a single cause.

The E4 design was selected after seeing the test performance and uses a data-dependent stopping point from its baseline; disclose this limitation. No claim of independent replication or exact single-factor preregistration. Remaining: prediction illustrations, final report, checks, commits and push.

## Local checkpoint archive completed

Imported E2 and E3 300-update checkpoints supplied by the user. Both run.json files exactly match latest local training logs; each contains LoRA adapter and separate projector.pt. E3 275-update checkpoint moved to ignored work/checkpoints/e3_275_archived. No checkpoint weights are tracked by Git.
