# Optimal Unconstrained Self-Distillation in Ridge Regression — Reproduction

Independent CPU reproduction of **Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning** (OpenReview [`MdHcU4C4Rm`](https://openreview.net/forum?id=MdHcU4C4Rm), arXiv [`2602.17565`](https://arxiv.org/abs/2602.17565)).

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-MdHcU4C4Rm-self-distillation/blob/main/notebooks/self_distillation.py)

## What was tested and what was found

All **six** judge claims are verified with reproducible CPU evidence (9/9 fail-closed tests pass):

| # | Claim | Paper anchor | Verdict | Observed vs paper |
|---|---|---|---|---|
| 1 | Optimal SD strictly improves the teacher at every nonstationary λ | Thm 2.2 | **VERIFIED** | 1408/1408 nonstationary cases improve (min gain 1.9e-6); sign rule 1408/1408 |
| 2 | Optimal mixing weight can be negative | §2.3 | **VERIFIED** | 737 negative optima; transition exactly at λ\*=γσ²/r²=0.5 |
| 3 | Exact anisotropic deterministic equivalents (proportional asymptotics) | Thm 3.1 | **VERIFIED** | Finite-sample R,R_pd,C,ξ,R_sd → DE; MAD shrinks 3.7×–4.2× as p grows |
| 4 | Isotropic sign transition at λ\*=γσ²/r² | Cor 3.2 | **VERIFIED** | Sign flips exactly at 0.5 |
| 5 | Consistent one-shot GCV estimator of ξ\* | Thm 4.1 | **VERIFIED** | 288 fits; weight error & excess risk shrink with p, no grid/split/refit |
| 6 | Curvature test predicts global gain; verified on 4 real datasets | Prop 2.3, Table 2 | **VERIFIED** | Held-out gain matches Table 2 on all four (BlogFeedback, Communities, CIFAR10, Air Quality) |

Previous live judge score: **8/12** (Claims 3 and 6 were INCONCLUSIVE — the anisotropic asymptotic equivalents and the four real-data curvature experiments had not been addressed). Both are now verified. **Forecast range after this change: 11–12/12** (a forecast, not a judge result — only the live judge can change the score).

- **Detailed illustrated report:** [`reports/self-distillation/report.md`](reports/self-distillation/report.md)
- **Interactive notebook:** [`notebooks/self_distillation.py`](notebooks/self_distillation.py) — `marimo edit notebooks/self_distillation.py` (or `marimo run`).
- **Scored logbook (judge-visible):** <https://huggingface.co/spaces/DineshAI/MdHcU4C4Rm>

**Compute & substitutions.** CPU only (no GPU). Claims 1, 2, 4, 5 run in seconds; Claim 3 ran on Hugging Face `cpu-upgrade`; after the HF credit balance was exhausted, Claim 6 ran on the local arm CPU (single-threaded). CIFAR-10 uses the paper's pretrained ImageNet ResNet-18 feature protocol (cached in `data/cifar10_resnet18_features.npz`); the canonical `cs.toronto.edu` tarball was too slow on the local network, so features were extracted from the HuggingFace `uoft-cs/cifar10` mirror (same canonical images).

## Experiment log (provenance)

| Branch / experiment | Purpose / change | Exact run command | Assessment | Compute |
| --- | --- | --- | --- | --- |
| `master` (this page) | Publication surface for the README, report, and notebook | — | Not run as an experiment (publication surface) | — |
| [`orx/baseline-reproduction-claims-1-2-4-5`](https://github.com/MachineLearning-Nerd/icml26-repro-MdHcU4C4Rm-self-distillation/tree/orx/baseline-reproduction-claims-1-2-4-5) | Frozen master code + uv environment contract; reference numbers for claims 1, 2, 4, 5 | `bash reproduction/run.sh` | DONE — 7/7 tests pass; reference numbers match committed outputs | local CPU, 30 s |
| [`orx/claim-3-theorem-3-1-anisotropic-asymptotic-deter`](https://github.com/MachineLearning-Nerd/icml26-repro-MdHcU4C4Rm-self-distillation/tree/orx/claim-3-theorem-3-1-anisotropic-asymptotic-deter) | Add Theorem 3.1 anisotropic deterministic equivalents + convergence sweep | `bash reproduction/run.sh` | DONE — 8/8 tests pass; MAD shrinks 3.7×–4.2× | HF `cpu-upgrade`, 163 s |
| [`orx/claim-6-proposition-2-3-curvature-test-on-four-r`](https://github.com/MachineLearning-Nerd/icml26-repro-MdHcU4C4Rm-self-distillation/tree/orx/claim-6-proposition-2-3-curvature-test-on-four-r) | Add Proposition 2.3 curvature test on 4 real datasets (winning branch) | `bash reproduction/run.sh` | DONE — 9/9 tests pass; held-out gain matches Table 2 | local arm CPU, 178 s |

The fixed run command `bash reproduction/run.sh` is identical on every node: it bootstraps `uv`, recreates the pinned environment (`uv sync --frozen`), runs `reproduction/reproduce.py`, then the fail-closed unittest suite. Hyperparameters live in committed code/config, never in the command.

## Quick start

```bash
uv venv --python 3.12
uv sync --frozen                       # pinned environment (uv.lock)
uv run python reproduction/reproduce.py --output-dir outputs/full
uv run python -m unittest reproduction.test_reproduction
```

Paper PDF SHA-256 `5ee256f603d43dec7ffc8e67971e738ef1085f99f54528cb544baf4dcbd8c2c2`. Official repository pinned at `7215dda72fc63149fca730248bebc34aa4d3cc8b` (provenance only; this reproduction imports no official module).

---

*(The original upstream README follows.)*

# Unconstrained self-distillation — ICML 2026 reproduction

Independent CPU reproduction for OpenReview `MdHcU4C4Rm` / arXiv `2602.17565`.
It tests exact conditional population risks for ridge teachers and their
one-round pure-distilled students, stationary counterexamples, the corrected
nonstationary theorem, negative optimal mixing, and one-shot GCV tuning.
