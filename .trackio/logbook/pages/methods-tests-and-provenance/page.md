# Methods, tests, and provenance


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_methods_v2", "created_at": "2026-07-25T00:00:00+00:00", "title": "Independent exact-risk protocol + reproducible command/env"}
-->
# Methods and provenance

**Model.** For `y = Xβ + ε`, ridge uses `β̂ = (XᵀX/n + λI)⁻¹Xᵀy/n`; the pure-distilled coefficient applies the same smoother a second time. Synthetic-claim risks are **exact conditional population risks** `(β̂−β)ᵀΣ(β̂−β) + σ²` — no held-out Monte Carlo can hide or create a gain.

**Fixed run command (identical on every node):**
```bash
bash reproduction/run.sh
```
which boots `uv`, runs `uv sync --frozen` (pinned `uv.lock`), then `uv run python reproduction/reproduce.py`, then `uv run python -m unittest reproduction.test_reproduction`.

**Pinned environment:** Python 3.12, numpy 2.3.5, scipy 1.17.1, pandas 3.0.3, matplotlib 3.11.0, ucimlrepo (see `pyproject.toml` / `uv.lock`). Single-threaded BLAS (`OMP_NUM_THREADS=1` etc., set in `run.sh`) for reproducible numerics across platforms.

**Per-claim code (all in `reproduction/`, independent — imports no official module):**
- Claims 1, 2, 4, 5: `reproduce.py` (structural, stationary-root, isotropic-limit, one-shot GCV, brute-force).
- Claim 3 (Theorem 3.1): `claim3_asymptotics.py` — deterministic equivalents from Eqs. 12–16 + convergence sweep.
- Claim 6 (Prop 2.3): `claim6_curvature.py` — four-dataset curvature test; CIFAR-10 ResNet-18 features cached in `data/cifar10_resnet18_features.npz` (regenerable via `extract_cifar10_features.py`).

**Seeds:** synthetic structural/one-shot and the asymptotic sweep use explicit `np.random.default_rng`/`legacy seed` streams; real-data splits reproduce the official `np.random.seed(2026)` permutations.

**Compute / runtime:** Claims 1,2,4,5 in ~4 s; Claim 3 on HF cpu-upgrade (163 s, single-core); Claim 6 on local arm CPU (178 s; HF credits exhausted). No GPU used.

**Verifiers.** 9 fail-closed unittests in `reproduction/test_reproduction.py` (Theorem 2.2 identities; stationary boundary; 1408 strict gains; sign rule + transition; one-shot contraction; brute-force argmins; Claim 3 convergence + reduction factors; Claim 6 held-out gain matches Table 2 + curvature boundary; CPU-only metadata). All exit nonzero on evidence failure.

**Provenance.** Paper PDF SHA-256 `5ee256f603d43dec7ffc8e67971e738ef1085f99f54528cb544baf4dcbd8c2c2`. Official repo pinned at `7215dda72fc63149fca730248bebc34aa4d3cc8b` (provenance only; not imported). Winning branch `orx/claim-6-proposition-2-3-curvature-test-on-four-r`, Git SHA `80dfd70`.
