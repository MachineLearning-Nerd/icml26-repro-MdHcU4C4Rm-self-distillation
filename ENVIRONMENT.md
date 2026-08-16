# Reproduction environment

This is the recorded environment and artifact boundary for the current audit.
uv.lock is the dependency authority.

| Field | Value |
| --- | --- |
| Python | 3.12.13 in the recorded summary |
| NumPy | 2.3.5 |
| SciPy | 1.17.1 |
| Pandas | pinned by uv.lock |
| GPU | none in the recorded summary |
| Baseline compute | local CPU, approximately 30 seconds |
| Claim 3 compute | historical Hugging Face cpu-upgrade run |
| Claim 6 compute | historical local arm CPU run after HF credits were exhausted |
| Official code | provenance only; not imported |

## Commands

~~~bash
bash reproduction/run.sh
~~~

The runner executes uv sync --frozen, reproduction/reproduce.py, and the
nine-test fail-closed suite. The current checkout is expected to fail before
the suite can initialize because the Claim 3 and Claim 6 raw CSV artifacts
are absent. No claim is upgraded by a command that has not completed from the
current tree.
