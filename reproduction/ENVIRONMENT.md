# Execution environment

The locked dependency environment is defined by uv.lock and pyproject.toml.

- Python 3.12; the recorded summary used Python 3.12.13.
- NumPy 2.3.5.
- SciPy 1.17.1.
- CPU-only recorded summary; no GPU.
- Baseline computation was local CPU.
- The historical Claim 3 producer ran on Hugging Face cpu-upgrade.
- The historical Claim 6 producer ran on a local arm CPU after HF credits
  were exhausted.
- Official code commit 7215dda72fc63149fca730248bebc34aa4d3cc8b is provenance
  only and is not imported.

The fixed runner is:

~~~bash
bash reproduction/run.sh
~~~

It runs uv sync --frozen, reproduction/reproduce.py, and the nine-test
fail-closed suite. On the current checkout, the suite cannot initialize until
the missing Claim 3 and Claim 6 output files are regenerated.
