# Unconstrained self-distillation — ICML 2026 reproduction

Independent CPU reproduction for OpenReview `MdHcU4C4Rm` / arXiv `2602.17565`.
It tests exact conditional population risks for ridge teachers and their
one-round pure-distilled students, stationary counterexamples, the corrected
nonstationary theorem, negative optimal mixing, and one-shot GCV tuning.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r reproduction/requirements-cpu.txt
.venv/bin/python reproduction/reproduce.py --output-dir outputs/full
.venv/bin/python -m unittest -v reproduction/test_reproduction.py
```

The scored Claim 1 is falsified as written because it omits the paper's
`R'(lambda) != 0` condition. This does not falsify Theorem 2.2; the corrected
statement is independently stress-tested.
