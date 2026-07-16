# Claim 2 - Negative optimal mixing


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_85cd8224b2c7", "created_at": "2026-07-16T16:06:40+00:00", "title": "VERIFIED with two independent identities"}
-->
## Exact scored claim

> Optimal mixing weight can surprisingly be negative.

## Verdict: VERIFIED

The optimum is calculated twice: directly as `(R-C)/D`, and independently as
`-lambda R'/(2D)`. Across all 1408 native-scale
evaluations, the maximum discrepancy is `1.439e-13`.
There are **737 negative** and **671
positive** optima. Every nonstationary case satisfies
`sign(xi*)=-sign(R')`.

An orthogonal/isotropic expectation control fixes `gamma=0.5`, `sigma^2=r^2=1`.
The analytic risk derivative changes sign at the paper's predicted
`lambda*=gamma sigma^2/r^2=0.5`: all evaluated penalties below have positive
mixing, all above have negative mixing, and at 0.5 both weight and gain are
exactly zero. Negative weights are therefore systematic over-regularization
corrections, not numerical accidents.
