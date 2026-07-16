# Source and claim audit

Paper: **Optimal Unconstrained Self-Distillation in Ridge Regression: Strict
Improvements, Precise Asymptotics, and One-Shot Tuning** (`MdHcU4C4Rm`, arXiv
`2602.17565`). The paper PDF is SHA-256
`5ee256f603d43dec7ffc8e67971e738ef1085f99f54528cb544baf4dcbd8c2c2`.

The official repository is pinned at
`7215dda72fc63149fca730248bebc34aa4d3cc8b`. The reproduction is an independent
implementation and imports no official module.

## Exact claim mapping

- Claim 1 compresses Theorem 2.2 too aggressively. Equation (9) states
  `xi* = -lambda R'/(2D)` and `R*_sd = R-lambda^2 R'^2/(4D)`, under `D>0`.
  The theorem's strict inequality explicitly requires `R'(lambda) != 0`, and
  the paper says equality holds at a ridge-optimal stationary penalty. We treat
  root-solved, nondegenerate stationary examples as decisive falsification of
  the scored sentence—not of the theorem—and separately test the corrected
  theorem away from stationary points.
- Claim 2 follows from the same identity: `sign(xi*)=-sign(R')`. Corollary 3.2
  gives the isotropic transition `lambda*=gamma sigma^2/r^2`; our controlled
  case uses `gamma=0.5`, `sigma^2=r^2=1`.
- Claim 3 maps to Equations (17)-(19). We compute the degrees-of-freedom
  corrected residual norms of the teacher smoother `H` and pure-distilled
  smoother `H^2`, then plug their risks and cross-risk into the closed-form
  ratio once. There is no weight grid, data split, or student refit.

Conditional population risk is evaluated exactly as
`(beta_hat-beta)^T Sigma (beta_hat-beta)+sigma^2`, rather than estimated from a
finite test set. Both isotropic and AR(1) anisotropic covariances are included.
