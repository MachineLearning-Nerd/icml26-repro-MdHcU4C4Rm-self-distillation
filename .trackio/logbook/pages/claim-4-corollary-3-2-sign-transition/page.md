# Claim 4 - Corollary 3.2 isotropic sign transition


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c4", "created_at": "2026-07-25T00:00:00+00:00", "title": "VERIFIED: sign flips exactly at lambda*=gamma*sigma^2/r^2"}
-->
## Exact scored claim

> Under the isotropic-signal setting, the optimal mixing weight changes sign exactly at `λ* := γσ²/r²`, with `ξ*(λ) < 0` for `λ > λ*` and `ξ*(λ) > 0` for `λ < λ*` (Corollary 3.2).

## Verdict: VERIFIED

Isropic control: `γ = 0.5`, `σ² = r² = 1`, so `λ* = γσ²/r² = 0.5`. Evaluating the closed-form isotropic limit over a dense λ-grid:

- `ξ*(λ) > 0` for **every** `λ < 0.5`;
- `ξ*(λ) < 0` for **every** `λ > 0.5`;
- at `λ = 0.5` exactly, both `ξ*` and the gain are **zero** (to machine precision).

Finite-sample structural trials corroborate the transition. Raw data: `isotropic_limit.csv`.
