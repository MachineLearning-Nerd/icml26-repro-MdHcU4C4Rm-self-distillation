# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_conc_v2", "created_at": "2026-07-25T00:00:00+00:00", "title": "Final six-claim outcomes"}
-->
# Conclusion

- **Claim 1 (Theorem 2.2): VERIFIED.** 1408/1408 nonstationary cases strictly improve; sign rule and identity hold; 64 stationary penalties are the equality boundary.
- **Claim 2 (negative mixing): VERIFIED.** 737 negative optima; sign rule 1408/1408; exact isotropic transition at `λ*=0.5`.
- **Claim 3 (Theorem 3.1 anisotropic asymptotics): VERIFIED.** Finite-sample `R, R_pd, C, ξ, R_sd` converge to the independently-reimplemented deterministic equivalents (MAD shrinks 3.7×–4.2× as p grows).
- **Claim 4 (Corollary 3.2 sign transition): VERIFIED.** Sign flips exactly at `λ*=γσ²/r²`.
- **Claim 5 (Theorem 4.1 one-shot GCV): VERIFIED.** 288 fits, weight error and excess risk shrink with p, no grid/split/refit.
- **Claim 6 (Proposition 2.3 curvature): VERIFIED.** Non-circular held-out global gain matches Table 2 on all four real datasets; curvature condition matches on the clear cases (Communities is a boundary case, `D/RHS=0.884`).

All raw evidence, the fail-closed test suite (9/9 PASS), source pins, and limitations are reproducible from `bash reproduction/run.sh`. The earlier 3-claim logbook is preserved as the [Historical rejected baseline](#/historical-rejected-baseline).
