import marimo

__generated_with = "0.0.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Optimal self-distillation in ridge regression

        **Paper:** *Optimal Unconstrained Self-Distillation in Ridge Regression* (`MdHcU4C4Rm`, arXiv `2602.17565`).

        A ridge *teacher* $\hat\beta=(X^\top X/n+\lambda I)^{-1}X^\top y/n$ is
        "self-distilled" by applying its smoother a second time to get a
        *pure-distilled* student $\tilde\beta=M\hat\beta$. The best **mixed**
        student $\xi\tilde\beta+(1-\xi)\hat\beta$ has a closed form.

        This notebook walks through the central claim — that the optimally mixed
        student strictly improves the teacher at every nonstationary $\lambda$
        (Theorem 2.2) — and opens with the already-produced evidence. It does
        **not** rerun the expensive experiments; numbers are reproduced inline
        from the committed outputs.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Headline result (from `outputs/full/summary.json`)

        The strict-improvement identity $\xi^\star=-\lambda R'/(2D)$ holds and the
        sign rule $\mathrm{sign}(\xi^\star)=-\mathrm{sign}(R')$ passes on every
        nonstationary case:

        | check | value |
        |---|---|
        | nonstationary cases that strictly improve | **1408 / 1408** |
        | minimum strict gain | `1.95e-06` |
        | identity error $\|xi_{\rm direct}-xi_{\rm deriv}\|$ | `1.4e-13` |
        | negative optimal weights (pro-learning) | **737** |
        | isotropic sign transition | exactly at $\lambda^\star=\gamma\sigma^2/r^2=0.5$ |
        """
    )
    return


@app.cell
def _():
    # The reproduction runs from one fixed command on every node:
    #   bash reproduction/run.sh
    # which does: uv sync --frozen  ->  reproduction/reproduce.py  ->  unittests.
    #
    # Claim 3 (Theorem 3.1 anisotropic asymptotics) is verified by showing
    # finite-sample R, R_pd, C, xi, R_sd converge to the deterministic
    # equivalents as p grows (mean abs. deviation shrinks 3.7x-4.2x, p=50->800).
    # Claim 6 (Proposition 2.3 curvature) is verified on all four real datasets,
    # with a non-circular held-out global gain that matches Table 2.
    summary = {
        "claim_1": {"nonstationary_strict_gains": 1408, "min_gain": 1.949e-06},
        "claim_2": {"negative_weights": 737, "isotropic_transition": 0.5},
        "claim_3": {"mad_reduction_R": 4.18, "mad_reduction_R_sd": 3.67},
        "claim_6": {
            "BlogFeedback": {"held_out_gain_rel": 0.0071, "table2": "gain"},
            "Communities and Crime": {"held_out_gain_rel": -0.0022, "table2": "no gain"},
            "CIFAR10": {"held_out_gain_rel": -0.00001, "table2": "no gain"},
            "Air Quality": {"held_out_gain_rel": 0.3667, "table2": "gain"},
        },
    }
    return (summary,)


@app.cell
def _(mo, summary):
    mo.md(
        f"""
        **Six-claim verdict (this reproduction):** all VERIFIED.

        * Claim 3 — anisotropic Theorem 3.1 equivalents: mean-abs-deviation
          reduction factor R = **{summary['claim_3']['mad_reduction_R']}×**,
          R_sd = **{summary['claim_3']['mad_reduction_R_sd']}×** (p=50→800).
        * Claim 6 — Proposition 2.3 real-data curvature: held-out gain matches
          Table 2 on all four datasets.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
