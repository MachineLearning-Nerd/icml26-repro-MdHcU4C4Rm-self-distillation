# Limitations and falsification scope


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_223c0c9fe785", "created_at": "2026-07-16T16:06:45+00:00", "title": "What is and is not established"}
-->
# Limitations and negative evidence

- Claim 1 is falsified only **as worded by the challenge**. The paper's actual
  nonstationary theorem is supported, not contradicted.
- The experiments are synthetic Gaussian ridge problems. This is appropriate
  for the three scored mathematical claims, but the paper's four real-dataset
  curvature experiments and multi-round extensions were not rerun.
- One-shot consistency is finite-scale evidence through p=400, not a proof of
  the asymptotic theorem. Exact formulas, multiple dimensions, raw seeds, and
  regret identities make the evidence independently auditable.
- At the isotropic transition lambda=0.5 the limiting optimal weight is zero;
  finite-sample sign is unstable by construction. We evaluate magnitude and
  excess risk there and reserve sign accuracy for penalties separated from the
  boundary.
- The official repository is pinned for provenance and source comparison only;
  the reproduction deliberately does not import it.
