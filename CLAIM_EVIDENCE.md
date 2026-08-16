# Claim-to-evidence map

This file records the production chain for every paper claim. A status is
terminal only when the current checkout contains the artifact needed to
recreate the stated result.

## Shared production model

1. The paper PDF and claim anchor are pinned.
2. reproduction/reproduce.py computes exact conditional ridge quantities.
3. A claim-specific producer derives the theorem-specific quantity.
4. CSV and JSON outputs are committed under outputs/full.
5. reproduction/test_reproduction.py checks identities, dimensions, signs,
   convergence, and artifact completeness.
6. verify_final.py checks the repository surface without rerunning the costly
   experiment.

Claims 3 and 6 stop at step 4 in the current tree because their raw outputs
are missing. Their historical report and plots remain useful provenance but
are not treated as raw proof.

## Claim 1 — Theorem 2.2

- Source anchor: Theorem 2.2.
- Producer: reproduction/reproduce.py, structural_experiment(), and the
  root-solved stationary sweep.
- Raw outputs: outputs/full/structural_trials.csv and
  outputs/full/stationary_counterexamples.csv.
- Boundary result: 64 nondegenerate stationary penalties have maximum
  absolute gain 2.544043748557646e-28 and maximum absolute optimal weight
  5.207888399099298e-14.
- Corrected result: after requiring R'(lambda) != 0, 1,408/1,408 structural
  cases have positive gain; the minimum is 1.949232328837568e-06.
- Test path: test_claim_one_strict_improvement_and_boundary and
  test_corrected_nonstationary_theorem.
- Verdict: FALSIFIED_AS_WRITTEN for the unconditional summary;
  VERIFIED_CORRECTED_NONSTATIONARY_THEOREM for the correctly conditioned
  theorem.

## Claim 2 — negative mixing and sign rule

- Source anchor: Section 2.3 and Corollary 3.2.
- Producer: RidgeProblem.quantities() computes the exact oracle xi and its
  derivative identity.
- Raw outputs: structural_trials.csv and isotropic_limit.csv.
- Result: 737 negative weights, 671 positive weights, 1,408/1,408 sign-rule
  matches, and maximum closed-form identity error 1.438849039914203e-13.
- Boundary: lambda*=0.5, xi*=0, and gain=0 in the isotropic construction.
- Test path: test_negative_weights_and_exact_transition.
- Verdict: VERIFIED_SCOPED.

## Claim 3 — anisotropic Theorem 3.1

- Source anchor: Theorem 3.1, Equations 12--19.
- Producer: reproduction/claim3_asymptotics.py.
- Method: solve the kappa fixed point, construct trace and alignment
  functionals, evaluate the five deterministic equivalents, and compare them
  with exact finite-sample conditional risks under anisotropic AR(1)
  covariance at fixed gamma=0.5.
- Historical report: records MAD reductions of approximately 3.7x--4.2x
  from p=50 to p=800.
- Current raw gap: outputs/full/claim3_convergence_trials.csv and
  outputs/full/claim3_convergence.png are absent; summary.json has no
  claim_3_asymptotic section.
- Test path: test_claim3_anisotropic_asymptotic_convergence, which cannot
  initialize on the current checkout because it loads the missing CSV.
- Verdict: BLOCKED_REPRODUCTION_REQUIRED.

## Claim 4 — isotropic sign transition

- Source anchor: Corollary 3.2.
- Producer: reproduction/reproduce.py, isotropic_limit().
- Raw output: outputs/full/isotropic_limit.csv.
- Method: evaluate the closed-form risk derivative, discrepancy, optimal xi,
  and gain on a lambda grid containing 0.5 exactly.
- Result: positive xi below 0.5, negative xi above 0.5, and exact zero at 0.5.
- Test path: test_negative_weights_and_exact_transition.
- Verdict: VERIFIED_SCOPED.

## Claim 5 — one-shot GCV

- Source anchor: Theorem 4.1.
- Producer: reproduction/reproduce.py, oneshot_experiment().
- Raw outputs: outputs/full/oneshot_trials.csv and oneshot_summary.csv.
- Method: compare the GCV residual-correction estimate with the exact oracle
  xi for 288 fits over dimensions 50, 100, 200, 400 and penalties 0.1, 0.5,
  and 2.0; no grid search, sample split, or refit.
- Result: all GCV discrepancy values are positive and median weight/excess-risk
  errors shrink with p.
- Test path: test_one_shot_consistency.
- Verdict: VERIFIED_SCOPED.

## Claim 6 — Proposition 2.3

- Source anchor: Proposition 2.3, Equation 10, and Table 2.
- Producer: reproduction/claim6_curvature.py.
- Method: reproduce the curvature ratio at the teacher's optimum, compare
  global teacher and optimal-SD risks, and run the protocol on BlogFeedback,
  Communities and Crime, CIFAR10 ResNet-18 features, and Air Quality.
- Inputs: committed data/cifar10_resnet18_features.npz plus the documented
  external dataset loaders.
- Historical report: records held-out agreement with Table 2 on all four
  datasets and identifies Communities as a boundary case.
- Current raw gap: outputs/full/claim6_curvature_summary.csv and
  outputs/full/claim6_curvature.png are absent; summary.json has no
  claim_6_curvature section.
- Test path: test_claim6_real_data_curvature, which cannot initialize on the
  current checkout because it loads the missing CSV.
- Verdict: BLOCKED_REPRODUCTION_REQUIRED.
