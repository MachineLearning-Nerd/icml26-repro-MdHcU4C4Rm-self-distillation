# Branch audit

The public branch names describe the scientific role of each route. The
legacy names and tips are preserved so the cleanup can be independently
checked.

## Final public inventory

The intended final inventory is exactly:

1. main
2. baseline/judged-8-of-12
3. proof/claim-3-theorem-3-1
4. experiment/claim-6-curvature-four-datasets

## Legacy-to-final mapping

The tips below are the legacy tips before canonical author/committer rewrite.

| Final branch | Legacy branch | Legacy tip before rewrite | Role |
| --- | --- | --- | --- |
| main | master | 8b1cb509bcc45faba8379206ca005c48eaf1d687 | Publication surface |
| baseline/judged-8-of-12 | orx/baseline-reproduction-claims-1-2-4-5 | 5bce4b418a958a16485c402a344270bad4c1fc1b | Baseline Claims 1, 2, 4, 5 |
| proof/claim-3-theorem-3-1 | orx/claim-3-theorem-3-1-anisotropic-asymptotic-deter | 58c2f48cb5a2a55d2b50b1691f560edb63049827 | Theorem 3.1 producer |
| experiment/claim-6-curvature-four-datasets | orx/claim-6-proposition-2-3-curvature-test-on-four-r | cdd2eeaff8df1b68105ab82a930240024af5a41c | Proposition 2.3 producer |

The baseline branch intentionally preserves the earlier claims while the two
specialized branches carry the later producer code. The main branch is the
publication and audit surface; it does not imply that every specialized
producer's raw output is present on main.

## Cleanup invariants

- All reachable commits use author and committer
  MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>.
- No reachable commit contains a Co-authored-by: trailer.
- GitHub default branch is main.
- The public branch inventory contains only the four final names above.
- Legacy automation-prefixed branch refs are deleted after publication.
