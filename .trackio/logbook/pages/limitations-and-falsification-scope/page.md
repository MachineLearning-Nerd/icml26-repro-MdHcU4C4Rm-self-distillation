# Limitations and falsification scope


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_lim_v2", "created_at": "2026-07-25T00:00:00+00:00", "title": "What is and is not established"}
-->
# Limitations and deviations

- **Claim 1** is verified under Theorem 2.2's explicit `R'(λ)≠0` condition; at a root-solved stationary penalty the gain is exactly zero (the equality the theorem predicts), retained as boundary evidence.
- **Claim 3** convergence is finite-scale corroboration through p=800 (not a proof of the asymptotic theorem); the deterministic equivalents are independently reconstructed and cross-checked against the official snapshot to 8 decimals.
- **Claim 6** — the curvature condition is a continuous ratio; **Communities & Crime is a boundary case** (`D/RHS = 0.884`, right at the decision boundary; CIFAR10's 1.017 is just on the other side). Its binary curvature classification is therefore sample-sensitive; the **non-circular held-out global gain (−0.22%, no gain) matches Table 2** and is the robust outcome. The official full-test gain is optimistically biased (ξ fit and evaluated on the same test set); the held-out split is reported as the rigorous measure.
- **CIFAR-10 features** use the official pretrained-ImageNet ResNet-18 protocol (FC→Identity, 512-dim, 2000/2000 seed-2026 subsample). The canonical `cs.toronto.edu` CIFAR-10 tarball was unreachable at viable speed on the local network, so features were extracted from the HuggingFace `uoft-cs/cifar10` mirror (same canonical images) and cached as `data/cifar10_resnet18_features.npz`; `reproduction/extract_cifar10_features.py` documents the exact regenerable torchvision path.
- **Compute:** Claim 3 ran on HF cpu-upgrade; after the HF credit balance was exhausted, Claim 6 ran on the local arm CPU (single-threaded). No GPU was used anywhere.
- The official repository is pinned for provenance only; the reproduction imports no official module.
