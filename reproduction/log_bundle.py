#!/usr/bin/env python3
from pathlib import Path
import json
import trackio

ROOT = Path(__file__).resolve().parents[1]
run = trackio.init(
    project="unconstrained-self-distillation-repro",
    name="exact-cpu-three-claim-reproduction",
    config={"openreview_id": "MdHcU4C4Rm", "claims": 3, "device": "cpu", "gpu_used": False},
    embed=False, auto_log_gpu=False, auto_log_cpu=False,
)
artifact = trackio.Artifact(
    "unconstrained-self-distillation-cpu-reproduction",
    type="dataset",
    description="Independent exact-risk CPU reproduction with stationary counterexamples, one-shot trials, raw evidence, tests, source audit, and hashes.",
)
artifact.add_dir(ROOT / "reproduction", name="reproduction")
artifact.add_dir(ROOT / "outputs", name="outputs")
artifact.add_dir(ROOT / "source" / "official-snapshot", name="source/official-snapshot")
for name in ("paper.pdf", "claims.json", "SOURCE_AUDIT.md", "README.md"):
    artifact.add_file(ROOT / name, name=name)
logged = trackio.log_artifact(artifact, aliases=["challenge", "cpu", "complete"])
trackio.finish()
print(json.dumps({"artifact": logged.qualified_name, "files": len(logged.manifest or []), "size": logged.size}, sort_keys=True))
