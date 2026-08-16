#!/usr/bin/env python3
"""Verify the current, intentionally scoped repository release surface."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_REPO = "MachineLearning-Nerd/icml26-self-distillation-ridge"
EXPECTED_REMOTE = f"https://github.com/{EXPECTED_REPO}.git"
CANONICAL_NAME = "MachineLearning-Nerd"
CANONICAL_EMAIL = "MachineLearning-Nerd@users.noreply.github.com"
EXPECTED_BRANCHES = [
    "main",
    "baseline/judged-8-of-12",
    "proof/claim-3-theorem-3-1",
    "experiment/claim-6-curvature-four-datasets",
]
EXPECTED_PDF_SHA256 = "5ee256f603d43dec7ffc8e67971e738ef1085f99f54528cb544baf4dcbd8c2c2"
CURRENT_DOCS = [
    "README.md",
    "STATUS.md",
    "claims.json",
    "CLAIM_EVIDENCE.md",
    "SOURCE_AUDIT.md",
    "SOURCE_MANIFEST.md",
    "ENVIRONMENT.md",
    "CITATION.cff",
    "reports/self-distillation/report.md",
    "reproduction/ENVIRONMENT.md",
]
MISSING_CURRENT_ARTIFACTS = [
    "outputs/full/claim3_convergence_trials.csv",
    "outputs/full/claim3_convergence.png",
    "outputs/full/claim6_curvature_summary.csv",
    "outputs/full/claim6_curvature.png",
]


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load(path: str):
    return json.loads(read(path))


errors: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


require(not git("status", "--porcelain").strip(), "working tree is not clean")
require(git("symbolic-ref", "--short", "HEAD").strip() == "main", "HEAD is not on main")
require(git("remote", "get-url", "origin").strip().rstrip("/") == EXPECTED_REMOTE, "origin is not final")

local_branches = sorted(
    line.strip()
    for line in git("for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
    if line.strip()
)
require(local_branches == sorted(EXPECTED_BRANCHES), "local branch inventory differs")

for path in CURRENT_DOCS + ["EVIDENCE_MANIFEST.json", "verify_final.py"]:
    require((ROOT / path).is_file(), f"missing required file: {path}")

for path in CURRENT_DOCS:
    text = read(path)
    require(
        "https://github.com/MachineLearning-Nerd/icml26-repro-MdHcU4C4Rm-self-distillation" not in text,
        f"old repository URL remains in current document: {path}",
    )
    require("orx/" not in text, f"legacy branch prefix remains in current document: {path}")

readme = read("README.md")
require(EXPECTED_REPO in readme, "README does not identify final repository")
require("Citation" in readme and "Thank you" in readme, "README missing citation or thanks")
require("BLOCKED_REPRODUCTION_REQUIRED" in readme, "README missing artifact-gap status")

manifest = load("EVIDENCE_MANIFEST.json")
require(manifest["repository"]["final_name"] == "icml26-self-distillation-ridge", "manifest name differs")
require(manifest["branch_inventory"] == EXPECTED_BRANCHES, "manifest branches differ")
require(
    manifest["collection_status"]
    == "PARTIALLY_VERIFIED_WITH_LITERAL_CLAIM_BOUNDARY_AND_UNSEALED_CLAIMS",
    "manifest collection status differs",
)
require(manifest["artifact_gate"]["fresh_test_suite_sealed"] is False, "artifact gate was incorrectly sealed")

claims = load("claims.json")
require([item["id"] for item in claims] == [f"claim_{i}" for i in range(1, 7)], "claim IDs differ")
require(
    [item["status"] for item in claims]
    == [
        "FALSIFIED_AS_WRITTEN; VERIFIED_CORRECTED_NONSTATIONARY_THEOREM",
        "VERIFIED_SCOPED",
        "BLOCKED_REPRODUCTION_REQUIRED",
        "VERIFIED_SCOPED",
        "VERIFIED_SCOPED",
        "BLOCKED_REPRODUCTION_REQUIRED",
    ],
    "claim statuses differ",
)

paper_hash = hashlib.sha256((ROOT / "paper.pdf").read_bytes()).hexdigest()
require(paper_hash == EXPECTED_PDF_SHA256, "paper PDF hash differs")

summary = load("outputs/full/summary.json")
require(summary["claim_1"]["stationary_counterexamples"] == 64, "Claim 1 stationary count differs")
require(summary["claim_1"]["corrected_nonstationary_checks"] == 1408, "Claim 1 corrected count differs")
require(summary["claim_2"]["negative_weights"] == 737, "Claim 2 negative count differs")
require(summary["claim_3"]["fits"] == 288, "Claim 5 one-shot fit count differs")
require("claim_3_asymptotic" not in summary, "unsealed Claim 3 summary unexpectedly present")
require("claim_6_curvature" not in summary, "unsealed Claim 6 summary unexpectedly present")

for path in MISSING_CURRENT_ARTIFACTS:
    require(not (ROOT / path).exists(), f"artifact-gap expectation changed: {path} is present")

test_source = read("reproduction/test_reproduction.py")
require(test_source.count("    def test_") == 9, "nine-test suite shape differs")
for path in [
    "outputs/full/structural_trials.csv",
    "outputs/full/stationary_counterexamples.csv",
    "outputs/full/isotropic_limit.csv",
    "outputs/full/oneshot_trials.csv",
    "outputs/full/oneshot_summary.csv",
]:
    require((ROOT / path).is_file(), f"verified-scope artifact missing: {path}")

identity_lines = git("log", "--all", "--format=%an%x09%ae%x09%cn%x09%ce").splitlines()
for line in identity_lines:
    fields = line.split("\t")
    require(
        len(fields) == 4
        and fields[0] == CANONICAL_NAME
        and fields[1] == CANONICAL_EMAIL
        and fields[2] == CANONICAL_NAME
        and fields[3] == CANONICAL_EMAIL,
        f"non-canonical commit identity: {line}",
    )
require("co-authored-by:" not in git("log", "--all", "--format=%B").lower(), "co-author trailer remains")

if errors:
    print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
    sys.exit(1)

print(
    json.dumps(
        {
            "status": "PASS",
            "repository": EXPECTED_REPO,
            "default_branch": "main",
            "branch_count": len(EXPECTED_BRANCHES),
            "audit_status": manifest["collection_status"],
            "paper_sha256": paper_hash,
            "claim_statuses": [item["status"] for item in claims],
            "fresh_scientific_suite": "NOT_SEALED_ARTIFACT_GAP",
            "canonical_identity": f"{CANONICAL_NAME} <{CANONICAL_EMAIL}>",
        },
        indent=2,
    )
)
