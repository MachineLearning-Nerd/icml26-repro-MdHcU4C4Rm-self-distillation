#!/usr/bin/env bash
# Fixed run command (identical on every experiment node).
# Portable across the local macOS worktree and the Hugging Face cpu-upgrade
# container (default image python:3.12): it bootstraps uv if missing, recreates
# the pinned environment from uv.lock, runs the reproduction, and runs the
# fail-closed test suite. Set REPRO_OUTPUT_DIR to redirect outputs.
set -euo pipefail

# 1. Ensure uv is available (local mac has it; HF python:3.12 image does not).
if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --quiet uv
fi

# 2. Reproduce the exact pinned environment from the committed lockfile.
uv sync --frozen --quiet

# 3. Run the reproduction (produces every committed claim's raw evidence).
OUTPUT_DIR="${REPRO_OUTPUT_DIR:-outputs/full}"
uv run python reproduction/reproduce.py --output-dir "$OUTPUT_DIR"

# 4. Run the fail-closed verifier suite (exits nonzero on any evidence failure).
uv run python -m unittest -v reproduction.test_reproduction
