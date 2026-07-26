#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
from huggingface_hub import snapshot_download
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/models.yaml").read_text())
# downloads to HF cache; optional local_dir under models/
print("llm", cfg["llm"]["model_id"])
snapshot_download(cfg["llm"]["model_id"])
print("image", cfg["image"]["model_id"])
snapshot_download(cfg["image"]["model_id"])
print("done")
PY
