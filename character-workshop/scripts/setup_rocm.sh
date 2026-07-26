#!/usr/bin/env bash
# Placeholder setup notes for AMD Radeon Cloud / ROCm PyTorch.
# Prefer the official AMD ROCm PyTorch wheels that match your image ROCm version.
# See: https://rocm.docs.amd.com/ and Radeon Cloud template docs.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Character Workshop ROCm setup (placeholder)"
echo "1. Use a PyTorch ROCm container/image from Radeon Cloud."
echo "2. Confirm GPU: python -c \"import torch; print(torch.cuda.is_available(), torch.version.hip)\""
echo "3. Install app deps (torch already provided by image):"
echo "     pip install -r requirements.txt"
echo "4. Optional: pin matching torch/vision/audio ROCm wheels if the image is bare."
echo "     # Example only — replace with AMD-published index/URL for your ROCm:"
echo "     # pip install torch torchvision torchaudio --index-url <rocm-wheel-index>"
echo "5. Download models: bash scripts/download_models.sh"
echo "6. export MOCK=0 && python app.py"
echo "Done (no packages installed by this script)."
