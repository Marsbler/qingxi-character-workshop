from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml
from PIL import Image

from src.paths import ROOT


def _video_cfg() -> dict:
    p = ROOT / "configs" / "models.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data.get("video") or {}


def generate_ability_video(
    portrait_path: str | Path,
    motion_prompt: str,
    out_path: str | Path,
    mock: bool | None = None,
) -> tuple[Path | None, str | None]:
    """Returns (path, error_message)."""
    if mock is None:
        mock = os.environ.get("MOCK", "0").strip() in {"1", "true", "True", "yes"}
    out_path = Path(out_path)
    portrait_path = Path(portrait_path)
    cfg = _video_cfg()
    if not cfg.get("enabled", True) and not mock:
        return None, "video disabled in config"

    if mock:
        return _mock_video(portrait_path, out_path)

    backend = cfg.get("backend", "auto")
    try:
        if backend in {"auto", "cogvideox"}:
            return _cogvideox(portrait_path, motion_prompt, out_path, cfg), None
    except Exception as e:
        if backend == "cogvideox":
            return None, f"I2V failed: {e}"
        try:
            return _animatediff_placeholder(portrait_path, out_path), None
        except Exception as e2:
            return None, f"I2V failed: {e}; fallback: {e2}"
    return None, "no video backend"


def _mock_video(portrait_path: Path, out_path: Path) -> tuple[Path, None]:
    """Create a short mp4 via ffmpeg if available, else multi-frame gif renamed note."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # try ffmpeg still-image video
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(portrait_path),
        "-t",
        "2",
        "-vf",
        "scale=512:-2,format=yuv420p",
        "-r",
        "8",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path, None
    except (FileNotFoundError, subprocess.CalledProcessError):
        # fallback: save a copy as png sequence message — still create tiny placeholder file
        img = Image.open(portrait_path).convert("RGB")
        # write a minimal valid-enough placeholder: animated webp if possible
        frames = [img]
        try:
            img.save(
                out_path.with_suffix(".webp"),
                save_all=True,
                append_images=frames,
                duration=100,
                loop=0,
            )
            return out_path.with_suffix(".webp"), None
        except Exception:
            dest = out_path.with_suffix(".png")
            img.save(dest)
            return dest, None


def _cogvideox(
    portrait_path: Path, motion_prompt: str, out_path: Path, cfg: dict
) -> Path:
    """Real path — implement fully on Radeon Cloud after smoke test.

    Keep import inside function. Raise on failure so caller can degrade.
    """
    raise NotImplementedError(
        "Wire CogVideoX I2V on ROCm after smoke test; see models.yaml video.model_id"
    )


def _animatediff_placeholder(portrait_path: Path, out_path: Path) -> Path:
    raise NotImplementedError("Wire AnimateDiff fallback after CogVideoX attempt")
