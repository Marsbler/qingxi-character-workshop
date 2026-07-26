# character-workshop/tests/test_video_gen_mock.py
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video_gen import generate_ability_video


def test_mock_video(tmp_path):
    portrait = tmp_path / "p.png"
    Image.new("RGB", (512, 512), (10, 20, 40)).save(portrait)
    out = tmp_path / "a.mp4"
    path, err = generate_ability_video(
        portrait_path=portrait,
        motion_prompt="swirling rift particles",
        out_path=out,
        mock=True,
    )
    assert err is None
    assert path is not None and path.exists()
