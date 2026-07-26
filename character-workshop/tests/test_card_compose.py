# character-workshop/tests/test_card_compose.py
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.card_compose import compose_card, draw_radar
from src.llm_role import generate_character


def test_draw_radar_and_compose(tmp_path):
    card = generate_character("测试角色 空系", affinity_pref="空", mock=True)
    portrait = Image.new("RGB", (768, 1024), (40, 60, 90))
    radar = draw_radar(card.affinities, size=400)
    assert radar.size == (400, 400)
    out = tmp_path / "card.png"
    compose_card(card, portrait, out)
    assert out.exists()
    im = Image.open(out)
    assert im.size[0] >= 800
