# character-workshop/tests/test_card_compose.py
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.card_compose import compose_card, draw_radar, has_cjk_font
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


def test_has_cjk_font_returns_bool():
    assert isinstance(has_cjk_font(), bool)


def test_compose_card_wraps_long_text(tmp_path):
    card = generate_character("测试角色 空系", affinity_pref="空", mock=True)
    # Override fields with long Chinese strings to exercise pixel-width wrapping
    card = card.model_copy(
        update={
            "appearance": "长发及腰" * 30,
            "personality": "冷静沉着" * 25,
            "backstory": "在云雾缭绕的山谷中长大" * 20,
            "ability_showcase": "释放空系灵力形成风暴" * 20,
            "one_liner": "由梦境启发的原创角色" * 10,
        }
    )
    portrait = Image.new("RGB", (768, 1024), (40, 60, 90))
    out = tmp_path / "card_long.png"
    compose_card(card, portrait, out)
    assert out.exists()
    im = Image.open(out)
    assert im.size >= (1400, 900)