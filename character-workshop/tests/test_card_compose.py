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


def test_font_supports_cjk_check_on_default():
    from PIL import ImageFont
    from src.card_compose import _font_supports_cjk
    # PIL default font has no CJK coverage — check must return False
    assert _font_supports_cjk(ImageFont.load_default()) is False


def test_compose_card_no_crash_without_cjk(tmp_path, monkeypatch):
    # simulate no CJK font anywhere
    monkeypatch.setattr("src.card_compose._CACHED_FONT_PATH", "")
    monkeypatch.setattr("src.card_compose._resolve_font_path", lambda: None)
    from src.llm_role import generate_character
    from PIL import Image
    from src.card_compose import compose_card, has_cjk_font
    card = generate_character("测试", mock=True)
    assert has_cjk_font() is False
    out = tmp_path / "card.png"
    compose_card(card, Image.new("RGB", (768, 1024), (40, 60, 90)), out)
    assert out.exists()


def test_display_content_english_fallback():
    from src.card_compose import _display_content
    from src.llm_role import generate_character
    card = generate_character("空系旅人", affinity_pref="空", mock=True)
    d = _display_content(card, english=True)
    # all strings must be pure ASCII (no tofu possible with default font)
    for s in [d["title"], d["subtitle"], d["footer"]]:
        assert s and all(ord(c) < 128 for c in s)
    for label, value in d["entries"]:
        assert all(ord(c) < 128 for c in label)
        assert all(ord(c) < 128 for c in value)
    assert d["radar_labels"] and len(d["radar_labels"]) == 6


def test_display_content_zh_mode():
    from src.card_compose import _display_content
    from src.llm_role import generate_character
    card = generate_character("空系旅人", affinity_pref="空", mock=True)
    d = _display_content(card, english=False)
    assert d["title"] == card.name
    assert d["radar_labels"] is None


def test_compose_card_english_mode_renders(tmp_path, monkeypatch):
    monkeypatch.setattr("src.card_compose._CACHED_FONT_PATH", "")
    monkeypatch.setattr("src.card_compose._resolve_font_path", lambda: None)
    from PIL import Image
    from src.llm_role import generate_character
    from src.card_compose import compose_card
    card = generate_character("空系旅人", affinity_pref="空", mock=True)
    out = tmp_path / "card_en.png"
    compose_card(card, Image.new("RGB", (768, 1024), (40, 60, 90)), out)
    assert out.exists()
    im = Image.open(out)
    assert im.size[0] >= 800