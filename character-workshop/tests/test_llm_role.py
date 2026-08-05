import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_role import generate_character
from src.models_schema import CharacterCard


def test_mock_generate_character():
    card = generate_character(
        user_text="喜欢折纸的少年，空系",
        affinity_pref="空",
        mock=True,
    )
    assert isinstance(card, CharacterCard)
    assert card.primary_affinity == "空"
    assert card.image_prompt


def test_mock_card_has_english_fields():
    from src.llm_role import generate_character
    card = generate_character("生系治愈师", affinity_pref="生", mock=True)
    assert card.name_en
    assert card.one_liner_en
    assert card.lore_en
    assert all(ord(c) < 128 for c in card.lore_en)