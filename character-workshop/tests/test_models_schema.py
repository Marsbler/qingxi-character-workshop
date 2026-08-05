import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models_schema import CharacterCard, parse_character_json
from src.world import load_world


def _sample() -> dict:
    w = load_world()
    aff = {a.name: 50 for a in w.affinities}
    aff["Void"] = 90
    return {
        "name": "Paperfold Boy",
        "one_liner": "A spirit user who folds alleys into shortcuts",
        "appearance": "Black hair, teal coat, faint rift light on fingertips",
        "personality": "Quiet and observant",
        "backstory": "Grew up on the edge of the Qingxi Rift, used space folds for courier work.",
        "primary_affinity": "Void",
        "affinities": aff,
        "spirit_domain": "A paper alley that keeps folding inward",
        "ability_showcase": "Tears open a rift with a wave and vanishes in afterimages",
        "image_prompt": "anime boy, black hair, teal coat, spatial cracks, portrait",
        "image_negative": "blurry, lowres",
        "motion_prompt": "camera slow push-in, purple rift particles swirling",
    }


def test_parse_valid():
    card = parse_character_json(json.dumps(_sample(), ensure_ascii=False))
    assert isinstance(card, CharacterCard)
    assert card.primary_affinity == "Void"
    assert card.affinities["Void"] == 90


def test_rejects_bad_affinity():
    data = _sample()
    data["primary_affinity"] = "Fire"
    with pytest.raises(Exception):
        parse_character_json(json.dumps(data, ensure_ascii=False))


def test_clamps_scores():
    data = _sample()
    data["affinities"]["Form"] = 150
    card = parse_character_json(json.dumps(data, ensure_ascii=False))
    assert card.affinities["Form"] == 100


def test_extra_english_fields_ignored():
    data = _sample()
    data["name_en"] = "Legacy Field"
    data["lore_en"] = "Legacy field should not break validation"
    card = parse_character_json(json.dumps(data, ensure_ascii=False))
    assert card.primary_affinity == "Void"
