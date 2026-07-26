import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models_schema import CharacterCard, parse_character_json
from src.world import load_world


def _sample() -> dict:
    w = load_world()
    aff = {a.name_zh: 50 for a in w.affinities}
    aff["空"] = 90
    return {
        "name": "折纸少年",
        "one_liner": "把巷口折叠成捷径的灵师",
        "appearance": "黑发，青灰外套，指尖常有细小裂隙光",
        "personality": "寡言，观察细致",
        "backstory": "在青汐裂隙边缘长大，习惯用空间折痕送快递。",
        "primary_affinity": "空",
        "affinities": aff,
        "spirit_domain": "内里是一个不断翻折的纸巷。",
        "ability_showcase": "抬手撕开空气裂缝，残影闪过",
        "image_prompt": "anime boy, black hair, teal coat, spatial cracks, portrait",
        "image_negative": "blurry, lowres",
        "motion_prompt": "camera slow push-in, purple rift particles swirling",
    }


def test_parse_valid():
    card = parse_character_json(json.dumps(_sample(), ensure_ascii=False))
    assert isinstance(card, CharacterCard)
    assert card.primary_affinity == "空"
    assert card.affinities["空"] == 90


def test_rejects_bad_affinity():
    data = _sample()
    data["primary_affinity"] = "火"
    with pytest.raises(Exception):
        parse_character_json(json.dumps(data, ensure_ascii=False))


def test_clamps_scores():
    data = _sample()
    data["affinities"]["形"] = 150
    card = parse_character_json(json.dumps(data, ensure_ascii=False))
    assert card.affinities["形"] == 100