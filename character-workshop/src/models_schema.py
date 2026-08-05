from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.world import load_world


class CharacterCard(BaseModel):
    name: str
    one_liner: str
    appearance: str
    personality: str
    backstory: str
    primary_affinity: str
    affinities: dict[str, int]
    spirit_domain: str
    ability_showcase: str
    image_prompt: str
    image_negative: str = ""
    motion_prompt: str = ""

    @field_validator("name", "one_liner", "appearance", "image_prompt")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("field must be non-empty")
        return v

    @model_validator(mode="after")
    def check_affinities(self) -> CharacterCard:
        world = load_world()
        names = world.affinity_names()
        if self.primary_affinity not in names:
            raise ValueError(f"primary_affinity must be one of {names}")
        fixed: dict[str, int] = {}
        for n in names:
            raw = int(self.affinities.get(n, 0))
            fixed[n] = max(0, min(100, raw))
        self.affinities = fixed
        return self

    def lore_markdown(self) -> str:
        lines = [
            f"# {self.name}",
            f"*{self.one_liner}*",
            "",
            f"**Affinity:** {self.primary_affinity}",
            f"**Appearance:** {self.appearance}",
            f"**Personality:** {self.personality}",
            f"**Backstory:** {self.backstory}",
            f"**Spirit Domain:** {self.spirit_domain}",
            f"**Ability Showcase:** {self.ability_showcase}",
            "",
            "**Six Affinities:** "
            + ", ".join(f"{k} {v}" for k, v in self.affinities.items()),
        ]
        return "\n".join(lines)


def extract_json_object(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return text[start : end + 1]


def parse_character_json(text: str) -> CharacterCard:
    raw = extract_json_object(text)
    data: dict[str, Any] = json.loads(raw)
    return CharacterCard.model_validate(data)