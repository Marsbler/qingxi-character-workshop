from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.paths import CONFIGS


@dataclass(frozen=True)
class Affinity:
    id: str
    name_zh: str
    name_en: str
    description: str
    visual_keywords: tuple[str, ...]


@dataclass(frozen=True)
class SpiritDomain:
    name_zh: str
    name_en: str
    description: str


@dataclass(frozen=True)
class WorldConfig:
    world_name: str
    world_name_en: str
    tagline: str
    ip_policy: str
    affinities: tuple[Affinity, ...]
    spirit_domain: SpiritDomain
    style_art: str
    style_avoid: str
    banned_substrings: tuple[str, ...]

    def affinity_zh_names(self) -> list[str]:
        return [a.name_zh for a in self.affinities]

    def affinity_by_zh(self, name_zh: str) -> Affinity | None:
        for a in self.affinities:
            if a.name_zh == name_zh:
                return a
        return None

    def contains_banned(self, text: str) -> bool:
        return any(b and b in text for b in self.banned_substrings)


def _parse(data: dict[str, Any]) -> WorldConfig:
    aff = tuple(
        Affinity(
            id=a["id"],
            name_zh=a["name_zh"],
            name_en=a["name_en"],
            description=a["description"],
            visual_keywords=tuple(a.get("visual_keywords") or []),
        )
        for a in data["affinities"]
    )
    sd = data["spirit_domain"]
    sg = data.get("style_guide") or {}
    return WorldConfig(
        world_name=data["world_name"],
        world_name_en=data.get("world_name_en") or data["world_name"],
        tagline=data["tagline"],
        ip_policy=data["ip_policy"].strip(),
        affinities=aff,
        spirit_domain=SpiritDomain(sd["name_zh"], sd["name_en"], sd["description"]),
        style_art=sg.get("art", ""),
        style_avoid=sg.get("avoid", ""),
        banned_substrings=tuple(data.get("banned_substrings") or []),
    )


@lru_cache(maxsize=1)
def load_world(path: str | None = None) -> WorldConfig:
    p = Path(path) if path else CONFIGS / "world.yaml"
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _parse(data)