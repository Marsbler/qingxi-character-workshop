# Character Workshop (Track 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an AMD Radeon + ROCm local multimodal web workbench that turns a text brief (optional reference image) into an original-fantasy character card (lore + portrait + affinity radar + poster), with optional 2–4s ability I2V.

**Architecture:** Single-process Gradio app calling a job orchestrator. Modules: world config → local LLM JSON → Diffusers portrait (± ref) → Pillow card compose → optional I2V. MOCK mode enables UI development without GPU. Critical inference runs on Radeon Cloud for demo/submission.

**Tech Stack:** Python 3.10+, PyTorch (ROCm on cloud), Transformers, Diffusers, Gradio, Pillow, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-07-26-amd-hackathon-character-workshop-design.md`

**App root:** `character-workshop/` (create under repo root `AMDAIHackathon/`)

---

## File map (create)

```
character-workshop/
  app.py
  requirements.txt
  requirements-dev.txt
  pyproject.toml                 # optional; pytest path ok via requirements-dev
  README.md
  .gitignore
  configs/
    world.yaml
    models.yaml
  prompts/
    system_role.txt
    json_schema.txt
    few_shot.jsonl
    repair_json.txt
  src/
    __init__.py
    paths.py
    device.py
    world.py
    models_schema.py             # Pydantic CharacterCard
    llm_role.py
    image_gen.py
    card_compose.py
    video_gen.py
    orchestrator.py
  tests/
    test_device.py
    test_world.py
    test_models_schema.py
    test_llm_role.py
    test_card_compose.py
    test_orchestrator.py
    test_image_gen_mock.py
    test_video_gen_mock.py
  data/eval/
    cases.jsonl
  scripts/
    setup_rocm.sh
    download_models.sh
    run_eval_json.py
  outputs/                       # gitignore contents
  models/                        # gitignore contents
```

---

### Task 1: Scaffold project + gitignore + dependencies

**Files:**
- Create: `character-workshop/.gitignore`
- Create: `character-workshop/requirements.txt`
- Create: `character-workshop/requirements-dev.txt`
- Create: `character-workshop/src/__init__.py`
- Create: `character-workshop/src/paths.py`

- [ ] **Step 1: Create directories and `__init__.py`**

```bash
mkdir -p character-workshop/src character-workshop/tests character-workshop/configs character-workshop/prompts character-workshop/data/eval character-workshop/scripts character-workshop/outputs character-workshop/models
```

Write `character-workshop/src/__init__.py`:

```python
"""Lingxi Character Workshop — AMD Track 1 multimodal tool."""
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
outputs/*
!outputs/.gitkeep
models/*
!models/.gitkeep
*.mp4
.DS_Store
.env
*.egg-info/
dist/
build/
```

Create empty keep files:

```bash
touch character-workshop/outputs/.gitkeep character-workshop/models/.gitkeep
```

- [ ] **Step 3: Write dependency files**

`character-workshop/requirements.txt`:

```text
gradio>=4.44.0
pydantic>=2.7.0
pyyaml>=6.0.1
Pillow>=10.4.0
numpy>=1.26.0
huggingface_hub>=0.24.0
# Install torch/diffusers/transformers per ROCm or CUDA env — see README
transformers>=4.44.0
diffusers>=0.30.0
accelerate>=0.33.0
safetensors>=0.4.0
sentencepiece>=0.2.0
protobuf>=4.25.0
```

`character-workshop/requirements-dev.txt`:

```text
-r requirements.txt
pytest>=8.3.0
```

- [ ] **Step 4: Write `src/paths.py`**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PROMPTS = ROOT / "prompts"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"


def ensure_runtime_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: Init git if missing (repo root) and commit scaffold**

```bash
cd "D:/APP/Obsidian/Documents/Marbler/主业副业/AMDAIHackathon"
git init
git add character-workshop/docs 2>/dev/null; git add character-workshop .gitignore 2>/dev/null
git add character-workshop
git status
git commit -m "chore: scaffold character-workshop app layout"
```

If git user not configured, set local user only if user asks; otherwise skip commit and continue.

---

### Task 2: Device detection module

**Files:**
- Create: `character-workshop/src/device.py`
- Create: `character-workshop/tests/test_device.py`

- [ ] **Step 1: Write failing tests**

```python
# character-workshop/tests/test_device.py
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.device import DeviceInfo, detect_device, vram_profile


def test_detect_device_returns_info():
    info = detect_device()
    assert isinstance(info, DeviceInfo)
    assert info.device_type in {"cpu", "cuda", "mps"}
    assert isinstance(info.name, str)
    assert info.vram_gb is None or info.vram_gb >= 0


def test_vram_profile_cpu():
    info = DeviceInfo(device_type="cpu", name="cpu", vram_gb=None, rocm_hint=False)
    assert vram_profile(info) == "cpu"


def test_vram_profile_16_and_24():
    assert vram_profile(DeviceInfo("cuda", "x", 15.0, True)) == "16gb"
    assert vram_profile(DeviceInfo("cuda", "x", 22.0, True)) == "24gb"


def test_mock_env_forces_cpu_label(monkeypatch):
    monkeypatch.setenv("MOCK", "1")
    info = detect_device()
    # MOCK still reports real backend if present, but mock_mode True
    assert info.mock_mode is True
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd character-workshop
pip install pytest pydantic pyyaml Pillow numpy -q
pytest tests/test_device.py -v
```

Expected: `ModuleNotFoundError` or import error for `src.device`.

- [ ] **Step 3: Implement `src/device.py`**

```python
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    device_type: str  # cpu | cuda | mps
    name: str
    vram_gb: float | None
    rocm_hint: bool
    mock_mode: bool = False
    torch_version: str | None = None


def detect_device() -> DeviceInfo:
    mock_mode = os.environ.get("MOCK", "0").strip() in {"1", "true", "True", "yes"}
    try:
        import torch
    except ImportError:
        return DeviceInfo("cpu", "torch-not-installed", None, False, mock_mode, None)

    torch_version = getattr(torch, "__version__", None)
    rocm_hint = bool(getattr(torch.version, "hip", None)) or (
        "rocm" in (torch_version or "").lower()
    )

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        vram_gb = round(props.total_memory / (1024**3), 2)
        return DeviceInfo("cuda", name, vram_gb, rocm_hint, mock_mode, torch_version)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo("mps", "apple-mps", None, False, mock_mode, torch_version)

    return DeviceInfo("cpu", "cpu", None, False, mock_mode, torch_version)


def vram_profile(info: DeviceInfo) -> str:
    if info.device_type != "cuda" or info.vram_gb is None:
        return "cpu"
    if info.vram_gb >= 20:
        return "24gb"
    if info.vram_gb >= 10:
        return "16gb"
    return "low"


def torch_device_string(info: DeviceInfo | None = None) -> str:
    info = info or detect_device()
    if info.device_type == "cuda":
        return "cuda"
    if info.device_type == "mps":
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_device.py -v
```

- [ ] **Step 5: Commit**

```bash
git add character-workshop/src/device.py character-workshop/tests/test_device.py
git commit -m "feat: add ROCm/CUDA device detection and VRAM profiles"
```

---

### Task 3: World config loader

**Files:**
- Create: `character-workshop/configs/world.yaml`
- Create: `character-workshop/src/world.py`
- Create: `character-workshop/tests/test_world.py`

- [ ] **Step 1: Write `configs/world.yaml`**

```yaml
world_name: "青汐灵隙"
world_name_en: "Qingxi Rift"
tagline: "现代都市缝隙中，灵力以六系流转，人人可觉醒属于自己的灵域。"
ip_policy: >
  Never use names, places, or distinctive proper nouns from existing
  copyrighted anime/games/films (including but not limited to any
  well-known black-cat fairy franchises). All characters must be original.

affinities:
  - id: form
    name_zh: "形"
    name_en: "Form"
    description: "塑形与物质外形的改写。"
    visual_keywords: ["shapeshift silhouette", "clay-like matter", "shadow morph"]
  - id: mind
    name_zh: "念"
    name_en: "Mind"
    description: "意念、符文与精神干涉。"
    visual_keywords: ["runes", "psychic ripples", "pale blue glow"]
  - id: life
    name_zh: "生"
    name_en: "Life"
    description: "生长、治愈与生机。"
    visual_keywords: ["vines", "healing motes", "emerald light"]
  - id: matter
    name_zh: "质"
    name_en: "Matter"
    description: "密度、金属与重力质感。"
    visual_keywords: ["metal crystal", "gravity rings", "ore sheen"]
  - id: void
    name_zh: "空"
    name_en: "Void"
    description: "空间裂隙与折叠。"
    visual_keywords: ["spatial rift", "folded paper space", "deep purple cracks"]
  - id: time
    name_zh: "时"
    name_en: "Time"
    description: "残影、迟滞与瞬间凝固。"
    visual_keywords: ["afterimages", "amber freeze", "clock pendulum motifs"]

spirit_domain:
  name_zh: "灵域"
  name_en: "Spirit Domain"
  description: "角色内在的能力空间，决定战斗姿态与特效气质。"

style_guide:
  art: "anime character portrait, clean lineart, cel shading, expressive eyes, detailed hair, masterpiece"
  avoid: "photorealistic, NSFW, logo, watermark, text artifacts, extra fingers, blurry"

banned_substrings:
  - "罗小黑"
  - "老君"
  - "无限"
  - "Luo Xiaohei"
```

- [ ] **Step 2: Write failing tests**

```python
# character-workshop/tests/test_world.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.world import load_world, WorldConfig


def test_load_world_has_six_affinities():
    w = load_world()
    assert isinstance(w, WorldConfig)
    assert len(w.affinities) == 6
    assert w.world_name
    names = {a.name_zh for a in w.affinities}
    assert names == {"形", "念", "生", "质", "空", "时"}


def test_affinity_ids_unique():
    w = load_world()
    ids = [a.id for a in w.affinities]
    assert len(ids) == len(set(ids))


def test_banned_check():
    w = load_world()
    assert w.contains_banned("我想要罗小黑同款") is True
    assert w.contains_banned("黑发少年灵师") is False
```

- [ ] **Step 3: Run — expect FAIL**

```bash
pytest tests/test_world.py -v
```

- [ ] **Step 4: Implement `src/world.py`**

```python
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
```

- [ ] **Step 5: Run — expect PASS**

```bash
pytest tests/test_world.py -v
```

- [ ] **Step 6: Commit**

```bash
git add character-workshop/configs/world.yaml character-workshop/src/world.py character-workshop/tests/test_world.py
git commit -m "feat: add thin original world config and loader"
```

---

### Task 4: Character JSON schema (Pydantic)

**Files:**
- Create: `character-workshop/src/models_schema.py`
- Create: `character-workshop/tests/test_models_schema.py`

- [ ] **Step 1: Write failing tests**

```python
# character-workshop/tests/test_models_schema.py
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
        "spirit_domain": "内里是无不断翻折的纸巷。",
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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_models_schema.py -v
```

- [ ] **Step 3: Implement `src/models_schema.py`**

```python
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
        names = world.affinity_zh_names()
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
            f"**主系：** {self.primary_affinity}",
            f"**外形：** {self.appearance}",
            f"**性格：** {self.personality}",
            f"**背景：** {self.backstory}",
            f"**灵域：** {self.spirit_domain}",
            f"**能力展示：** {self.ability_showcase}",
            "",
            "**六系：** " + ", ".join(f"{k} {v}" for k, v in self.affinities.items()),
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
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_models_schema.py -v
```

- [ ] **Step 5: Commit**

```bash
git add character-workshop/src/models_schema.py character-workshop/tests/test_models_schema.py
git commit -m "feat: add CharacterCard pydantic schema and JSON parse"
```

---

### Task 5: Prompt assets + MOCK LLM

**Files:**
- Create: `character-workshop/prompts/system_role.txt`
- Create: `character-workshop/prompts/json_schema.txt`
- Create: `character-workshop/prompts/few_shot.jsonl`
- Create: `character-workshop/prompts/repair_json.txt`
- Create: `character-workshop/src/llm_role.py`
- Create: `character-workshop/tests/test_llm_role.py`

- [ ] **Step 1: Write prompt files**

`prompts/system_role.txt`:

```text
You are the character designer for the original world described below.
Output ONLY a single JSON object matching the schema. No markdown outside JSON unless fenced.
Respect ip_policy strictly. If the user references banned or copyrighted names, rewrite into original equivalents without mentioning the banned names.
Fill all fields. affinities keys must be exactly the six Chinese names provided. Scores 0-100 integers.
image_prompt must be detailed English tags suitable for anime diffusion. motion_prompt is a short English cinematic motion description.
```

`prompts/json_schema.txt`:

```text
Fields: name, one_liner, appearance, personality, backstory, primary_affinity,
affinities (object of six zh names -> 0-100), spirit_domain, ability_showcase,
image_prompt, image_negative, motion_prompt.
```

`prompts/few_shot.jsonl` (one line example — keep valid JSON):

```json
{"user": "沉默的白发女孩，生系，随身带着一盆会动的苔藓", "assistant": {"name": "苔声", "one_liner": "把街角空地养成小小森林的治愈师", "appearance": "白发及肩，青绿外袍，指尖苔痕", "personality": "温柔少言", "backstory": "在青汐湿润裂隙旁长大，苔藓是她的第一个灵伴。", "primary_affinity": "生", "affinities": {"形": 20, "念": 35, "生": 92, "质": 25, "空": 30, "时": 40}, "spirit_domain": "潮湿的微型雨林温室", "ability_showcase": "抬手洒出绿光孢子，藤蔓温柔缠绕", "image_prompt": "anime girl, white hair, green robe, moss accents, gentle expression, portrait", "image_negative": "nsfw, blurry, lowres", "motion_prompt": "soft green particles rising, vines slowly unfurling, gentle camera orbit"}}
```

`prompts/repair_json.txt`:

```text
Your previous output was invalid. Return corrected JSON only, matching the schema and six affinity keys exactly.
```

- [ ] **Step 2: Write failing tests for mock LLM**

```python
# character-workshop/tests/test_llm_role.py
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
```

- [ ] **Step 3: Implement `src/llm_role.py` (mock first; real path stubbed)**

```python
from __future__ import annotations

import json
import os
from pathlib import Path

from src.models_schema import CharacterCard, parse_character_json
from src.paths import PROMPTS
from src.world import WorldConfig, load_world


def _read(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def build_system_prompt(world: WorldConfig | None = None) -> str:
    world = world or load_world()
    parts = [
        _read("system_role.txt"),
        "",
        f"World: {world.world_name} ({world.world_name_en})",
        world.tagline,
        "IP policy:",
        world.ip_policy,
        "",
        "Affinities:",
    ]
    for a in world.affinities:
        parts.append(
            f"- {a.name_zh}/{a.name_en} ({a.id}): {a.description} "
            f"visual={', '.join(a.visual_keywords)}"
        )
    parts.extend(
        [
            "",
            f"Spirit domain name: {world.spirit_domain.name_zh}",
            world.spirit_domain.description,
            "",
            "Schema:",
            _read("json_schema.txt"),
            "",
            f"Art style bias: {world.style_art}",
            f"Avoid: {world.style_avoid}",
        ]
    )
    return "\n".join(parts)


def _mock_card(user_text: str, affinity_pref: str | None) -> CharacterCard:
    world = load_world()
    primary = affinity_pref if affinity_pref in world.affinity_zh_names() else "空"
    aff = {n: 35 for n in world.affinity_zh_names()}
    aff[primary] = 88
    kw = ""
    a = world.affinity_by_zh(primary)
    if a:
        kw = ", ".join(a.visual_keywords)
    data = {
        "name": "试作·灵行者",
        "one_liner": f"回应「{user_text[:24]}」而成形的原创角色",
        "appearance": f"二次元造型，主系气质：{primary}",
        "personality": "冷静而好奇",
        "backstory": f"在{world.world_name}中觉醒。用户设定摘要：{user_text[:120]}",
        "primary_affinity": primary,
        "affinities": aff,
        "spirit_domain": f"与{primary}系共鸣的{world.spirit_domain.name_zh}",
        "ability_showcase": f"施展{primary}系能力，特效：{kw}",
        "image_prompt": (
            f"anime character portrait, {primary} affinity vibe, {kw}, "
            f"{world.style_art}"
        ),
        "image_negative": world.style_avoid,
        "motion_prompt": f"dynamic ability shot, {kw}, cinematic lighting, slow orbit",
    }
    return CharacterCard.model_validate(data)


def generate_character(
    user_text: str,
    affinity_pref: str | None = None,
    mock: bool | None = None,
    base_card: CharacterCard | None = None,
    revise_instruction: str | None = None,
) -> CharacterCard:
    if mock is None:
        mock = os.environ.get("MOCK", "0").strip() in {"1", "true", "True", "yes"}
    user_text = (user_text or "").strip()
    if not user_text and not (base_card and revise_instruction):
        raise ValueError("user_text is required")

    if mock:
        if base_card and revise_instruction:
            # shallow revise in mock
            updated = base_card.model_copy(
                update={
                    "appearance": f"{base_card.appearance}；修订：{revise_instruction}",
                    "image_prompt": f"{base_card.image_prompt}, {revise_instruction}",
                }
            )
            return updated
        return _mock_card(user_text, affinity_pref)

    return _generate_with_transformers(
        user_text=user_text,
        affinity_pref=affinity_pref,
        base_card=base_card,
        revise_instruction=revise_instruction,
    )


_llm_model = None
_llm_tokenizer = None


def _generate_with_transformers(
    user_text: str,
    affinity_pref: str | None,
    base_card: CharacterCard | None,
    revise_instruction: str | None,
) -> CharacterCard:
    """Local HF generate. Loaded lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.device import detect_device, torch_device_string
    from src.paths import ROOT
    import yaml

    global _llm_model, _llm_tokenizer
    cfg_path = ROOT / "configs" / "models.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    model_id = (cfg.get("llm") or {}).get("model_id", "Qwen/Qwen2.5-7B-Instruct")

    device = torch_device_string()
    if _llm_tokenizer is None:
        _llm_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if _llm_model is None:
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            _llm_model.to(device)

    world = load_world()
    system = build_system_prompt(world)
    if base_card and revise_instruction:
        user = (
            "Revise this character JSON per instruction. Return full JSON.\n"
            f"Instruction: {revise_instruction}\n"
            f"Current: {base_card.model_dump_json(ensure_ascii=False)}"
        )
    else:
        pref = f"\nPreferred primary affinity: {affinity_pref}" if affinity_pref else ""
        user = f"Create a character from this brief:{pref}\n{user_text}"

    # optional few-shot
    messages = [{"role": "system", "content": system}]
    fs = PROMPTS / "few_shot.jsonl"
    if fs.exists():
        for line in fs.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            messages.append({"role": "user", "content": ex["user"]})
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(ex["assistant"], ensure_ascii=False),
                }
            )
    messages.append({"role": "user", "content": user})

    text = _llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _llm_tokenizer([text], return_tensors="pt").to(_llm_model.device)
    with torch.no_grad():
        out = _llm_model.generate(
            **inputs,
            max_new_tokens=int((cfg.get("llm") or {}).get("max_new_tokens", 1024)),
            temperature=float((cfg.get("llm") or {}).get("temperature", 0.7)),
            do_sample=True,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    raw = _llm_tokenizer.decode(gen, skip_special_tokens=True)

    try:
        return parse_character_json(raw)
    except Exception:
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _read("repair_json.txt")},
        ]
        text2 = _llm_tokenizer.apply_chat_template(
            repair_messages, tokenize=False, add_generation_prompt=True
        )
        inputs2 = _llm_tokenizer([text2], return_tensors="pt").to(_llm_model.device)
        with torch.no_grad():
            out2 = _llm_model.generate(
                **inputs2,
                max_new_tokens=1024,
                temperature=0.4,
                do_sample=True,
            )
        gen2 = out2[0][inputs2["input_ids"].shape[-1] :]
        raw2 = _llm_tokenizer.decode(gen2, skip_special_tokens=True)
        return parse_character_json(raw2)
```

- [ ] **Step 4: Run mock tests — PASS**

```bash
pytest tests/test_llm_role.py -v
```

- [ ] **Step 5: Commit**

```bash
git add character-workshop/prompts character-workshop/src/llm_role.py character-workshop/tests/test_llm_role.py
git commit -m "feat: add prompt pack and LLM character generator (mock + HF)"
```

---

### Task 6: Card composer (Pillow)

**Files:**
- Create: `character-workshop/src/card_compose.py`
- Create: `character-workshop/tests/test_card_compose.py`

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Implement `src/card_compose.py`**

```python
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.models_schema import CharacterCard
from src.world import load_world


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_radar(affinities: dict[str, int], size: int = 400) -> Image.Image:
    world = load_world()
    labels = world.affinity_zh_names()
    img = Image.new("RGBA", (size, size), (20, 24, 32, 255))
    draw = ImageDraw.Draw(img)
    cx = cy = size / 2
    radius = size * 0.35
    n = len(labels)
    # grid
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = []
        for i in range(n):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            r = radius * ring
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        draw.polygon(pts, outline=(80, 90, 110, 255))
    # values
    val_pts = []
    font = _font(14)
    for i, lab in enumerate(labels):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        score = max(0, min(100, int(affinities.get(lab, 0))))
        r = radius * (score / 100.0)
        val_pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        lx = cx + (radius + 28) * math.cos(ang)
        ly = cy + (radius + 28) * math.sin(ang)
        draw.text((lx - 8, ly - 8), lab, fill=(220, 230, 240, 255), font=font)
    draw.polygon(val_pts, fill=(80, 160, 220, 90), outline=(120, 200, 255, 255))
    return img.convert("RGB")


def compose_card(
    card: CharacterCard,
    portrait: Image.Image,
    out_path: str | Path,
    canvas_size: tuple[int, int] = (1400, 900),
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = canvas_size
    canvas = Image.new("RGB", (w, h), (18, 20, 28))
    draw = ImageDraw.Draw(canvas)

    # portrait left
    pw, ph = 520, 720
    port = portrait.convert("RGB").copy()
    port.thumbnail((pw, ph), Image.Resampling.LANCZOS)
    px, py = 40, (h - port.height) // 2
    canvas.paste(port, (px, py))

    # radar
    radar = draw_radar(card.affinities, size=320)
    rx, ry = 600, 80
    canvas.paste(radar, (rx, ry))

    title_f = _font(36)
    body_f = _font(20)
    small_f = _font(16)
    tx = 600
    ty = 420
    draw.text((tx, 30), card.name, fill=(240, 244, 255), font=title_f)
    draw.text((tx, 80), card.one_liner[:40], fill=(160, 180, 210), font=small_f)

    world = load_world()
    lines = [
        f"主系 · {card.primary_affinity}",
        f"{world.spirit_domain.name_zh} · {card.spirit_domain}",
        f"外形 · {card.appearance}",
        f"性格 · {card.personality}",
        f"背景 · {card.backstory}",
        f"展示 · {card.ability_showcase}",
    ]
    y = ty
    for line in lines:
        # simple wrap
        chunk = line if len(line) < 36 else line[:36] + "…"
        draw.text((tx, y), chunk, fill=(210, 218, 230), font=body_f)
        y += 36

    draw.text(
        (40, h - 36),
        f"{world.world_name} · Local ROCm Character Workshop",
        fill=(100, 110, 130),
        font=small_f,
    )
    canvas.save(out_path, format="PNG")
    return out_path
```

- [ ] **Step 3: Run — PASS**

```bash
pytest tests/test_card_compose.py -v
```

- [ ] **Step 4: Commit**

```bash
git add character-workshop/src/card_compose.py character-workshop/tests/test_card_compose.py
git commit -m "feat: compose character card poster with affinity radar"
```

---

### Task 7: Image gen MOCK + interface

**Files:**
- Create: `character-workshop/configs/models.yaml`
- Create: `character-workshop/src/image_gen.py`
- Create: `character-workshop/tests/test_image_gen_mock.py`

- [ ] **Step 1: Write `configs/models.yaml`**

```yaml
llm:
  model_id: Qwen/Qwen2.5-7B-Instruct
  max_new_tokens: 1024
  temperature: 0.7

image:
  # Prefer a clearly licensed anime SDXL checkpoint; override on cloud after smoke test
  model_id: "cagliostrolab/animagine-xl-3.1"
  width: 832
  height: 1216
  steps: 30
  guidance: 6.0
  ref_scale: 0.65
  dtype: fp16
  enable_attention_slicing: true
  enable_vae_tiling: true
  lora_path: null
  lora_scale: 0.8

video:
  enabled: true
  # Set after ROCm smoke test — placeholder id
  model_id: "THUDM/CogVideoX-2b"
  backend: auto   # auto | cogvideox | animatediff | mock
  num_frames: 49
  fps: 8
  timeout_sec: 240
```

- [ ] **Step 2: Tests**

```python
# character-workshop/tests/test_image_gen_mock.py
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image_gen import generate_portrait
from src.llm_role import generate_character


def test_mock_portrait(tmp_path):
    card = generate_character("红发质系武士", affinity_pref="质", mock=True)
    out = tmp_path / "p.png"
    path = generate_portrait(card, out_path=out, ref_image=None, mock=True)
    assert path.exists()
    im = Image.open(path)
    assert im.size[0] > 64
```

- [ ] **Step 3: Implement `src/image_gen.py`**

```python
from __future__ import annotations

import os
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from src.device import detect_device, torch_device_string, vram_profile
from src.models_schema import CharacterCard
from src.paths import ROOT


def _image_cfg() -> dict:
    p = ROOT / "configs" / "models.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data.get("image") or {}


def _mock_portrait(card: CharacterCard, out_path: Path) -> Path:
    cfg = _image_cfg()
    w, h = int(cfg.get("width", 832)), int(cfg.get("height", 1216))
    img = Image.new("RGB", (w, h), (32, 40, 64))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, w - 40, h - 40], outline=(120, 180, 255), width=4)
    d.text((60, 80), card.name[:20], fill=(240, 240, 255))
    d.text((60, 140), card.primary_affinity, fill=(180, 220, 255))
    d.text((60, 200), "MOCK PORTRAIT", fill=(200, 200, 120))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


_pipe = None


def generate_portrait(
    card: CharacterCard,
    out_path: str | Path,
    ref_image: Image.Image | None = None,
    mock: bool | None = None,
    ref_scale: float | None = None,
) -> Path:
    if mock is None:
        mock = os.environ.get("MOCK", "0").strip() in {"1", "true", "True", "yes"}
    out_path = Path(out_path)
    if mock:
        return _mock_portrait(card, out_path)

    return _generate_diffusers(card, out_path, ref_image, ref_scale)


def _generate_diffusers(
    card: CharacterCard,
    out_path: Path,
    ref_image: Image.Image | None,
    ref_scale: float | None,
) -> Path:
    import torch
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

    global _pipe
    cfg = _image_cfg()
    model_id = cfg["model_id"]
    device = torch_device_string()
    info = detect_device()
    profile = vram_profile(info)

    width = int(cfg.get("width", 832))
    height = int(cfg.get("height", 1216))
    steps = int(cfg.get("steps", 30))
    if profile == "16gb":
        width, height = min(width, 768), min(height, 1024)
        steps = min(steps, 28)
    if profile == "low":
        width, height = 512, 768
        steps = min(steps, 24)

    dtype = torch.float16 if device == "cuda" else torch.float32
    if _pipe is None:
        _pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )
        if cfg.get("enable_attention_slicing", True):
            _pipe.enable_attention_slicing()
        if cfg.get("enable_vae_tiling", True) and hasattr(_pipe, "enable_vae_tiling"):
            _pipe.enable_vae_tiling()
        _pipe = _pipe.to(device)

    prompt = card.image_prompt
    negative = card.image_negative or ""
    guidance = float(cfg.get("guidance", 6.0))
    scale = float(ref_scale if ref_scale is not None else cfg.get("ref_scale", 0.65))

    generator = torch.Generator(device=device).manual_seed(42)

    if ref_image is not None:
        # img2img path (IP-Adapter can replace later)
        img2img = StableDiffusionXLImg2ImgPipeline(
            vae=_pipe.vae,
            text_encoder=_pipe.text_encoder,
            text_encoder_2=_pipe.text_encoder_2,
            tokenizer=_pipe.tokenizer,
            tokenizer_2=_pipe.tokenizer_2,
            unet=_pipe.unet,
            scheduler=_pipe.scheduler,
        )
        img2img = img2img.to(device)
        init = ref_image.convert("RGB").resize((width, height))
        result = img2img(
            prompt=prompt,
            negative_prompt=negative,
            image=init,
            strength=min(0.95, max(0.2, 1.0 - scale * 0.5)),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        image = result.images[0]
    else:
        result = _pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        image = result.images[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def unload_image_models() -> None:
    global _pipe
    _pipe = None
    try:
        import torch
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
```

- [ ] **Step 4: Run — PASS**

```bash
pytest tests/test_image_gen_mock.py -v
```

- [ ] **Step 5: Commit**

```bash
git add character-workshop/configs/models.yaml character-workshop/src/image_gen.py character-workshop/tests/test_image_gen_mock.py
git commit -m "feat: add portrait generation with mock and Diffusers path"
```

---

### Task 8: Video gen MOCK + soft interface

**Files:**
- Create: `character-workshop/src/video_gen.py`
- Create: `character-workshop/tests/test_video_gen_mock.py`

- [ ] **Step 1: Test**

```python
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
```

- [ ] **Step 2: Implement**

```python
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_video_gen_mock.py -v
```

- [ ] **Step 4: Commit**

```bash
git add character-workshop/src/video_gen.py character-workshop/tests/test_video_gen_mock.py
git commit -m "feat: add ability video module with mock and I2V hooks"
```

---

### Task 9: Orchestrator end-to-end (MOCK)

**Files:**
- Create: `character-workshop/src/orchestrator.py`
- Create: `character-workshop/tests/test_orchestrator.py`

- [ ] **Step 1: Tests**

```python
# character-workshop/tests/test_orchestrator.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator import JobState, generate_card_job, revise_job, animate_job
from src.paths import OUTPUTS


def test_generate_card_job_mock():
    result = generate_card_job(
        user_text="青衣空系旅人，背着灯笼",
        affinity_pref="空",
        ref_image=None,
        mock=True,
    )
    assert result.state == JobState.READY
    assert result.job_dir.exists()
    assert (result.job_dir / "character.json").exists()
    assert (result.job_dir / "portrait.png").exists()
    assert (result.job_dir / "card.png").exists()
    assert result.card is not None
    assert result.error is None


def test_revise_and_animate_mock():
    base = generate_card_job("黑发念系少年", affinity_pref="念", mock=True)
    rev = revise_job(base.job_id, "改成金色短发", mock=True)
    assert rev.state == JobState.READY
    assert "金" in rev.card.appearance or "金" in rev.card.image_prompt
    anim = animate_job(rev.job_id, mock=True)
    assert anim.state in {JobState.DONE, JobState.READY}
    # video path or soft fail both ok in mock if file exists
    assert anim.error is None or anim.video_path is None
```

- [ ] **Step 2: Implement `src/orchestrator.py`**

```python
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from PIL import Image

from src.card_compose import compose_card
from src.device import detect_device
from src.image_gen import generate_portrait, unload_image_models
from src.llm_role import generate_character
from src.models_schema import CharacterCard
from src.paths import OUTPUTS, ensure_runtime_dirs
from src.video_gen import generate_ability_video


class JobState(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    IMAGING = "imaging"
    COMPOSING = "composing"
    READY = "ready"
    ANIMATING = "animating"
    DONE = "done"
    FAILED = "failed"


ProgressCb = Callable[[JobState, str], None]


@dataclass
class JobResult:
    job_id: str
    state: JobState
    job_dir: Path
    card: CharacterCard | None = None
    card_path: Path | None = None
    portrait_path: Path | None = None
    video_path: Path | None = None
    lore_md: str = ""
    error: str | None = None
    failed_step: str | None = None
    meta: dict = field(default_factory=dict)


def _job_dir(job_id: str) -> Path:
    ensure_runtime_dirs()
    d = OUTPUTS / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _emit(cb: ProgressCb | None, state: JobState, msg: str) -> None:
    if cb:
        cb(state, msg)


def generate_card_job(
    user_text: str,
    affinity_pref: str | None = None,
    ref_image: Image.Image | None = None,
    mock: bool | None = None,
    progress_cb: ProgressCb | None = None,
    job_id: str | None = None,
) -> JobResult:
    t0 = time.time()
    job_id = job_id or uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    (job_dir / "input.txt").write_text(user_text or "", encoding="utf-8")
    if ref_image is not None:
        ref_image.convert("RGB").save(job_dir / "ref.png")

    meta = {
        "device": detect_device().__dict__,
        "timings": {},
    }

    try:
        _emit(progress_cb, JobState.PLANNING, "结构化角色设定…")
        t1 = time.time()
        card = generate_character(
            user_text=user_text,
            affinity_pref=affinity_pref or None,
            mock=mock,
        )
        meta["timings"]["llm_sec"] = round(time.time() - t1, 3)
        (job_dir / "character.json").write_text(
            card.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
        )

        _emit(progress_cb, JobState.IMAGING, "绘制立绘…")
        t2 = time.time()
        portrait_path = job_dir / "portrait.png"
        ref = None
        ref_file = job_dir / "ref.png"
        if ref_file.exists():
            ref = Image.open(ref_file)
        generate_portrait(card, portrait_path, ref_image=ref, mock=mock)
        meta["timings"]["image_sec"] = round(time.time() - t2, 3)

        _emit(progress_cb, JobState.COMPOSING, "合成角色卡…")
        t3 = time.time()
        card_path = job_dir / "card.png"
        compose_card(card, Image.open(portrait_path), card_path)
        meta["timings"]["compose_sec"] = round(time.time() - t3, 3)
        meta["timings"]["total_sec"] = round(time.time() - t0, 3)
        (job_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        _emit(progress_cb, JobState.READY, "角色卡已就绪")
        return JobResult(
            job_id=job_id,
            state=JobState.READY,
            job_dir=job_dir,
            card=card,
            card_path=card_path,
            portrait_path=portrait_path,
            lore_md=card.lore_markdown(),
            meta=meta,
        )
    except Exception as e:
        _emit(progress_cb, JobState.FAILED, str(e))
        return JobResult(
            job_id=job_id,
            state=JobState.FAILED,
            job_dir=job_dir,
            error=str(e),
            failed_step="generate_card",
            meta=meta,
        )


def revise_job(
    job_id: str,
    instruction: str,
    mock: bool | None = None,
    progress_cb: ProgressCb | None = None,
) -> JobResult:
    job_dir = _job_dir(job_id)
    raw = (job_dir / "character.json").read_text(encoding="utf-8")
    base = CharacterCard.model_validate_json(raw)
    try:
        _emit(progress_cb, JobState.PLANNING, "根据修订更新设定…")
        card = generate_character(
            user_text=base.one_liner,
            mock=mock,
            base_card=base,
            revise_instruction=instruction,
        )
        (job_dir / "character.json").write_text(
            card.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _emit(progress_cb, JobState.IMAGING, "重绘立绘…")
        portrait_path = job_dir / "portrait.png"
        ref = Image.open(job_dir / "ref.png") if (job_dir / "ref.png").exists() else None
        generate_portrait(card, portrait_path, ref_image=ref, mock=mock)
        _emit(progress_cb, JobState.COMPOSING, "重新合成…")
        card_path = job_dir / "card.png"
        compose_card(card, Image.open(portrait_path), card_path)
        _emit(progress_cb, JobState.READY, "修订完成")
        return JobResult(
            job_id=job_id,
            state=JobState.READY,
            job_dir=job_dir,
            card=card,
            card_path=card_path,
            portrait_path=portrait_path,
            lore_md=card.lore_markdown(),
        )
    except Exception as e:
        return JobResult(
            job_id=job_id,
            state=JobState.FAILED,
            job_dir=job_dir,
            error=str(e),
            failed_step="revise",
        )


def animate_job(
    job_id: str,
    mock: bool | None = None,
    progress_cb: ProgressCb | None = None,
) -> JobResult:
    job_dir = _job_dir(job_id)
    portrait = job_dir / "portrait.png"
    card = CharacterCard.model_validate_json(
        (job_dir / "character.json").read_text(encoding="utf-8")
    )
    card_path = job_dir / "card.png"
    try:
        _emit(progress_cb, JobState.ANIMATING, "生成能力动画…")
        unload_image_models()
        video_path = job_dir / "ability.mp4"
        path, err = generate_ability_video(
            portrait_path=portrait,
            motion_prompt=card.motion_prompt,
            out_path=video_path,
            mock=mock,
        )
        if err:
            _emit(progress_cb, JobState.READY, f"动画失败（角色卡仍可用）：{err}")
            return JobResult(
                job_id=job_id,
                state=JobState.READY,
                job_dir=job_dir,
                card=card,
                card_path=card_path if card_path.exists() else None,
                portrait_path=portrait,
                lore_md=card.lore_markdown(),
                error=err,
                failed_step="animate",
            )
        _emit(progress_cb, JobState.DONE, "动画完成")
        return JobResult(
            job_id=job_id,
            state=JobState.DONE,
            job_dir=job_dir,
            card=card,
            card_path=card_path if card_path.exists() else None,
            portrait_path=portrait,
            video_path=path,
            lore_md=card.lore_markdown(),
        )
    except Exception as e:
        return JobResult(
            job_id=job_id,
            state=JobState.READY,
            job_dir=job_dir,
            card=card,
            card_path=card_path if card_path.exists() else None,
            portrait_path=portrait,
            lore_md=card.lore_markdown(),
            error=str(e),
            failed_step="animate",
        )
```

- [ ] **Step 3: Run**

```bash
pytest tests/test_orchestrator.py -v
```

- [ ] **Step 4: Commit**

```bash
git add character-workshop/src/orchestrator.py character-workshop/tests/test_orchestrator.py
git commit -m "feat: add job orchestrator for card/revise/animate paths"
```

---

### Task 10: Gradio workbench UI

**Files:**
- Create: `character-workshop/app.py`

- [ ] **Step 1: Implement `app.py`**

```python
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.device import detect_device, vram_profile
from src.orchestrator import JobState, animate_job, generate_card_job, revise_job
from src.paths import ensure_runtime_dirs
from src.world import load_world

ensure_runtime_dirs()
WORLD = load_world()
DEVICE = detect_device()
AFFINITY_CHOICES = ["自动"] + WORLD.affinity_zh_names()


CSS = """
.gradio-container {max-width: 1200px !important;}
footer {display: none !important;}
#title {font-size: 1.6rem; font-weight: 700;}
#tagline {opacity: 0.8;}
#badge {color: #7dd3fc;}
"""


def _pref(affinity: str) -> str | None:
    if not affinity or affinity == "自动":
        return None
    return affinity


def ui_generate(brief, ref, affinity, progress=gr.Progress(track_tqdm=False)):
    if not (brief or "").strip():
        raise gr.Error("请填写角色设定描述")

    states = []

    def cb(state: JobState, msg: str):
        states.append(f"{state.value}: {msg}")
        progress(len(states) / 5.0, desc=msg)

    result = generate_card_job(
        user_text=brief,
        affinity_pref=_pref(affinity),
        ref_image=ref,
        mock=None,
        progress_cb=cb,
    )
    if result.state == JobState.FAILED:
        raise gr.Error(result.error or "生成失败")

    status = " → ".join(states) if states else result.state.value
    video = str(result.video_path) if result.video_path else None
    return (
        result.job_id,
        str(result.card_path) if result.card_path else None,
        result.lore_md,
        status,
        video,
        None,
    )


def ui_revise(job_id, instruction, progress=gr.Progress(track_tqdm=False)):
    if not job_id:
        raise gr.Error("请先生成角色卡")
    if not (instruction or "").strip():
        raise gr.Error("请填写修订说明")

    def cb(state: JobState, msg: str):
        progress(0.5, desc=msg)

    result = revise_job(job_id, instruction, mock=None, progress_cb=cb)
    if result.state == JobState.FAILED:
        raise gr.Error(result.error or "修订失败")
    return (
        str(result.card_path) if result.card_path else None,
        result.lore_md,
        f"revised: {result.state.value}",
        None,
    )


def ui_animate(job_id, progress=gr.Progress(track_tqdm=False)):
    if not job_id:
        raise gr.Error("请先生成角色卡")

    def cb(state: JobState, msg: str):
        progress(0.5, desc=msg)

    result = animate_job(job_id, mock=None, progress_cb=cb)
    err = result.error
    video = str(result.video_path) if result.video_path else None
    status = "动画完成" if video and not err else f"动画未成功（角色卡仍可用）：{err or 'unknown'}"
    return video, status


def build_app() -> gr.Blocks:
    profile = vram_profile(DEVICE)
    with gr.Blocks(css=CSS, title=f"{WORLD.world_name} · Character Workshop") as demo:
        gr.Markdown(
            f"<div id='title'>{WORLD.world_name} · 角色工坊</div>"
            f"<div id='tagline'>{WORLD.tagline}</div>"
            f"<div id='badge'>Local · ROCm/AMD · device={DEVICE.name} · profile={profile} · mock={DEVICE.mock_mode}</div>"
        )
        job_id = gr.State("")

        with gr.Row():
            with gr.Column(scale=4):
                brief = gr.Textbox(
                    label="描述角色",
                    lines=8,
                    placeholder="外形、性格、能力倾向…（原创设定，勿使用第三方 IP 专名）",
                )
                ref = gr.Image(label="参考图（可选）", type="pil")
                affinity = gr.Dropdown(
                    label="六系偏好", choices=AFFINITY_CHOICES, value="自动"
                )
                btn = gr.Button("生成角色卡", variant="primary")
                revise_txt = gr.Textbox(label="在结果上改一版", placeholder="例如：改成金色短发")
                btn_rev = gr.Button("应用修订")
            with gr.Column(scale=6):
                card_img = gr.Image(label="角色卡海报", type="filepath")
                lore = gr.Markdown(label="设定")
                status = gr.Textbox(label="进度", interactive=False)
                btn_vid = gr.Button("生成能力动画")
                video = gr.Video(label="能力动画")
                vid_status = gr.Textbox(label="动画状态", interactive=False)

        btn.click(
            ui_generate,
            inputs=[brief, ref, affinity],
            outputs=[job_id, card_img, lore, status, video, vid_status],
        )
        btn_rev.click(
            ui_revise,
            inputs=[job_id, revise_txt],
            outputs=[card_img, lore, status, video],
        )
        btn_vid.click(ui_animate, inputs=[job_id], outputs=[video, vid_status])

        gr.Markdown(
            "Powered by **AMD Radeon + ROCm** local inference · "
            f"World: {WORLD.world_name_en} · Track 1 Multimodal"
        )
    return demo


if __name__ == "__main__":
    ensure_runtime_dirs()
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "7860")))
```

- [ ] **Step 2: Manual smoke (MOCK)**

```bash
cd character-workshop
set MOCK=1
pip install gradio pydantic pyyaml Pillow numpy -q
python app.py
```

Open `http://127.0.0.1:7860` — generate card with any Chinese brief; expect poster + lore.

- [ ] **Step 3: Commit**

```bash
git add character-workshop/app.py
git commit -m "feat: add Gradio character workbench UI"
```

---

### Task 11: Eval cases + JSON regression script

**Files:**
- Create: `character-workshop/data/eval/cases.jsonl`
- Create: `character-workshop/scripts/run_eval_json.py`

- [ ] **Step 1: Write ≥20 cases** (sample first 5; expand to 20+ in same format)

```json
{"id": "case01", "user_text": "青衣空系旅人，背着纸灯笼", "has_ref": false, "expect_affinity": "空", "notes": "demo"}
{"id": "case02", "user_text": "白发生系治愈师", "has_ref": false, "expect_affinity": "生", "notes": "demo"}
{"id": "case03", "user_text": "红 spar 质系重装", "has_ref": false, "expect_affinity": "质", "notes": "fix typo ok"}
```

Fix case03 text to proper Chinese when writing file. Include short prompts, locked affinity, and one banned-name attempt case.

- [ ] **Step 2: `scripts/run_eval_json.py`**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_role import generate_character
from src.world import load_world


def main() -> int:
    world = load_world()
    path = ROOT / "data" / "eval" / "cases.jsonl"
    ok = fail = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        try:
            card = generate_character(
                case["user_text"],
                affinity_pref=case.get("expect_affinity"),
                mock=True,
            )
            assert card.primary_affinity in world.affinity_zh_names()
            assert card.image_prompt
            ok += 1
            print("OK", case["id"], card.name, card.primary_affinity)
        except Exception as e:
            fail += 1
            print("FAIL", case["id"], e)
    print(f"summary ok={ok} fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run**

```bash
python scripts/run_eval_json.py
```

Expected: `summary ok=N fail=0`

- [ ] **Step 4: Commit**

```bash
git add character-workshop/data/eval character-workshop/scripts/run_eval_json.py
git commit -m "test: add eval cases and mock JSON regression script"
```

---

### Task 12: README + setup scripts + root docs link

**Files:**
- Create: `character-workshop/README.md`
- Create: `character-workshop/scripts/setup_rocm.sh`
- Create: `character-workshop/scripts/download_models.sh`

- [ ] **Step 1: Write README** covering: Overview, Features, Hardware/ROCm, Radeon Cloud + PVC, model download, `MOCK=1` local UI, `MOCK=0` real run, `rc-tunnel`, structure, limitations, licenses, Track 1 submission notes.

Minimum sections (English for PR):

```markdown
# Qingxi Rift Character Workshop

Track 1 multimodal tool for AMD AI DevMaster Hackathon.

## Quick start (mock)

```bash
cd character-workshop
python -m venv .venv
# activate venv
pip install -r requirements-dev.txt
set MOCK=1   # export MOCK=1 on Linux
python app.py
```

## Real inference (Radeon Cloud + ROCm)

1. Launch template with PyTorch ROCm image, Persistent PVC.
2. Install deps matching ROCm torch wheels from AMD docs.
3. `bash scripts/download_models.sh`
4. `export MOCK=0`
5. `python app.py`
6. Optional: `rc-tunnel expose --port 7860`

## Tests

```bash
pytest -q
python scripts/run_eval_json.py
```
```

- [ ] **Step 2: `scripts/download_models.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
from huggingface_hub import snapshot_download
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/models.yaml").read_text())
# downloads to HF cache; optional local_dir under models/
print("llm", cfg["llm"]["model_id"])
snapshot_download(cfg["llm"]["model_id"])
print("image", cfg["image"]["model_id"])
snapshot_download(cfg["image"]["model_id"])
print("done")
PY
```

- [ ] **Step 3: Commit**

```bash
git add character-workshop/README.md character-workshop/scripts
git commit -m "docs: add README and model setup scripts"
```

---

### Task 13: Radeon Cloud real-model bring-up (manual checklist)

No new code required until smoke tests fail — then patch `image_gen.py` / `video_gen.py` / `models.yaml`.

- [ ] **Step 1:** Create Radeon Cloud template (ROCm PyTorch), PVC on, optional SSH.  
- [ ] **Step 2:** `git clone` / upload project; install torch ROCm + requirements.  
- [ ] **Step 3:** `python -c "from src.device import detect_device; print(detect_device())"` — expect `cuda` + `rocm_hint=True`.  
- [ ] **Step 4:** Download LLM; run one `generate_character(..., mock=False)` in Python REPL; fix JSON issues.  
- [ ] **Step 5:** Download SDXL anime model; one `generate_portrait` real; if OOM, rely on profile downscale.  
- [ ] **Step 6:** Full `generate_card_job(..., mock=False)` once.  
- [ ] **Step 7:** Smoke I2V; implement `_cogvideox` or AnimateDiff; else set `video.enabled: false` and document.  
- [ ] **Step 8:** Record 3–5 min demo with `rocm-smi` + UI.  
- [ ] **Step 9:** Destroy instance when idle.

**Implement CogVideoX body only after Step 7 succeeds on cloud** — replace `NotImplementedError` with working Diffusers I2V call matching installed package API.

---

### Task 14: Submission package (final days)

- [ ] **Step 1:** Write English PDF (architecture diagram, models, ROCm adaptation, screenshots).  
- [ ] **Step 2:** PPT or poster.  
- [ ] **Step 3:** Demo video file.  
- [ ] **Step 4:** Fork `AMD-DEV-CONTEST/Radeon-hackathon-2026-07`, PR title `Track 1, <Team>, Qingxi Character Workshop`.  
- [ ] **Step 5:** Ensure README reproduction path works on a clean cloud notebook.

---

## Spec coverage checklist

| Spec requirement | Task(s) |
|------------------|---------|
| Thin original world | T3 |
| Character JSON contract | T4–T5 |
| Local LLM path + mock | T5 |
| Portrait txt2img / img2img ref | T7 |
| Card + radar compose | T6 |
| Orchestrator A/B/C paths | T9 |
| Video soft-fail | T8–T9 |
| Gradio workbench | T10 |
| Device / VRAM profiles | T2, T7 |
| Eval cases | T11 |
| Hybrid cloud workflow docs | T12–T13 |
| Submission artifacts | T14 |
| No multi-session / no IP training | T3 banned list, T10 copy, non-goals |

## Placeholder / consistency self-review

- Types aligned: `CharacterCard`, `JobState`, `JobResult`, `generate_card_job` / `revise_job` / `animate_job`.  
- MOCK via env `MOCK=1` consistently.  
- Video real backends intentionally stubbed until cloud smoke (T13) — not a design gap; explicit gate.  
- Checkpoint IDs in `models.yaml` are starting points; T13 may swap after license/ROCm check.

---

## Execution handoff

Plan complete and saved to:

`docs/superpowers/plans/2026-07-26-character-workshop-implementation.md`

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — this session with executing-plans, batched with checkpoints  

**Which approach?**
