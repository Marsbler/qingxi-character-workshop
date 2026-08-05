from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from src.models_schema import CharacterCard, parse_character_json
from src.paths import OUTPUTS, PROMPTS, ROOT
from src.world import WorldConfig, load_world


def _read(name: str) -> str:
    p = PROMPTS / name
    return p.read_text(encoding="utf-8")


def build_system_prompt(world: WorldConfig | None = None) -> str:
    world = world or load_world()
    role = _read("system_role.txt")
    schema = _read("json_schema.txt")
    few_shot = _read("few_shot.jsonl").strip()
    parts = [
        role,
        "",
        f"# World: {world.world_name} ({world.world_name_en})",
        f"tagline: {world.tagline}",
        f"ip_policy: {world.ip_policy}",
        "",
        "Affinities (zh -> en): "
        + ", ".join(f"{a.name_zh}/{a.name_en}" for a in world.affinities),
        "spirit_domain: "
        + f"{world.spirit_domain.name_zh}/{world.spirit_domain.name_en} - "
        + world.spirit_domain.description,
        "",
        f"style_art: {world.style_art}",
        f"style_avoid: {world.style_avoid}",
        "",
        "# JSON schema",
        schema,
        "",
        "# Few-shot examples (one JSON per line)",
        few_shot,
    ]
    return "\n".join(parts)


def _mock_card(
    user_text: str,
    affinity_pref: str | None = None,
    base_card: CharacterCard | None = None,
    revise_instruction: str | None = None,
) -> CharacterCard:
    world = load_world()
    names = world.affinity_zh_names()
    pref = affinity_pref or _guess_affinity(user_text, names) or names[0]
    if pref not in names:
        raise ValueError(f"affinity_pref must be one of {names}, got {pref!r}")

    aff = {n: 30 for n in names}
    aff[pref] = 90
    # spread a couple of secondary scores
    for n in names:
        if n != pref:
            aff[n] = 25 + (hash(n + pref) % 20)

    if base_card is not None and revise_instruction:
        # revise path: copy and tweak appearance + image_prompt
        instr = revise_instruction
        new_appearance = f"{base_card.appearance}（修订：{instr}）"
        new_image = f"{base_card.image_prompt}, revised: {instr}"
        return base_card.model_copy(
            update={"appearance": new_appearance, "image_prompt": new_image}
        )

    # fresh mock card
    name = _mock_name(user_text)
    one_liner = f"由「{user_text}」启发的原创角色"
    appearance = "原创角色外形，简洁动漫风格"
    personality = "性格待展开"
    backstory = f"在{world.world_name}的街角长大，与{pref}系灵力相伴。"
    spirit_domain = f"以{pref}系为主的内在灵域"
    ability_showcase = f"释放{pref}系灵力，展现原创招式"
    image_prompt = (
        f"anime character, {pref} affinity theme, "
        f"{world.style_art}"
    )
    image_negative = world.style_avoid
    motion_prompt = f"camera slow orbit, {pref} affinity particles drifting"

    data = {
        "name": name,
        "one_liner": one_liner,
        "appearance": appearance,
        "personality": personality,
        "backstory": backstory,
        "primary_affinity": pref,
        "affinities": aff,
        "spirit_domain": spirit_domain,
        "ability_showcase": ability_showcase,
        "image_prompt": image_prompt,
        "image_negative": image_negative,
        "motion_prompt": motion_prompt,
    }
    return CharacterCard.model_validate(data)


def _guess_affinity(text: str, names: list[str]) -> str | None:
    for n in names:
        if n in text:
            return n
    return None


def _mock_name(text: str) -> str:
    # derive a short original zh name from user text
    cleaned = re.sub(r"[，,。.！!？? \s]+", "", text)
    base = cleaned[:2] if cleaned else "无名"
    return f"{base}灵"


def generate_character(
    user_text: str,
    affinity_pref: str | None = None,
    mock: bool | None = None,
    base_card: CharacterCard | None = None,
    revise_instruction: str | None = None,
) -> CharacterCard:
    world = load_world()

    # decide mock vs real
    if mock is None:
        # default to mock unless transformers is importable
        try:
            import transformers  # noqa: F401
            mock = False
        except Exception:
            mock = True

    if mock:
        return _mock_card(
            user_text=user_text,
            affinity_pref=affinity_pref,
            base_card=base_card,
            revise_instruction=revise_instruction,
        )

    return _generate_with_transformers(
        user_text=user_text,
        affinity_pref=affinity_pref,
        world=world,
        base_card=base_card,
        revise_instruction=revise_instruction,
    )


_llm_model = None
_llm_tokenizer = None


def _llm_cfg() -> dict:
    p = ROOT / "configs" / "models.yaml"
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data.get("llm") or {}
    return {}


def _load_llm():
    """Load and cache tokenizer+model once per process."""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None and _llm_tokenizer is not None:
        return _llm_tokenizer, _llm_model

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    from src.device import torch_device_string

    cfg = _llm_cfg()
    model_name = cfg.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
    device = torch_device_string()
    _llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    _llm_model.eval()
    return _llm_tokenizer, _llm_model


def unload_llm() -> None:
    """Free LLM weights (call before loading big diffusion models on tight VRAM)."""
    global _llm_model, _llm_tokenizer
    _llm_model = None
    _llm_tokenizer = None
    try:
        import gc

        import torch  # type: ignore

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _dump_debug(raw: str, tag: str) -> Path:
    d = OUTPUTS / "_llm_debug"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}.txt"
    p.write_text(raw, encoding="utf-8")
    return p


def _chat_generate(messages: list[dict], max_new_tokens: int, greedy: bool) -> str:
    tokenizer, model = _load_llm()
    import torch  # type: ignore

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=0.7 if not greedy else None,
            top_p=0.9 if not greedy else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def _few_shot_messages() -> list[dict]:
    msgs: list[dict] = []
    fs = PROMPTS / "few_shot.jsonl"
    if fs.exists():
        for line in fs.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs.append({"role": "user", "content": ex["user"]})
            msgs.append(
                {
                    "role": "assistant",
                    "content": json.dumps(ex["assistant"], ensure_ascii=False),
                }
            )
    return msgs


def _generate_with_transformers(
    user_text: str,
    affinity_pref: str | None,
    world: WorldConfig,
    base_card: CharacterCard | None = None,
    revise_instruction: str | None = None,
) -> CharacterCard:
    """Lazy HF text-generation path. Model cached; JSON-first greedy decoding."""
    cfg = _llm_cfg()
    max_new_tokens = int(cfg.get("max_new_tokens", 1024))

    system_prompt = build_system_prompt(world)
    repair = _read("repair_json.txt")

    if affinity_pref and affinity_pref not in world.affinity_zh_names():
        raise ValueError(
            f"affinity_pref must be one of {world.affinity_zh_names()}, "
            f"got {affinity_pref!r}"
        )

    user_content = user_text
    if affinity_pref:
        user_content += f"\n主系偏好：{affinity_pref}"
    if base_card is not None and revise_instruction:
        user_content += (
            f"\n原角色：{base_card.model_dump_json()}\n"
            f"修订要求：{revise_instruction}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        *_few_shot_messages(),
        {"role": "user", "content": user_content},
    ]

    # Attempt 1: greedy — most reliable for strict JSON output
    raw = _chat_generate(messages, max_new_tokens=max_new_tokens, greedy=True)
    try:
        return parse_character_json(raw)
    except Exception as e1:
        _dump_debug(raw, "attempt1")

    # Attempt 2: repair with greedy decode
    repair_messages = messages + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": repair},
    ]
    raw2 = _chat_generate(repair_messages, max_new_tokens=max_new_tokens, greedy=True)
    try:
        return parse_character_json(raw2)
    except Exception as e2:
        p2 = _dump_debug(raw2, "attempt2")
        excerpt = (raw2 or "")[:300].replace("\n", " ")
        raise ValueError(
            f"LLM did not produce valid JSON after repair. "
            f"First error: {e1}; second: {e2}. "
            f"Raw dumped to {p2}. Excerpt: {excerpt!r}"
        )