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
        _emit(progress_cb, JobState.PLANNING, "Planning character...")
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

        _emit(progress_cb, JobState.IMAGING, "Painting portrait...")
        t2 = time.time()
        portrait_path = job_dir / "portrait.png"
        ref = None
        ref_file = job_dir / "ref.png"
        if ref_file.exists():
            ref = Image.open(ref_file)
        generate_portrait(card, portrait_path, ref_image=ref, mock=mock)
        meta["timings"]["image_sec"] = round(time.time() - t2, 3)

        _emit(progress_cb, JobState.COMPOSING, "Composing card...")
        t3 = time.time()
        card_path = job_dir / "card.png"
        compose_card(card, Image.open(portrait_path), card_path)
        meta["timings"]["compose_sec"] = round(time.time() - t3, 3)
        meta["timings"]["total_sec"] = round(time.time() - t0, 3)
        (job_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        _emit(progress_cb, JobState.READY, "Card ready")
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
        _emit(progress_cb, JobState.PLANNING, "Applying revision...")
        card = generate_character(
            user_text=base.one_liner,
            mock=mock,
            base_card=base,
            revise_instruction=instruction,
        )
        (job_dir / "character.json").write_text(
            card.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _emit(progress_cb, JobState.IMAGING, "Re-painting portrait...")
        portrait_path = job_dir / "portrait.png"
        ref = Image.open(job_dir / "ref.png") if (job_dir / "ref.png").exists() else None
        generate_portrait(card, portrait_path, ref_image=ref, mock=mock)
        _emit(progress_cb, JobState.COMPOSING, "Re-composing card...")
        card_path = job_dir / "card.png"
        compose_card(card, Image.open(portrait_path), card_path)
        _emit(progress_cb, JobState.READY, "Revision complete")
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
        _emit(progress_cb, JobState.ANIMATING, "Generating ability animation...")
        unload_image_models()
        video_path = job_dir / "ability.mp4"
        path, err = generate_ability_video(
            portrait_path=portrait,
            motion_prompt=card.motion_prompt,
            out_path=video_path,
            mock=mock,
        )
        if err:
            _emit(progress_cb, JobState.READY, f"Animation failed (card still usable): {err}")
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
        _emit(progress_cb, JobState.DONE, "Animation complete")
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
