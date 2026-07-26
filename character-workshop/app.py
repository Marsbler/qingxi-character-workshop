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
