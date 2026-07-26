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
