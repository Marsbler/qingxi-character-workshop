from __future__ import annotations

import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.models_schema import CharacterCard
from src.world import load_world

# Module-level cache for the resolved font PATH (not the font object), so
# repeated `_font(size)` calls skip the discovery search.
_CACHED_FONT_PATH: str | None = None

# Known candidate font paths (Windows, macOS, Linux).
_KNOWN_FONT_PATHS: list[str] = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]

# Substrings (case-insensitive) used to match CJK-capable font filenames when
# globbing the filesystem.
_CJK_FONT_HINTS: tuple[str, ...] = (
    "cjk",
    "wqy",
    "zenhei",
    "microhei",
    "uming",
    "ukai",
    "droidsansfallback",
    "notosanssc",
    "notosanstc",
    "sourcehan",
    "simhei",
    "msyh",
    "pingfang",
)

# Roots to recursively search for CJK fonts when no known path exists.
_GLOB_ROOTS: list[str] = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / ".local" / "share" / "fonts"),
]


def _try_load(path: str, size: int) -> ImageFont.ImageFont | None:
    """Try to load a truetype font; return None on OSError."""
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return None


def _glob_find_cjk_font() -> str | None:
    """Recursively search common font dirs for a CJK-capable font file."""
    for root in _GLOB_ROOTS:
        try:
            base = Path(root)
        except Exception:
            continue
        if not base.exists():
            continue
        try:
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                name = p.name.lower()
                if not any(hint in name for hint in _CJK_FONT_HINTS):
                    continue
                if _try_load(str(p), 20) is not None:
                    return str(p)
        except PermissionError:
            continue
        except OSError:
            continue
    return None


def _fc_match_font() -> str | None:
    """Use fontconfig's fc-match to resolve a default sans font path."""
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    candidate = out.stdout.strip()
    if candidate and Path(candidate).exists() and _try_load(candidate, 20) is not None:
        return candidate
    return None


def _resolve_font_path() -> str | None:
    """Resolve a CJK-capable font path via known paths, glob, then fc-match."""
    global _CACHED_FONT_PATH
    if _CACHED_FONT_PATH is not None:
        return _CACHED_FONT_PATH

    # 1. Known explicit paths.
    for p in _KNOWN_FONT_PATHS:
        if Path(p).exists() and _try_load(p, 20) is not None:
            _CACHED_FONT_PATH = p
            return p

    # 2. Recursive glob search.
    found = _glob_find_cjk_font()
    if found:
        _CACHED_FONT_PATH = found
        return found

    # 3. fc-match fallback.
    matched = _fc_match_font()
    if matched:
        _CACHED_FONT_PATH = matched
        return matched

    # 4. Nothing found; signal fallback.
    return None


def _font(size: int) -> ImageFont.ImageFont:
    """Return a CJK-capable ImageFont, falling back to PIL default if absent."""
    path = _resolve_font_path()
    if path is not None:
        f = _try_load(path, size)
        if f is not None:
            return f
    return ImageFont.load_default()


def has_cjk_font() -> bool:
    """True if a font capable of rendering CJK was found (heuristic check)."""
    f = _font(20)
    try:
        return f.getbbox("形") is not None and f != ImageFont.load_default()
    except Exception:
        return False


def _wrap_text(draw, text, font, max_width):
    """Wrap `text` to fit within `max_width` pixels, breaking per character."""
    lines: list[str] = []
    buf = ""
    for ch in text:
        if ch == "\n":
            if buf:
                lines.append(buf)
            buf = ""
            continue
        trial = buf + ch
        if draw.textlength(trial, font=font) <= max_width:
            buf = trial
        else:
            if buf:
                lines.append(buf)
            buf = ch
    if buf:
        lines.append(buf)
    return lines


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

    # --- Left column: portrait thumbnail ---
    pw, ph = 500, 760
    port = portrait.convert("RGB").copy()
    port.thumbnail((pw, ph), Image.Resampling.LANCZOS)
    px = 50
    py = 60 + (ph - port.height) // 2
    canvas.paste(port, (px, py))
    # subtle 2px border around portrait
    draw.rectangle(
        (px - 1, py - 1, px + port.width, py + port.height),
        outline=(80, 90, 110),
        width=2,
    )

    # --- Right column ---
    rx = 620
    content_w = 730
    right_edge = rx + content_w  # 1350
    bottom_limit = 860

    title_f = _font(40)
    oneliner_f = _font(20)
    body_f = _font(20)
    small_f = _font(15)

    # Title (name) at y=50
    draw.text((rx, 50), card.name, fill=(240, 244, 255), font=title_f)

    # one_liner at y=110
    draw.text((rx, 110), card.one_liner, fill=(160, 180, 210), font=oneliner_f)

    # Radar at (620, 160), size 300
    radar = draw_radar(card.affinities, size=300)
    canvas.paste(radar, (rx, 160))

    # Lore block starts at (620, 490), full right-column width
    world = load_world()
    lore_entries = [
        ("主系", card.primary_affinity),
        ("灵域", card.spirit_domain),
        ("外形", card.appearance),
        ("性格", card.personality),
        ("背景", card.backstory),
        ("展示", card.ability_showcase),
    ]

    lore_x = rx
    lore_y = 490
    lore_max_w = right_edge - lore_x  # 730
    line_h = int(20 * 1.4)  # ~28px per line at body font size 20

    for label, value in lore_entries:
        if lore_y > bottom_limit:
            break
        combined = f"{label} · {value}"
        wrapped = _wrap_text(draw, combined, body_f, lore_max_w)
        for ln in wrapped:
            if lore_y > bottom_limit:
                break
            draw.text((lore_x, lore_y), ln, fill=(210, 218, 230), font=body_f)
            lore_y += line_h
        # small gap between entries
        lore_y += 4

    # Footer at (50, 855)
    draw.text(
        (50, 855),
        f"{world.world_name} · {card.primary_affinity} · Local ROCm",
        fill=(100, 110, 130),
        font=small_f,
    )

    # CJK font warning if no CJK font found
    if not has_cjk_font():
        warn_f = _font(13)
        warn_text = "no CJK font — install fonts-noto-cjk"
        # bottom-right placement; measure then anchor to right edge
        tw = draw.textlength(warn_text, font=warn_f)
        draw.text(
            (right_edge - tw, 855),
            warn_text,
            fill=(200, 120, 80),
            font=warn_f,
        )

    canvas.save(out_path, format="PNG")
    return out_path