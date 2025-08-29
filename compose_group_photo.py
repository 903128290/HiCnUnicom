#!/usr/bin/env python3

"""
Compose a realistic group photo from two individual portraits without altering faces.

Usage:
  python compose_group_photo.py \
    --img1 /path/to/person1.jpg \
    --img2 /path/to/person2.jpg \
    --out /path/to/output.png \
    [--background /path/to/bg.jpg] [--left-first]

Notes:
  - By default, a clean studio-like gradient background is generated so neither
    subject is visually changed. If --background is provided, it will be used
    (resized to fit) as the shared scene.
  - The script auto-segments people using rembg, trims to subject bounds,
    scales them to similar heights, and places them side-by-side with natural
    spacing so it reads as a group photo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
from rembg import remove


def _ensure_image_mode_rgba(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGBA")
    return image


def _remove_background_keep_subject(image: Image.Image) -> Image.Image:
    """Return an RGBA image with background removed using rembg.

    The returned image retains the original colors and alpha transparency
    around hair/edges as produced by the model.
    """
    image = image.convert("RGBA")
    fg = remove(image)
    # Some rembg versions may return bytes; ensure PIL Image
    if not isinstance(fg, Image.Image):
        fg = Image.open(io.BytesIO(fg)).convert("RGBA")  # type: ignore[name-defined]
    return fg


def _trim_to_alpha_bbox(image: Image.Image, padding: int = 2) -> Image.Image:
    image = _ensure_image_mode_rgba(image)
    alpha = image.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    return image.crop((left, top, right, bottom))


def _scale_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    scale = target_height / float(image.height)
    new_w = max(1, int(round(image.width * scale)))
    return image.resize((new_w, target_height), Image.LANCZOS)


def _create_gradient_background(size: Tuple[int, int]) -> Image.Image:
    width, height = size
    # Light vertical gradient reminiscent of studio backdrops
    top_rgb = np.array([246, 246, 248], dtype=np.float32)
    bottom_rgb = np.array([230, 230, 234], dtype=np.float32)
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    grad = (top_rgb * (1 - t) + bottom_rgb * t).astype(np.uint8)
    grad = np.repeat(grad, width, axis=1)
    bg = Image.fromarray(np.dstack([grad, np.full((height, width), 255, np.uint8)]), mode="RGBA")
    # Slight blur for smoothness
    return bg.filter(ImageFilter.GaussianBlur(radius=0.8))


@dataclass
class Subject:
    image: Image.Image  # RGBA trimmed person


def _load_subject(path: Path) -> Subject:
    src = Image.open(path).convert("RGB")
    cut = _remove_background_keep_subject(src)
    cut = _trim_to_alpha_bbox(cut, padding=2)
    return Subject(image=cut)


def _compose(side_a: Subject, side_b: Subject, *, background: Optional[Image.Image], left_first: bool) -> Image.Image:
    # Harmonize heights to the median of both to avoid distorting anyone
    heights = sorted([side_a.image.height, side_b.image.height])
    target_h = max(480, int(round(sum(heights) / 2)))
    person_a = _scale_to_height(side_a.image, target_h)
    person_b = _scale_to_height(side_b.image, target_h)

    # Canvas sizing with margins
    margin = max(24, int(round(target_h * 0.08)))
    spacing = max(12, int(round(target_h * 0.04)))
    canvas_w = person_a.width + person_b.width + margin * 2 + spacing
    canvas_h = target_h + margin * 2

    if background is None:
        canvas = _create_gradient_background((canvas_w, canvas_h))
    else:
        bg_rgba = _ensure_image_mode_rgba(background)
        canvas = bg_rgba.resize((canvas_w, canvas_h), Image.LANCZOS)

    # Placement: slight vertical alignment to bottom so feet align naturally
    y_base = canvas_h - margin - target_h

    if left_first:
        left_img, right_img = person_a, person_b
    else:
        left_img, right_img = person_b, person_a

    # Natural slight overlap
    overlap = max(0, int(round(min(left_img.width, right_img.width) * 0.05)))
    left_x = margin
    right_x = margin + left_img.width + spacing - overlap

    canvas.alpha_composite(left_img, dest=(left_x, y_base))
    canvas.alpha_composite(right_img, dest=(right_x, y_base))

    return canvas.convert("RGB")


def _maybe_load_background(path: Optional[Path]) -> Optional[Image.Image]:
    if path is None:
        return None
    img = Image.open(path)
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a two-person group photo without altering faces")
    p.add_argument("--img1", required=True, type=Path, help="Path to first portrait image")
    p.add_argument("--img2", required=True, type=Path, help="Path to second portrait image")
    p.add_argument("--out", required=True, type=Path, help="Output image path (PNG/JPG)")
    p.add_argument("--background", type=Path, default=None, help="Optional shared background image")
    p.add_argument("--left-first", action="store_true", help="Place img1 on the left (default places img2 on left)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    subj1 = _load_subject(args.img1)
    subj2 = _load_subject(args.img2)
    bg = _maybe_load_background(args.background)

    composed = _compose(subj1, subj2, background=bg, left_first=bool(args.left_first))

    out_ext = args.out.suffix.lower()
    if out_ext in {".jpg", ".jpeg"}:
        composed.save(args.out, format="JPEG", quality=95)
    else:
        composed.save(args.out, format="PNG")

    print(f"Saved group photo -> {args.out}")


if __name__ == "__main__":
    main()

