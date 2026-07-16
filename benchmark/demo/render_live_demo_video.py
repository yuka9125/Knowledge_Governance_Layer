#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a silent live-demo video using actual demo outputs and screenshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_DIR = REPO_ROOT / "_system" / "data" / "phase_f_live_demo"
FONT_REGULAR = Path("C:/Windows/Fonts/BIZ-UDGothicR.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/BIZ-UDGothicB.ttc")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    lines: List[str] = []
    for raw in str(text).splitlines():
        current = ""
        for ch in raw:
            candidate = current + ch
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = ch
        lines.append(current)
    return lines


def _text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill: str, max_width: int, gap: int = 8) -> int:
    x, y = xy
    for line in _wrap(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + gap
    return y


def _rounded(draw: ImageDraw.ImageDraw, box, fill, outline="#d8dee8", radius=10, width=1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _base(width: int, height: int, progress: float, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    for y in range(height):
        blend = y / max(1, height - 1)
        draw.line((0, y, width, y), fill=(int(238 + 17 * blend), int(244 + 11 * blend), 255))
    _rounded(draw, (44, 30, 88, 74), "#2563eb", "#2563eb", radius=8)
    draw.text((60, 38), "K", font=_font(24, True), fill="#ffffff")
    draw.text((104, 30), "Knowledge Governance Layer", font=_font(25, True), fill="#172033")
    draw.text((104, 60), title, font=_font(14), fill="#627084")
    _rounded(draw, (width - 344, 36, width - 44, 68), "#eef3ff", "#b9c8ff", radius=16)
    draw.text((width - 325, 43), "Synthetic demo data / No customer data", font=_font(13, True), fill="#1d4ed8")
    _rounded(draw, (44, 94, width - 44, height - 62), "#ffffff", "#d8dee8", radius=10)
    x0, y0, x1, y1 = 44, height - 36, width - 190, height - 28
    _rounded(draw, (x0, y0, x1, y1), "#dbe4ef", "#dbe4ef", radius=5)
    _rounded(draw, (x0, y0, int(x0 + (x1 - x0) * progress), y1), "#2563eb", "#2563eb", radius=5)
    seconds = int(progress * 100)
    draw.text((width - 160, height - 43), f"{seconds // 60:02d}:{seconds % 60:02d}", font=_font(14, True), fill="#627084")
    return img, draw


def _code_panel(draw, box, lines: List[str]) -> None:
    _rounded(draw, box, "#0f172a", "#0f172a", radius=8)
    y = box[1] + 18
    for line in lines:
        draw.text((box[0] + 18, y), line, font=_font(15), fill="#e2e8f0")
        y += 25


def _fit_image(path: Path, box: Tuple[int, int, int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    max_w = box[2] - box[0]
    max_h = box[3] - box[1]
    ratio = min(max_w / img.width, max_h / img.height)
    return img.resize((int(img.width * ratio), int(img.height * ratio)))


def _paste_screenshot(base: Image.Image, draw, path: Path, box) -> None:
    _rounded(draw, box, "#f8fafc", "#d8dee8", radius=8)
    if not path.exists():
        _text(draw, (box[0] + 24, box[1] + 24), f"Screenshot missing:\n{path}", _font(18), "#c2410c", box[2] - box[0] - 48)
        return
    shot = _fit_image(path, (box[0] + 16, box[1] + 16, box[2] - 16, box[3] - 16))
    x = box[0] + (box[2] - box[0] - shot.width) // 2
    y = box[1] + (box[3] - box[1] - shot.height) // 2
    base.paste(shot, (x, y))


def _results_text(results_path: Path) -> List[str]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    lines: List[str] = []
    for row in data["results"]:
        result = row["result"]
        if result.get("answerable"):
            value = f"{row['scene']}: answerable=true / {result['knowledge_id']} / {result['answer']}"
        else:
            value = f"{row['scene']}: answerable=false / fallback={result.get('fallback')}"
        lines.append(value)
    return lines


def render_slide(index: int, width: int, height: int, progress: float, demo_dir: Path) -> Image.Image:
    img, draw = _base(width, height, progress, "Live screen-recording material - Phase F")
    x, y, right, bottom = 78, 128, width - 78, height - 96

    if index == 0:
        draw.text((x, y), "Live Demo Scope", font=_font(15, True), fill="#2563eb")
        _text(draw, (x, y + 38), "実画面で見せる4ステップ", _font(36, True), "#172033", 1050)
        bullets = [
            "1. knowledge_distillation で合成問い合わせログを正規化・候補化",
            "2. Excelレビューで P3-2確認（既存FAQ更新候補）を確認し、採用にする",
            "3. approved_knowledge.json に既存FAQ更新としてマージする",
            "4. Servingで Before / After / 該当なし の質問結果を確認する",
        ]
        _text(draw, (x, y + 130), "\n".join(bullets), _font(22), "#334155", 1060, gap=14)
    elif index == 1:
        draw.text((x, y), "Step 1: knowledge_distillation", font=_font(15, True), fill="#2563eb")
        _text(draw, (x, y + 38), "合成問い合わせログを入力し、候補生成まで実行", _font(32, True), "#172033", 1060)
        _code_panel(draw, (x, y + 130, right, y + 420), [
            "python benchmark/demo/prepare_live_demo.py",
            "python -m knowledge_distillation normalize --adapter csv ...",
            "-> input_count=2 / output_count=2",
            "python -m knowledge_distillation ingest --adapter csv --no-openai ...",
            "-> faq_count=2 / saved_count=2",
            "python -m knowledge_distillation export ...",
            "-> distillation_output/knowledge_export.json",
        ])
    elif index == 2:
        draw.text((x, y), "Step 2: Excel Review", font=_font(15, True), fill="#2563eb")
        _text(draw, (x, y + 38), "候補・既存FAQ比較・推奨アクションをSheet1で確認", _font(31, True), "#172033", 1060)
        _paste_screenshot(img, draw, demo_dir / "screenshots" / "01_excel_review_candidate.png", (x, y + 110, right, bottom))
    elif index == 3:
        draw.text((x, y), "Step 2: Approve", font=_font(15, True), fill="#2563eb")
        _text(draw, (x, y + 38), "レビュー結果を採用にして、既存FAQ更新として扱う", _font(31, True), "#172033", 1060)
        _paste_screenshot(img, draw, demo_dir / "screenshots" / "02_excel_current.png", (x, y + 110, right, bottom))
        _rounded(draw, (right - 365, y + 125, right - 30, y + 225), "#ecfdf5", "#bbf7d0", radius=8)
        _text(draw, (right - 345, y + 145), "実操作: Excelを開き、レビュー結果セルを採用へ変更", _font(18, True), "#166534", 295)
    elif index == 4:
        draw.text((x, y), "Step 3: Merge", font=_font(15, True), fill="#2563eb")
        _text(draw, (x, y + 38), "承認済み行だけをServing用Knowledgeへupsert", _font(32, True), "#172033", 1060)
        _code_panel(draw, (x, y + 130, right, y + 430), [
            "python -m knowledge_distillation.approved_knowledge_exporter",
            "  _system/data/phase_f_live_demo/03_FAQ_final_result_approved.xlsx",
            "  -o _system/data/phase_f_live_demo/approved_knowledge_merged.json",
            "  --base _system/data/phase_f_live_demo/approved_knowledge_before.json",
            "",
            "result: knowledge_id=demo-vpn-001 is updated in place",
        ])
    else:
        draw.text((x, y), "Step 4: Serving Questions", font=_font(15, True), fill="#2563eb")
        _text(draw, (x, y + 38), "Before / After / 該当なしを実Serviceで確認", _font(32, True), "#172033", 1060)
        lines = _results_text(demo_dir / "04_serving_query_results.json")
        _code_panel(draw, (x, y + 130, right, y + 470), [
            "from serving.governed_knowledge_api import GovernedKnowledgeService",
            "",
            *lines,
        ])
    return img


def _frames(width: int, height: int, fps: int, scale: float, demo_dir: Path) -> Iterable[Image.Image]:
    durations = [8, 14, 18, 14, 16, 18]
    counts = [max(1, int(d * scale * fps)) for d in durations]
    total = sum(counts)
    frame = 0
    for idx, count in enumerate(counts):
        for _ in range(count):
            yield render_slide(idx, width, height, frame / max(1, total - 1), demo_dir)
            frame += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Phase F live demo MP4.")
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_DEMO_DIR / "phase_f_live_demo.mp4")
    parser.add_argument("--poster", type=Path, default=DEFAULT_DEMO_DIR / "phase_f_live_demo_poster.png")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    args = parser.parse_args()

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("Missing imageio. Install with: python -m pip install imageio imageio-ffmpeg", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_slide(0, args.width, args.height, 0.0, args.demo_dir).save(args.poster)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for frame in _frames(args.width, args.height, args.fps, args.duration_scale, args.demo_dir):
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()
    print(f"Wrote {args.out}")
    print(f"Wrote {args.poster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
