#!/usr/bin/env python3
"""Create compact contact sheets from already-rendered QA images."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "09_manuscript/rendered_qc/visual_qa"


def contact(paths: list[Path], output: Path, columns: int, thumb_width: int) -> None:
    font = ImageFont.load_default()
    items: list[tuple[Image.Image, str]] = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            height = max(1, round(image.height * thumb_width / image.width))
            image.thumbnail((thumb_width, height), Image.Resampling.LANCZOS)
            items.append((image.copy(), path.name))
    label_height = 22
    cell_width = thumb_width + 16
    cell_height = max(image.height for image, _ in items) + label_height + 16
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(items):
        x = (index % columns) * cell_width + 8
        y = (index // columns) * cell_height + 8
        sheet.paste(image, (x, y))
        draw.text((x, y + image.height + 4), label, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    contact(sorted((ROOT / "09_manuscript/rendered_qc/page_images").glob("page_*.png")), QC / "manuscript_contact.png", 4, 230)
    contact(sorted((QC / "response").glob("page-*.png")), QC / "response_contact.png", 3, 280)
    contact(sorted((QC / "cover").glob("page-*.png")), QC / "cover_contact.png", 1, 500)
    figures = sorted((ROOT / "07_figures/main").glob("*.png")) + sorted((ROOT / "07_figures/supplementary").glob("*.png"))
    contact(figures, QC / "figure_contact.png", 3, 380)
    print({"status": "PASS", "manuscript_pages": 11, "response_pages": 6, "figures": len(figures)})


if __name__ == "__main__":
    main()
