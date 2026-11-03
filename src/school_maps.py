"""
school_maps.py — real floor-plan image for schools that have one.

Central.pdf (workspace root, git-ignored, NEVER committed) holds a
scanned floor plan whose title block names the real school and address.
This module extracts the embedded scan, paints over the identifying
title block and labels, and saves a clean room-only PNG under
processed/figures/ (also git-ignored). The dashboard shows the clean
PNG; the PDF itself is never displayed or shared.

Only School_06 has a plan today; other schools keep the
schematic button map. Add entries to SCHOOL_PDFS as plans arrive.
"""

from __future__ import annotations

from pathlib import Path

# school -> pdf filename in the workspace root (all git-ignored)
SCHOOL_PDFS = {"School_06": "Central.pdf"}

# school -> room name -> (x, y) dot position, fractions of image size
# with (0, 0) at the top-left. Placed on each room's number label: box
# centers from dark-pixel centroids on the full-res scan, hand-checked
# against enlarged crops (rooms 174/132/136/138/105/106/108/m1 were all
# re-seated; the old 138 dot sat up in room 142's row, the old 108 dot on
# the "Community Center" caption text). Verified by dots-over-image.
SCHOOL_DOTS = {"School_06": {
    "Room_170": (0.080, 0.170),
    "Room_174": (0.196, 0.252),
    "Room_161": (0.359, 0.175),
    "Room_157": (0.482, 0.278),
    "Room_142": (0.657, 0.480),
    "Room_138": (0.716, 0.555),
    "Room_136": (0.576, 0.555),
    "Room_132": (0.408, 0.555),
    "Room_105": (0.528, 0.752),
    "Room_106": (0.577, 0.752),
    "Room_108": (0.674, 0.833),
    "Room_m1": (0.775, 0.769),
}}


def _redact(image):
    """Paint over the title block and caption text (right column and the
    school-name/scale captions, bottom right). Room areas are untouched.
    """
    from PIL import ImageDraw

    width, height = image.size
    paint = ImageDraw.Draw(image)
    white = (255, 255, 255)
    # Right-hand title column (firm name, school name, address).
    paint.rectangle([width * 0.839, 0, width, height], fill=white)
    # Bottom-right school-name caption + scale bar in light grey.
    # The patch sits right of room 108 (which ends near x 0.69), so no
    # room is touched; it covers the caption line and the scale below.
    # A 1-2 letter fragment of the caption may survive left of x 0.70,
    # over no room, and names nothing on its own.
    paint.rectangle([width * 0.700, height * 0.840,
                     width * 0.860, height * 0.918], fill=white)
    paint.rectangle([width * 0.700, height * 0.945,
                     width * 0.852, height * 0.990], fill=white)
    return image


def floor_plan_png(processed: Path, workspace: Path, school: str) -> Path | None:
    """Clean room-only PNG for a school's floor plan, or None.

    Extracts the scan from the PDF on first run (or when the PDF is
    newer), redacts identifying text, and caches the PNG. Returns None
    for schools without a plan.
    """
    from PIL import Image
    from pypdf import PdfReader

    pdf_name = SCHOOL_PDFS.get(school)
    if pdf_name is None:
        return None
    pdf = workspace / pdf_name
    out = processed / "figures" / f"{school}_map.png"
    if not pdf.exists():
        return None
    if out.exists() and out.stat().st_mtime >= pdf.stat().st_mtime:
        return out
    page = PdfReader(pdf).pages[0]
    if not page.images:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        handle.write(page.images[0].data)
    _redact(Image.open(out).convert("RGB")).save(out)
    return out
