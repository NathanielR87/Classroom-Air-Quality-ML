"""
sam_rooms.py — find classroom boxes on a floor plan with SAM 3.1.

An engineer drops a new school's floor plan in; hand-placing the
dashboard's clickable dots does not scale. SAM 3.1 (Meta Model API,
model "sam-3.1", Responses API) segments "room" regions and returns
one box per object as text records:

    <0f>0<|box;x1=10;y1=20;x2=14;y2=24;w=100;h=80|><|mask;...|>

(Box/mask grammar: meta-models/meta-sam protocol/sam3.md; request
shape: meta-model-cookbook 07_sam/01_api_basics.) Only the box
centers are needed for dots, so this file parses boxes itself and
needs no extra packages. The official meta-sam-parser would also
decode the masks (pixel centroids instead of box centers) — a
refinement, not a requirement.

Coordinates come back in source pixels; centers are written as
fractions of image size, exactly the SCHOOL_DOTS format in
school_maps.py. Suggestions only: a person still checks each dot
against the plan before it ships.

Run it:
  source .venv/bin/activate
  python src/sam_rooms.py --dry-run          # build the request only
  MODEL_API_KEY="..." python src/sam_rooms.py
  (META_MODEL_API_KEY also works; same key as the principal letter.)
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
from pathlib import Path

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"
BOX_TOKEN = re.compile(r"<\|box;([^|]*)\|>")


def image_data_url(path: Path) -> str:
    """Encode a local image as the data URL the API expects.

    Oversized scans are shrunk to 1600 px on the long side first:
    box coordinates return in sent-image pixels, and centers are
    fractions, so dots land identically either way.
    """
    import io

    from PIL import Image

    media, _ = mimetypes.guess_type(path.name)
    if media is None:
        raise SystemExit(f"Cannot tell the image type of {path}.")
    image = Image.open(path).convert("RGB")
    if max(image.size) > 1600:
        image.thumbnail((1600, 1600))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return ("data:image/png;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii"))


def build_input(concept: str, data_url: str) -> list:
    """One Responses API input: a noun phrase plus the image."""
    return [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": concept},
        {"type": "input_image", "image_url": data_url}]}]


def find_key() -> str:
    """SAM key: MODEL_API_KEY, else the letter's META_MODEL_API_KEY."""
    return (os.environ.get("MODEL_API_KEY", "")
            or os.environ.get("META_MODEL_API_KEY", ""))


def collect_text(concept: str, data_url: str, key: str) -> str:
    """Stream one SAM request and gather its output_text lane."""
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=BASE_URL)
    parts, final = [], ""
    with client.responses.create(
            model=MODEL, stream=True,
            input=build_input(concept, data_url)) as stream:
        for event in stream:
            kind = getattr(event, "type", "")
            if kind == "response.output_text.delta":
                parts.append(event.delta)
            elif kind == "response.output_text.done":
                final = event.text
    return final or "".join(parts)


def parse_boxes(text: str) -> list[dict]:
    """Box records -> center fractions, straight from the protocol.

    Fields are read by name (their order is not significant), boxes
    are source pixels inside w x h, and x2/y2 are inclusive.
    """
    found = []
    for match in BOX_TOKEN.finditer(text):
        fields = {}
        for pair in match.group(1).split(";"):
            name, _, value = pair.partition("=")
            fields[name.strip()] = value.strip()
        try:
            x1, y1 = int(fields["x1"]), int(fields["y1"])
            x2, y2 = int(fields["x2"]), int(fields["y2"])
            width, height = int(fields["w"]), int(fields["h"])
        except (KeyError, ValueError):
            continue  # a malformed record must not kill the rest
        if width <= 0 or height <= 0:
            continue
        found.append({
            "x": (x1 + x2 + 1) / 2 / width,
            "y": (y1 + y2 + 1) / 2 / height,
            "w": (x2 - x1 + 1) / width,
            "h": (y2 - y1 + 1) / height,
        })
    return found


def suggest_dots(boxes: list[dict]) -> list[dict]:
    """Drop whole-plan and speck boxes; order reading-style."""
    kept = [b for b in boxes
            if 0.002 <= b["w"] * b["h"] <= 0.6]
    return sorted(kept, key=lambda b: (round(b["y"], 1), b["x"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM 3.1 room finder.")
    parser.add_argument("--input", type=Path,
                        default=Path("processed/figures/School_06_map.png"),
                        help="Floor-plan image (default: the redacted "
                             "School 06 plan the dashboard shows)")
    parser.add_argument("--concept", action="append", default=None,
                        help="Noun phrase to segment; repeatable "
                             "(default: room)")
    parser.add_argument("--output", type=Path,
                        default=Path("processed/sam_room_dots.csv"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Describe the request; call nothing.")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"No such image: {args.input}")
    concepts = args.concept or ["room"]
    data_url = image_data_url(args.input)
    if args.dry_run:
        print(f"Would send {args.input} ({len(data_url)} data-url chars) "
              f"to {MODEL} at {BASE_URL} for concepts {concepts}.")
        print("Boxes return as <0f>id<|box;...|><|mask;...|> lines; "
              "centers become dot fractions. No call made.")
        return
    key = find_key()
    if not key:
        raise SystemExit("Set MODEL_API_KEY (or META_MODEL_API_KEY) "
                         "in this session first.")
    boxes = []
    for concept in concepts:
        boxes += parse_boxes(collect_text(concept, data_url, key))
    dots = suggest_dots(boxes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["dot,x,y,box_w,box_h"]
    lines += [f"sam_{i},{d['x']:.3f},{d['y']:.3f},"
              f"{d['w']:.3f},{d['h']:.3f}"
              for i, d in enumerate(dots, 1)]
    args.output.write_text("\n".join(lines) + "\n")
    print(f"{len(dots)} suggested dots -> {args.output}")
    print("Paste-ready for SCHOOL_DOTS (rename sam_N to the real "
          "room names after checking each against the plan):")
    for i, d in enumerate(dots, 1):
        print(f'    "Room_sam{i}": ({d["x"]:.3f}, {d["y"]:.3f}),')


if __name__ == "__main__":
    main()
