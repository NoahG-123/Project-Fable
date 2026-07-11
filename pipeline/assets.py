"""Stage 3 — Assets.

Generate one stickman illustration per shot plus the cover via Pollinations.ai
(free REST API, no key). One random seed is generated per run and reused for
every image — this keeps the visual style consistent across the reel.

A failed image (after all retries) is substituted with a plain white
1080x1920 placeholder carrying the shot number. A missing image never kills
the run.
"""

import io
import logging
import os
import random
import time
import urllib.parse

import requests
from PIL import Image, ImageDraw, ImageFont

import branding

logger = logging.getLogger("grimm.assets")

POLLINATIONS_URL = (
    "https://image.pollinations.ai/prompt/{prompt}"
    "?width={width}&height={height}&model=flux&nologo=true&seed={seed}"
)
COVER_PROMPT_SUFFIX = (
    ", dynamic composition, dramatic pose, eye-catching, thumbnail-worthy, bold contrast"
)

SLEEP_BETWEEN_REQUESTS = 3
MAX_RETRIES = 3
RETRY_BACKOFF = 6
REQUEST_TIMEOUT = 180  # flux generations can take a while


def _fetch_log(images_dir, message):
    """Append a line to the image fetch log in the debug trail. Never raises."""
    logger.info("%s", message)
    try:
        debug_dir = os.path.join(os.path.dirname(images_dir), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "image_fetch_log.txt"), "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def _normalise_and_save(image_bytes, out_path):
    """Decode, convert to RGB and aspect-fill to canvas size, then save."""
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    image = image.convert("RGB")

    target_w, target_h = branding.CANVAS_WIDTH, branding.CANVAS_HEIGHT
    if image.size != (target_w, target_h):
        scale = max(target_w / image.width, target_h / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
        left = (resized.width - target_w) // 2
        top = (resized.height - target_h) // 2
        image = resized.crop((left, top, left + target_w, top + target_h))

    image.save(out_path, "PNG")


def _make_placeholder(label, out_path):
    """Plain white 1080x1920 PNG with the shot label in black text."""
    image = Image.new(
        "RGB",
        (branding.CANVAS_WIDTH, branding.CANVAS_HEIGHT),
        branding.hex_to_rgb(branding.BACKGROUND),
    )
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(branding.FONT_TITLE_PATH, branding.TITLE_FONT_SIZE)
    except (OSError, TypeError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    x = (branding.CANVAS_WIDTH - (bbox[2] - bbox[0])) // 2
    y = (branding.CANVAS_HEIGHT - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), label, fill=branding.hex_to_rgb(branding.INK), font=font)
    image.save(out_path, "PNG")


def _fetch_image(prompt, seed, out_path, label, images_dir):
    """Fetch one image with retries. Falls back to a placeholder. Returns True if real image."""
    url = POLLINATIONS_URL.format(
        prompt=urllib.parse.quote(prompt),
        width=branding.CANVAS_WIDTH,
        height=branding.CANVAS_HEIGHT,
        seed=seed,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200 and response.content:
                _normalise_and_save(response.content, out_path)
                _fetch_log(images_dir, f"OK   {label} (attempt {attempt}, {len(response.content)} bytes)")
                return True
            _fetch_log(
                images_dir,
                f"FAIL {label} attempt {attempt}: HTTP {response.status_code}",
            )
        except (requests.RequestException, OSError, Exception) as exc:  # noqa: BLE001
            _fetch_log(images_dir, f"FAIL {label} attempt {attempt}: {exc}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF)

    _fetch_log(images_dir, f"PLACEHOLDER {label}: all {MAX_RETRIES} attempts failed")
    _make_placeholder(label, out_path)
    return False


def fetch_all_images(script, working_dir="working"):
    """Fetch the cover plus one image per shot.

    Returns {"cover": path, "shots": [paths in shot order], "placeholders": int}.
    Never returns None — placeholders guarantee a full image set.
    """
    images_dir = os.path.join(working_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    seed = random.randint(0, 2**31 - 1)
    _fetch_log(images_dir, f"Seed for this run: {seed}")

    placeholders = 0

    cover_path = os.path.join(images_dir, "cover.png")
    cover_prompt = script["cover"]["visual_prompt"] + COVER_PROMPT_SUFFIX
    if not _fetch_image(cover_prompt, seed, cover_path, "Cover", images_dir):
        placeholders += 1

    shot_paths = []
    for index, shot in enumerate(script["shots"], start=1):
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        shot_path = os.path.join(images_dir, f"shot_{index:02d}.png")
        if not _fetch_image(
            shot["visual_prompt"], seed, shot_path, f"Shot {index}", images_dir
        ):
            placeholders += 1
        shot_paths.append(shot_path)

    logger.info(
        "Fetched %d images (%d placeholders)", len(shot_paths) + 1, placeholders
    )
    return {"cover": cover_path, "shots": shot_paths, "placeholders": placeholders}
