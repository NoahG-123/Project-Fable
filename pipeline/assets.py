"""Stage 3 — Assets (V2).

Each shot's visual_prompt has the format "scene: <bg> | character: <char>".
The pipeline makes two Pollinations calls per shot — a background and a
character sprite — then composites them with Pillow. The cover is a background
call only, full frame, no character.

Every call is anchored to the user's style_reference.png (committed under
assets/) via the &image= parameter, using a raw githubusercontent URL built
from the GITHUB_REPOSITORY environment variable.

A failed image (after retries) is substituted with a plain brand-background
placeholder carrying the shot label. A missing image never kills the run.
"""

import io
import logging
import os
import time
import urllib.parse

import requests
from PIL import Image, ImageDraw, ImageFont

import branding

logger = logging.getLogger("grimm.assets")

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/{prompt}"

BG_PREFIX = (
    "storybook illustration background, {desc}, muted earthy color palette, "
    "hand-drawn aesthetic, clean line work, no characters, no people, no animals"
)
CHAR_PREFIX = (
    "stickman character, transparent background, {desc}, expressive hand-drawn face, "
    "bold black outlines, minimal shading, muted earthy tones, full body visible, isolated figure"
)

SLEEP_BETWEEN_REQUESTS = 3
MAX_RETRIES = 3
RETRY_BACKOFF = 6
REQUEST_TIMEOUT = 180

BG_WIDTH, BG_HEIGHT = 1080, 1920
CHAR_WIDTH, CHAR_HEIGHT = 540, 960

WHITE_THRESHOLD = 240        # pixels brighter than this become transparent
CHARACTER_HEIGHT_RATIO = 0.60  # character ~60% of frame height


def style_reference_url():
    """Build the raw githubusercontent URL for assets/style_reference.png, or ''."""
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not repo:
        logger.warning("GITHUB_REPOSITORY not set — image generation will run without a style reference")
        return ""
    return f"https://raw.githubusercontent.com/{repo}/main/assets/style_reference.png"


def _fetch_log(images_dir, message):
    logger.info("%s", message)
    try:
        debug_dir = os.path.join(os.path.dirname(images_dir), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "image_fetch_log.txt"), "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def parse_visual_prompt(visual_prompt):
    """Split 'scene: X | character: Y' into (scene_desc, character_desc).

    Tolerant of missing labels/pipe. Returns (scene, character) with character
    possibly empty (cover shots have no character)."""
    text = (visual_prompt or "").strip()
    scene, character = text, ""
    if "|" in text:
        left, right = text.split("|", 1)
        scene, character = left.strip(), right.strip()
    for label in ("scene:", "Scene:"):
        if scene.startswith(label):
            scene = scene[len(label):].strip()
            break
    for label in ("character:", "Character:"):
        if character.startswith(label):
            character = character[len(label):].strip()
            break
    return scene, character


def _build_url(prompt, width, height, seed, style_ref):
    url = POLLINATIONS_BASE.format(prompt=urllib.parse.quote(prompt))
    params = f"?width={width}&height={height}&model=flux&nologo=true&seed={seed}"
    if style_ref:
        params += "&image=" + urllib.parse.quote(style_ref, safe="")
    return url + params


def _download(url, timeout=REQUEST_TIMEOUT):
    response = requests.get(url, timeout=timeout)
    if response.status_code == 200 and response.content:
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image, len(response.content)
    raise RuntimeError(f"HTTP {response.status_code}")


def _fetch_with_retries(url, label, images_dir):
    """Return a loaded PIL image or None after all retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            image, size = _download(url)
            _fetch_log(images_dir, f"OK   {label} (attempt {attempt}, {size} bytes)")
            return image
        except Exception as exc:  # noqa: BLE001
            _fetch_log(images_dir, f"FAIL {label} attempt {attempt}: {exc}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF)
    return None


def _aspect_fill(image, width, height):
    image = image.convert("RGB")
    if image.size != (width, height):
        scale = max(width / image.width, height / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS
        )
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        image = resized.crop((left, top, left + width, top + height))
    return image


def _placeholder(label):
    image = Image.new("RGB", (branding.CANVAS_WIDTH, branding.CANVAS_HEIGHT),
                      branding.hex_to_rgb(branding.BACKGROUND))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(branding.FONT_TITLE_PATH, branding.TITLE_FONT_SIZE)
    except (OSError, TypeError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    x = (branding.CANVAS_WIDTH - (bbox[2] - bbox[0])) // 2
    y = (branding.CANVAS_HEIGHT - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), label, fill=branding.hex_to_rgb(branding.INK), font=font)
    return image


def _key_out_white(character_image):
    """Replace near-white pixels with transparency. Returns RGBA."""
    rgba = character_image.convert("RGBA")
    pixels = rgba.getdata()
    out = []
    for r, g, b, a in pixels:
        if r >= WHITE_THRESHOLD and g >= WHITE_THRESHOLD and b >= WHITE_THRESHOLD:
            out.append((r, g, b, 0))
        else:
            out.append((r, g, b, a))
    rgba.putdata(out)
    return rgba


def _composite(background_rgb, character_rgba):
    """Paste the keyed character onto the background, ~60% height, bottom-centred."""
    canvas = background_rgb.convert("RGBA")
    target_h = int(branding.CANVAS_HEIGHT * CHARACTER_HEIGHT_RATIO)
    scale = target_h / character_rgba.height
    char = character_rgba.resize(
        (max(1, round(character_rgba.width * scale)), target_h), Image.LANCZOS
    )
    x = (branding.CANVAS_WIDTH - char.width) // 2
    # bottom third: sit the character so its feet are near the bottom margin
    y = branding.CANVAS_HEIGHT - char.height - int(branding.CANVAS_HEIGHT * 0.04)
    canvas.alpha_composite(char, (max(0, x), max(0, y)))
    return canvas.convert("RGB")


def generate_part_images(part, part_index, seed, style_ref, working_dir="working"):
    """Fetch cover + per-shot composited images for one part.

    Returns {"cover": path, "shots": [paths], "placeholders": int}. Never None.
    Files are namespaced by part to avoid collisions across parts.
    """
    images_dir = os.path.join(working_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    placeholders = 0
    tag = f"part{part_index}"

    # Cover — background only, full frame.
    cover_path = os.path.join(images_dir, f"cover_{tag}.png")
    cover_scene, _ = parse_visual_prompt(part["cover"]["visual_prompt"])
    cover_prompt = BG_PREFIX.format(desc=cover_scene)
    cover_img = _fetch_with_retries(
        _build_url(cover_prompt, BG_WIDTH, BG_HEIGHT, seed, style_ref),
        f"{tag} Cover BG", images_dir,
    )
    if cover_img is None:
        _fetch_log(images_dir, f"PLACEHOLDER {tag} Cover")
        _placeholder(f"{tag} Cover").save(cover_path, "PNG")
        placeholders += 1
    else:
        _aspect_fill(cover_img, BG_WIDTH, BG_HEIGHT).save(cover_path, "PNG")

    shot_paths = []
    for shot in part["shots"]:
        n = shot.get("shot_number", len(shot_paths) + 1)
        scene, character = parse_visual_prompt(shot["visual_prompt"])
        shot_path = os.path.join(images_dir, f"{tag}_shot_{n:02d}.png")
        label = f"{tag} Shot {n}"

        time.sleep(SLEEP_BETWEEN_REQUESTS)
        bg_img = _fetch_with_retries(
            _build_url(BG_PREFIX.format(desc=scene), BG_WIDTH, BG_HEIGHT, seed, style_ref),
            f"{label} BG", images_dir,
        )

        char_img = None
        if character:
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            char_img = _fetch_with_retries(
                _build_url(CHAR_PREFIX.format(desc=character), CHAR_WIDTH, CHAR_HEIGHT, seed, style_ref),
                f"{label} CHAR", images_dir,
            )

        try:
            if bg_img is None:
                raise RuntimeError("background missing")
            background = _aspect_fill(bg_img, BG_WIDTH, BG_HEIGHT)
            if char_img is not None:
                composed = _composite(background, _key_out_white(char_img))
            else:
                composed = background
            composed.save(shot_path, "PNG")
            if bg_img is None or (character and char_img is None):
                placeholders += 1
        except Exception as exc:  # noqa: BLE001
            _fetch_log(images_dir, f"PLACEHOLDER {label}: composite failed ({exc})")
            _placeholder(label).save(shot_path, "PNG")
            placeholders += 1

        shot_paths.append(shot_path)

    logger.info("Part %d: fetched %d images (%d placeholders)", part_index, len(shot_paths) + 1, placeholders)
    return {"cover": cover_path, "shots": shot_paths, "placeholders": placeholders}
