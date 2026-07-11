"""Stage 5 — Assembly.

Build the final 1080x1920 30fps reel with MoviePy + FFmpeg:

- Cover card: cover.png full frame with the title readout overlaid, held for
  at least 3 seconds (driven by the audio segments from the voice stage).
- Story shots: each image is shown as a STILL for exactly as long as its
  narration takes to speak (windows come from real TTS segment boundaries).
- Kinetic captions: whisper word timestamps grouped into 2-3 word chunks,
  uppercase bold white on a dark rounded pill at 72% frame height, with a
  scale-pop entrance and one accent-coloured emphasis word per sentence.
  Captions are hidden during the cover card.
- Watermark: CHANNEL_HANDLE in the top-right corner of every frame.
- Audio: narration at full volume, optional music bed at -18dB.

All text is rendered with PIL into RGBA arrays and turned into ImageClips
with explicit masks — MoviePy 1.0.3's TextClip (ImageMagick) and
resize(lambda) both mangle semi-transparent masks, so neither is used.
"""

import logging
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import branding

logger = logging.getLogger("grimm.assemble")

FRAME = 1.0 / branding.FPS

# Scale-pop entrance: 1.12 -> 1.0 over ~80ms (two intermediate frames at 30fps)
POP_SCALES = [1.12, 1.06]

CAPTION_MIN_FONT_SIZE = 30
CAPTION_FONT_STEP = 4
CAPTION_PAD_X = 36
CAPTION_PAD_Y = 22
CAPTION_HANG = 0.6         # captions linger briefly after their words end
MUSIC_BED_DB = -18.0

STOPWORDS = {
    "the", "and", "that", "this", "with", "your", "you", "yours", "from", "have",
    "has", "had", "for", "are", "was", "were", "will", "would", "could", "should",
    "into", "onto", "then", "than", "them", "they", "their", "there", "here",
    "what", "when", "where", "which", "who", "whom", "very", "just", "about",
    "been", "being", "over", "under", "after", "before", "because", "but", "not",
}


def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except (OSError, TypeError):
        logger.warning("Could not load font %s — falling back to PIL default", path)
        return ImageFont.load_default()


def _rgba_to_clip(rgba_image):
    """PIL RGBA image -> masked ImageClip (semi-transparency preserved)."""
    from moviepy.editor import ImageClip

    array = np.array(rgba_image)
    clip = ImageClip(array[:, :, :3])
    mask = ImageClip(array[:, :, 3] / 255.0, ismask=True)
    return clip.set_mask(mask)


def _load_canvas_image(path):
    """Load an image as a canvas-sized RGB numpy array (aspect-fill)."""
    image = Image.open(path).convert("RGB")
    target = (branding.CANVAS_WIDTH, branding.CANVAS_HEIGHT)
    if image.size != target:
        scale = max(target[0] / image.width, target[1] / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
        left = (resized.width - target[0]) // 2
        top = (resized.height - target[1]) // 2
        image = resized.crop((left, top, left + target[0], top + target[1]))
    return np.array(image)


# ---------------------------------------------------------------------------
# Cover title overlay
# ---------------------------------------------------------------------------

def _wrap_text(text, font, max_width, draw):
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render_cover_title(title_readout):
    """Title readout as bold white text with a dark shadow, centred, RGBA."""
    canvas = Image.new("RGBA", (branding.CANVAS_WIDTH, branding.CANVAS_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    max_width = int(branding.CANVAS_WIDTH * branding.CAPTION_MAX_WIDTH_RATIO)

    size = branding.TITLE_FONT_SIZE
    while size > CAPTION_MIN_FONT_SIZE:
        font = _load_font(branding.FONT_TITLE_PATH, size)
        lines = _wrap_text(title_readout, font, max_width, draw)
        if len(lines) <= 4:
            break
        size -= CAPTION_FONT_STEP
    else:
        font = _load_font(branding.FONT_TITLE_PATH, size)
        lines = _wrap_text(title_readout, font, max_width, draw)

    line_height = int(size * 1.3)
    block_height = line_height * len(lines)
    y = (branding.CANVAS_HEIGHT - block_height) // 2

    # White text with a dark shadow AND a dark stroke — stickman covers have
    # white backgrounds, so shadow alone is not enough for readability.
    stroke = max(3, size // 16)
    ink = branding.hex_to_rgb(branding.INK)
    for line in lines:
        width = draw.textlength(line, font=font)
        x = (branding.CANVAS_WIDTH - width) // 2
        draw.text(
            (x + 4, y + 4), line, font=font, fill=(0, 0, 0, 160),
            stroke_width=stroke, stroke_fill=(0, 0, 0, 160),
        )
        draw.text(
            (x, y), line, font=font, fill=(255, 255, 255, 255),
            stroke_width=stroke, stroke_fill=ink + (255,),
        )
        y += line_height

    return canvas


# ---------------------------------------------------------------------------
# Kinetic captions
# ---------------------------------------------------------------------------

def _clean_word(text):
    return re.sub(r"^[^\w']+|[^\w']+$", "", text)


def _mark_emphasis(words):
    """Flag one emphasis word per sentence (longest non-stopword)."""
    flags = [False] * len(words)
    sentence_start = 0
    for index, word in enumerate(words):
        is_last = index == len(words) - 1
        ends_sentence = word["text"].rstrip('"”’').endswith((".", "!", "?"))
        if ends_sentence or is_last:
            best, best_len = None, 0
            for j in range(sentence_start, index + 1):
                cleaned = _clean_word(words[j]["text"]).lower()
                if len(cleaned) >= 4 and cleaned not in STOPWORDS and len(cleaned) > best_len:
                    best, best_len = j, len(cleaned)
            if best is None and index >= sentence_start:
                lengths = [
                    (len(_clean_word(words[j]["text"])), j)
                    for j in range(sentence_start, index + 1)
                ]
                best = max(lengths)[1]
            if best is not None:
                flags[best] = True
            sentence_start = index + 1
    return flags


def _group_caption_chunks(words, emphasis_flags):
    """Group words into 2-3 word chunks, breaking at sentence boundaries."""
    chunks = []
    current = []
    for index, word in enumerate(words):
        current.append({**word, "emphasis": emphasis_flags[index]})
        ends_sentence = word["text"].rstrip('"”’').endswith((".", "!", "?"))
        if len(current) == 3 or ends_sentence:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _render_caption_pill(chunk):
    """One caption chunk as an RGBA pill image at scale 1.0.

    Uppercase bold words on a dark semi-transparent rounded pill; the
    emphasis word renders in the brand accent colour. Font size steps down
    until the pill fits within 80% of the frame width.
    """
    display_words = []
    for word in chunk:
        cleaned = _clean_word(word["text"]).upper()
        if cleaned:
            display_words.append((cleaned, word["emphasis"]))
    if not display_words:
        return None

    max_pill_width = int(branding.CANVAS_WIDTH * branding.CAPTION_MAX_WIDTH_RATIO)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))

    size = branding.CAPTION_FONT_SIZE
    while True:
        font = _load_font(branding.FONT_CAPTION_PATH, size)
        space = probe.textlength("  ", font=font)
        text_width = sum(probe.textlength(w, font=font) for w, _ in display_words)
        text_width += space * (len(display_words) - 1)
        ascent, descent = font.getmetrics()
        text_height = ascent + descent
        pill_width = int(text_width + 2 * CAPTION_PAD_X)
        if pill_width <= max_pill_width or size <= CAPTION_MIN_FONT_SIZE:
            break
        size -= CAPTION_FONT_STEP

    pill_height = int(text_height + 2 * CAPTION_PAD_Y)
    pill = Image.new("RGBA", (pill_width, pill_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pill)

    bg = branding.hex_to_rgb(branding.CAPTION_BG) + (int(255 * branding.CAPTION_BG_OPACITY),)
    draw.rounded_rectangle(
        (0, 0, pill_width - 1, pill_height - 1), radius=pill_height // 2, fill=bg
    )

    white = branding.hex_to_rgb(branding.CAPTION_TEXT) + (255,)
    accent = branding.hex_to_rgb(branding.ACCENT) + (255,)
    x = CAPTION_PAD_X
    for word_text, emphasis in display_words:
        draw.text((x, CAPTION_PAD_Y), word_text, font=font, fill=accent if emphasis else white)
        x += probe.textlength(word_text, font=font) + space

    return pill


def _caption_clips(words, cover_end, total_duration):
    """Build all caption clips. Returns (clips, covered_seconds)."""
    story_words = [w for w in words if w["start"] >= cover_end - 0.2 and w["end"] > cover_end]
    if not story_words:
        logger.warning("No caption words after the cover card — captions will be absent")
        return [], 0.0

    emphasis_flags = _mark_emphasis(story_words)
    chunks = _group_caption_chunks(story_words, emphasis_flags)

    clips = []
    covered = 0.0
    for index, chunk in enumerate(chunks):
        start = max(chunk[0]["start"], cover_end)
        next_start = chunks[index + 1][0]["start"] if index + 1 < len(chunks) else total_duration
        end = min(next_start, chunk[-1]["end"] + CAPTION_HANG, total_duration)
        if end - start < FRAME:
            continue

        pill = _render_caption_pill(chunk)
        if pill is None:
            continue

        duration = end - start
        covered += duration
        centre_y = branding.CAPTION_Y_RATIO * branding.CANVAS_HEIGHT

        # Scale-pop entrance: a couple of single-frame scaled clips, then the
        # base pill for the remainder. Pre-rendered with PIL — never
        # resize(lambda), which destroys semi-transparent masks.
        remaining = duration
        cursor = start
        if duration > FRAME * (len(POP_SCALES) + 1):
            for scale in POP_SCALES:
                scaled = pill.resize(
                    (max(1, round(pill.width * scale)), max(1, round(pill.height * scale))),
                    Image.LANCZOS,
                )
                clip = (
                    _rgba_to_clip(scaled)
                    .set_start(cursor)
                    .set_duration(FRAME)
                    .set_position(
                        (
                            int((branding.CANVAS_WIDTH - scaled.width) / 2),
                            int(centre_y - scaled.height / 2),
                        )
                    )
                )
                clips.append(clip)
                cursor += FRAME
                remaining -= FRAME

        clip = (
            _rgba_to_clip(pill)
            .set_start(cursor)
            .set_duration(remaining)
            .set_position(
                (
                    int((branding.CANVAS_WIDTH - pill.width) / 2),
                    int(centre_y - pill.height / 2),
                )
            )
        )
        clips.append(clip)

    logger.info("Built %d caption chunks covering %.1fs", len(chunks), covered)
    return clips, covered


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------

def _watermark_clip(total_duration):
    handle = os.environ.get("CHANNEL_HANDLE", "").strip()
    if not handle:
        logger.warning("CHANNEL_HANDLE is not set — no watermark will be rendered")
        return None

    font = _load_font(branding.FONT_HANDLE_PATH, branding.HANDLE_FONT_SIZE)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    width = int(probe.textlength(handle, font=font)) + 4
    ascent, descent = font.getmetrics()
    height = ascent + descent + 4

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # Ink-coloured at partial opacity — the stickman frames are white, so a
    # white watermark would vanish. Small and unobtrusive per the brief.
    ink = branding.hex_to_rgb(branding.INK) + (170,)
    draw.text((2, 2), handle, font=font, fill=ink)

    x = branding.CANVAS_WIDTH - width - branding.WATERMARK_MARGIN
    y = branding.WATERMARK_MARGIN
    return (
        _rgba_to_clip(image)
        .set_start(0)
        .set_duration(total_duration)
        .set_position((x, y))
    )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def _build_audio(narration_path, total_duration):
    from moviepy.editor import AudioFileClip, CompositeAudioClip
    from moviepy.audio.fx.all import audio_loop, volumex

    narration = AudioFileClip(narration_path)

    music_path = os.environ.get("MUSIC_BED_PATH", "").strip()
    if not music_path:
        logger.info("No MUSIC_BED_PATH set — voice-only audio")
        return narration
    if not os.path.exists(music_path):
        logger.warning(
            "MUSIC_BED_PATH points to %s but the file does not exist — voice-only audio",
            music_path,
        )
        return narration

    try:
        music = AudioFileClip(music_path)
        if music.duration < total_duration:
            music = audio_loop(music, duration=total_duration)
        else:
            music = music.subclip(0, total_duration)
        music = volumex(music, 10 ** (MUSIC_BED_DB / 20.0))
        mixed = CompositeAudioClip([narration, music])
        mixed.fps = 44100
        logger.info("Mixed music bed at %.0fdB under voice", MUSIC_BED_DB)
        return mixed
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not mix music bed (%s) — continuing voice-only", exc)
        return narration


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def assemble_video(script, images, voice_info, working_dir="working"):
    """Assemble the reel. Returns an assembly-info dict or None on failure.

    Info dict: {"output_path", "duration", "cover_duration",
                "caption_seconds", "shots_rendered"}
    """
    try:
        from moviepy.editor import CompositeVideoClip, ImageClip
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not import MoviePy: %s", exc)
        return None

    output_dir = os.path.join(working_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reel.mp4")

    segments = voice_info["segments"]
    total_duration = voice_info["duration"]

    # Display windows tile the audio exactly: each segment holds its image
    # from the previous segment's display_until to its own.
    clips = []
    window_start = 0.0
    cover_duration = None
    shots_rendered = 0

    for segment in segments:
        window_end = min(segment["display_until"], total_duration)
        if segment is segments[-1]:
            window_end = total_duration  # last image holds to the end of audio
        duration = window_end - window_start
        if duration <= 0:
            continue

        if segment["kind"] == "cover":
            image_path = images["cover"]
            cover_duration = window_end
        else:
            image_path = images["shots"][segment["shot_index"]]
            shots_rendered += 1

        try:
            frame = _load_canvas_image(image_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not load %s (%s) — using white frame", image_path, exc)
            frame = np.full(
                (branding.CANVAS_HEIGHT, branding.CANVAS_WIDTH, 3), 255, dtype=np.uint8
            )

        clips.append(
            ImageClip(frame).set_start(window_start).set_duration(duration)
        )
        window_start = window_end

    if cover_duration is None:
        cover_duration = min(3.0, total_duration)

    # Cover title overlay
    try:
        title_overlay = (
            _rgba_to_clip(_render_cover_title(script["cover"]["title_readout"]))
            .set_start(0)
            .set_duration(cover_duration)
            .set_position((0, 0))
        )
        clips.append(title_overlay)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render cover title overlay: %s", exc)

    # Kinetic captions (hidden during the cover card)
    caption_seconds = 0.0
    try:
        caption_clips, caption_seconds = _caption_clips(
            voice_info.get("words") or [], cover_duration, total_duration
        )
        clips.extend(caption_clips)
    except Exception as exc:  # noqa: BLE001
        logger.error("Caption rendering failed (%s) — continuing without captions", exc)

    # Watermark
    try:
        watermark = _watermark_clip(total_duration)
        if watermark is not None:
            clips.append(watermark)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render watermark: %s", exc)

    # Audio
    try:
        audio = _build_audio(voice_info["audio_path"], total_duration)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load narration audio: %s", exc)
        return None

    try:
        video = (
            CompositeVideoClip(clips, size=(branding.CANVAS_WIDTH, branding.CANVAS_HEIGHT))
            .set_duration(total_duration)
            .set_audio(audio)
        )
        logger.info(
            "Rendering %.1fs video (%d clips) to %s…", total_duration, len(clips), output_path
        )
        video.write_videofile(
            output_path,
            fps=branding.FPS,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(output_dir, "temp_audio.m4a"),
            remove_temp=True,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-maxrate", "8000k",
                "-bufsize", "16000k",
            ],
            threads=2,
            verbose=False,
            logger=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Video export failed: %s", exc)
        return None

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        logger.error("Export reported success but %s is missing or empty", output_path)
        return None

    logger.info(
        "Assembled reel: %.1fs, %d shots, %.1fs captions, %.1fMB",
        total_duration,
        shots_rendered,
        caption_seconds,
        os.path.getsize(output_path) / 1e6,
    )
    return {
        "output_path": output_path,
        "duration": total_duration,
        "cover_duration": cover_duration,
        "caption_seconds": caption_seconds,
        "shots_rendered": shots_rendered,
    }
