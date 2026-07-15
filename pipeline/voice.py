"""Stage 4 — Voice (V2).

Generate narration with Chatterbox TTS via the Hugging Face Space
(ResembleAI/Chatterbox) using gradio_client — this gives far better audio
than running Chatterbox on CPU, and keeps torch off the hot path for TTS.

The Space accepts up to ~300 characters per call, so narration is generated
per segment (cover title_readout, then each shot) and any segment longer than
300 chars is split into sentence sub-chunks. Segments are concatenated with
short pauses, giving exact per-shot audio boundaries for assembly. Word-level
caption timestamps come from whisper-timestamped run locally on the result.
"""

import json
import logging
import os
import re

import numpy as np

logger = logging.getLogger("grimm.voice")

HF_SPACE = "ResembleAI/Chatterbox"
HF_TIMEOUT = 600            # patient — the Space may cold-start
HF_MAX_ATTEMPTS = 5
HF_RETRY_GAP = 30          # seconds between retries
MAX_CHARS = 300

COVER_MIN_DURATION = 3.0
PAUSE_AFTER_COVER = 0.8
PAUSE_BETWEEN_SHOTS = 0.3
TAIL_PAUSE = 0.4

# Storytelling narration parameters (from the brief).
EXAGGERATION = 0.4
CFG_WEIGHT = 0.6
TEMPERATURE = 0.7

WHISPER_MODEL = "base"
SAMPLE_RATE = 24000        # Chatterbox native output rate


def _resolve_voice_reference():
    reference = os.environ.get("VOICE_REFERENCE_PATH")
    if not reference:
        logger.info("No VOICE_REFERENCE_PATH set — using default Chatterbox voice")
        return None
    if not os.path.exists(reference):
        logger.warning(
            "VOICE_REFERENCE_PATH points to %s but the file does not exist — default voice", reference
        )
        return None
    logger.info("Using voice reference for cloning: %s", reference)
    return reference


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text, limit=MAX_CHARS):
    """Greedily pack sentences into <=limit-char chunks (hard-split if needed)."""
    chunks = []
    current = ""
    for sentence in _split_sentences(text):
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), limit):
                chunks.append(sentence[i:i + limit])
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


class _TTS:
    """Lazily-constructed Hugging Face Space client wrapper."""

    def __init__(self, reference):
        self.reference = reference
        self.client = None

    def connect(self):
        from gradio_client import Client

        token = os.environ.get("HUGGINGFACE_TOKEN") or None
        self.client = Client(HF_SPACE, hf_token=token, download_files=True)
        logger.info("Connected to Hugging Face Space %s", HF_SPACE)

    def synthesize(self, text):
        """Return a float32 mono numpy array at SAMPLE_RATE, or None on failure."""
        import soundfile as sf
        from gradio_client import handle_file

        if self.client is None:
            self.connect()

        audio_prompt = handle_file(self.reference) if self.reference else None
        last_error = None
        for attempt in range(1, HF_MAX_ATTEMPTS + 1):
            try:
                result = self.client.predict(
                    text,
                    audio_prompt,
                    EXAGGERATION,
                    TEMPERATURE,
                    0,            # seed (0 = random inside the Space)
                    CFG_WEIGHT,
                    api_name="/generate",
                )
                path = result[0] if isinstance(result, (list, tuple)) else result
                data, sr = sf.read(path, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != SAMPLE_RATE:
                    data = _resample(data, sr, SAMPLE_RATE)
                return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("HF TTS attempt %d/%d failed: %s", attempt, HF_MAX_ATTEMPTS, exc)
                if attempt < HF_MAX_ATTEMPTS:
                    import time
                    time.sleep(HF_RETRY_GAP)
        logger.error("HF TTS failed after %d attempts: %s", HF_MAX_ATTEMPTS, last_error)
        return None


def _resample(data, src_sr, dst_sr):
    if src_sr == dst_sr:
        return data
    duration = data.shape[0] / src_sr
    new_len = int(round(duration * dst_sr))
    old_x = np.linspace(0.0, duration, num=data.shape[0], endpoint=False)
    new_x = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(new_x, old_x, data).astype(np.float32)


def _transcribe_words(audio_path):
    try:
        import whisper_timestamped as whisper
    except Exception as exc:  # noqa: BLE001
        logger.error("whisper-timestamped is not importable: %s — captions skipped", exc)
        return []
    try:
        model = whisper.load_model(WHISPER_MODEL, device="cpu")
        result = whisper.transcribe(model, audio_path, language="en", verbose=None)
    except Exception as exc:  # noqa: BLE001
        logger.error("Whisper transcription failed: %s — captions skipped", exc)
        return []

    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = (word.get("text") or "").strip()
            if text:
                words.append({
                    "text": text,
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                })
    logger.info("Whisper produced %d word timestamps", len(words))
    return words


def generate_part_narration(part, part_index, working_dir="working"):
    """Generate audio + timestamps for one part.

    Returns {"audio_path","timestamps_path","segments","words","duration"} or
    None if TTS failed entirely (no audio means no video for this part).
    """
    audio_dir = os.path.join(working_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"part_{part_index}_narration.wav")
    timestamps_path = os.path.join(audio_dir, f"part_{part_index}_timestamps.json")

    try:
        import soundfile as sf
    except Exception as exc:  # noqa: BLE001
        logger.error("soundfile not importable: %s", exc)
        return None

    tts = _TTS(_resolve_voice_reference())

    texts = [("cover", None, part["cover"]["title_readout"])]
    for index, shot in enumerate(part["shots"]):
        texts.append(("shot", index, shot["narration"]))

    pieces = []
    segments = []
    cursor = 0.0

    for position, (kind, shot_index, text) in enumerate(texts):
        sub_audio = []
        for chunk in _chunk_text(text):
            audio = tts.synthesize(chunk)
            if audio is None:
                logger.error("TTS failed on %s chunk — aborting part %d", kind, part_index)
                return None
            sub_audio.append(audio)
        wav = np.concatenate(sub_audio) if len(sub_audio) > 1 else sub_audio[0]
        duration = len(wav) / SAMPLE_RATE

        if kind == "cover":
            pause = max(PAUSE_AFTER_COVER, COVER_MIN_DURATION - duration)
        elif position == len(texts) - 1:
            pause = TAIL_PAUSE
        else:
            pause = PAUSE_BETWEEN_SHOTS

        pieces.append(wav)
        pieces.append(np.zeros(int(round(pause * SAMPLE_RATE)), dtype=np.float32))
        segments.append({
            "kind": kind,
            "shot_index": shot_index,
            "start": round(cursor, 3),
            "end": round(cursor + duration, 3),
            "display_until": round(cursor + duration + pause, 3),
        })
        cursor += duration + pause
        logger.info(
            "TTS part %d %s%s: %.2fs (running %.1fs)",
            part_index, kind, f" {shot_index + 1}" if shot_index is not None else "", duration, cursor,
        )

    try:
        full = np.concatenate(pieces)
        sf.write(audio_path, full, SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not save narration wav: %s", exc)
        return None

    total_duration = cursor
    logger.info("Part %d narration complete: %.1fs", part_index, total_duration)

    words = _transcribe_words(audio_path)
    try:
        with open(timestamps_path, "w", encoding="utf-8") as handle:
            json.dump({"duration": total_duration, "segments": segments, "words": words}, handle, indent=2)
    except OSError as exc:
        logger.warning("Could not write timestamps json: %s", exc)

    return {
        "audio_path": audio_path,
        "timestamps_path": timestamps_path,
        "segments": segments,
        "words": words,
        "duration": total_duration,
    }
