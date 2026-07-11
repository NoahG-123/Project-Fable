"""Stage 4 — Voice.

Generate the narration with Chatterbox TTS (free, open source, CPU) as one
continuous audio file, then run whisper-timestamped over it for word-level
timestamps used by the kinetic captions.

Narration order: cover title_readout (with a pause after it that guarantees
the cover card lasts at least 3 seconds), then every shot's narration in
order with a short natural pause between shots. Each segment's start/end is
recorded, so shot display windows come straight from the real audio — no
guessing at alignment.

Heavy imports (torch, chatterbox, whisper) happen inside functions so the
rest of the pipeline can be imported and tested without them installed.
"""

import json
import logging
import os

logger = logging.getLogger("grimm.voice")

COVER_MIN_DURATION = 3.0   # cover card must hold at least this long
PAUSE_AFTER_COVER = 0.6    # minimum breath after the title readout
PAUSE_BETWEEN_SHOTS = 0.15

# Energetic, clear storytelling voice, slightly faster than conversational —
# lower cfg_weight speeds up Chatterbox pacing, higher exaggeration adds energy.
CHATTERBOX_EXAGGERATION = 0.6
CHATTERBOX_CFG_WEIGHT = 0.35

WHISPER_MODEL = "base"


def _resolve_voice_reference():
    reference = os.environ.get("CHATTERBOX_VOICE_REFERENCE")
    if not reference:
        logger.info("No CHATTERBOX_VOICE_REFERENCE set — using default Chatterbox voice")
        return None
    if not os.path.exists(reference):
        logger.warning(
            "CHATTERBOX_VOICE_REFERENCE points to %s but the file does not exist — "
            "continuing with the default voice",
            reference,
        )
        return None
    logger.info("Using voice reference for cloning: %s", reference)
    return reference


def _transcribe_words(audio_path):
    """Word-level timestamps via whisper-timestamped. Returns a list (may be empty)."""
    try:
        import whisper_timestamped as whisper
    except Exception as exc:  # noqa: BLE001
        logger.error("whisper-timestamped is not importable: %s — captions will be skipped", exc)
        return []

    try:
        model = whisper.load_model(WHISPER_MODEL, device="cpu")
        result = whisper.transcribe(model, audio_path, language="en", verbose=None)
    except Exception as exc:  # noqa: BLE001
        logger.error("Whisper transcription failed: %s — captions will be skipped", exc)
        return []

    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = (word.get("text") or "").strip()
            if not text:
                continue
            words.append(
                {
                    "text": text,
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                }
            )
    logger.info("Whisper produced %d word timestamps", len(words))
    return words


def generate_narration(script, working_dir="working"):
    """Generate narration.wav and timestamps.json.

    Returns {"audio_path", "timestamps_path", "segments", "words", "duration"}
    or None if TTS failed entirely (no audio means no video, so this is fatal).

    segments is a list of dicts in playback order:
      {"kind": "cover"|"shot", "shot_index": int|None,
       "start": float, "end": float, "display_until": float}
    where display_until includes the trailing pause — consecutive segments'
    display windows tile the full audio with no gaps.
    """
    audio_dir = os.path.join(working_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, "narration.wav")
    timestamps_path = os.path.join(audio_dir, "timestamps.json")

    try:
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not import Chatterbox TTS stack: %s", exc)
        return None

    try:
        logger.info("Loading Chatterbox TTS model (CPU)…")
        model = ChatterboxTTS.from_pretrained(device="cpu")
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load Chatterbox TTS model: %s", exc)
        return None

    sample_rate = model.sr
    reference = _resolve_voice_reference()

    texts = [("cover", None, script["cover"]["title_readout"])]
    for index, shot in enumerate(script["shots"]):
        texts.append(("shot", index, shot["narration"]))

    pieces = []
    segments = []
    cursor = 0.0

    for position, (kind, shot_index, text) in enumerate(texts):
        try:
            kwargs = {
                "exaggeration": CHATTERBOX_EXAGGERATION,
                "cfg_weight": CHATTERBOX_CFG_WEIGHT,
            }
            if reference:
                kwargs["audio_prompt_path"] = reference
            wav = model.generate(text, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Chatterbox failed on %s (%r…): %s", kind, text[:60], exc
            )
            return None

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        duration = wav.shape[-1] / sample_rate

        if kind == "cover":
            pause = max(PAUSE_AFTER_COVER, COVER_MIN_DURATION - duration)
        elif position == len(texts) - 1:
            pause = 0.4  # small tail so the video doesn't cut dead on the last word
        else:
            pause = PAUSE_BETWEEN_SHOTS

        silence = torch.zeros(1, int(round(pause * sample_rate)))
        pieces.extend([wav, silence])
        segments.append(
            {
                "kind": kind,
                "shot_index": shot_index,
                "start": round(cursor, 3),
                "end": round(cursor + duration, 3),
                "display_until": round(cursor + duration + pause, 3),
            }
        )
        cursor += duration + pause
        logger.info(
            "TTS %s%s: %.2fs (running total %.1fs)",
            kind,
            f" {shot_index + 1}" if shot_index is not None else "",
            duration,
            cursor,
        )

    try:
        full = torch.cat(pieces, dim=1)
        torchaudio.save(audio_path, full, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not save narration.wav: %s", exc)
        return None

    total_duration = cursor
    logger.info("Narration complete: %.1fs total", total_duration)

    words = _transcribe_words(audio_path)

    try:
        with open(timestamps_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"duration": total_duration, "segments": segments, "words": words},
                handle,
                indent=2,
            )
    except OSError as exc:
        logger.warning("Could not write timestamps.json: %s", exc)

    return {
        "audio_path": audio_path,
        "timestamps_path": timestamps_path,
        "segments": segments,
        "words": words,
        "duration": total_duration,
    }
