"""Stage 4 — Voice.

Generate narration with Chatterbox TTS, running locally on the pipeline
runner's CPU — no external service, no GPU. Narration is generated per
segment (the cover title_readout, then each shot in order) and the clips
are concatenated with short pauses, giving exact per-shot audio boundaries
for assembly. Word-level caption timestamps come from whisper-timestamped
run locally on the result.

Heavy imports (torch, chatterbox, whisper) happen inside functions so the
rest of the pipeline can be imported and tested without them installed.
"""

import json
import logging
import os

logger = logging.getLogger("grimm.voice")

COVER_MIN_DURATION = 3.0   # cover card must hold at least this long
PAUSE_AFTER_COVER = 0.8
PAUSE_BETWEEN_SHOTS = 0.3
TAIL_PAUSE = 0.4

# Storytelling narration parameters (from the brief).
EXAGGERATION = 0.4
CFG_WEIGHT = 0.6
TEMPERATURE = 0.7

WHISPER_MODEL = "base"


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

    texts = [("cover", None, part["cover"]["title_readout"])]
    for index, shot in enumerate(part["shots"]):
        texts.append(("shot", index, shot["narration"]))

    pieces = []
    segments = []
    cursor = 0.0

    for position, (kind, shot_index, text) in enumerate(texts):
        try:
            kwargs = {
                "exaggeration": EXAGGERATION,
                "cfg_weight": CFG_WEIGHT,
                "temperature": TEMPERATURE,
            }
            if reference:
                kwargs["audio_prompt_path"] = reference
            wav = model.generate(text, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("Chatterbox failed on %s (%r…): %s", kind, text[:60], exc)
            return None

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        duration = wav.shape[-1] / sample_rate

        if kind == "cover":
            pause = max(PAUSE_AFTER_COVER, COVER_MIN_DURATION - duration)
        elif position == len(texts) - 1:
            pause = TAIL_PAUSE
        else:
            pause = PAUSE_BETWEEN_SHOTS

        silence = torch.zeros(1, int(round(pause * sample_rate)))
        pieces.extend([wav, silence])
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
        full = torch.cat(pieces, dim=1)
        torchaudio.save(audio_path, full, sample_rate)
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
