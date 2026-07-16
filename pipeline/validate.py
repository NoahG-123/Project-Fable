"""Stage 6 — Validate (V2).

Quality gates run per part after assembly. A failed check never crashes the
run — it is logged and written to working/debug/quality_report.txt so the
human can inspect the artifact before posting.
"""

import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger("grimm.validate")

MIN_DURATION = 40.0
MAX_DURATION = 100.0
EXPECTED_RESOLUTION = (1080, 1920)
MIN_CAPTION_COVERAGE = 0.85
MIN_FILE_SIZE_BYTES = 500 * 1024
SILENCE_RMS_THRESHOLD = 1e-4


def _probe_video(path):
    try:
        from moviepy.editor import VideoFileClip

        with VideoFileClip(path) as clip:
            return float(clip.duration), tuple(clip.size)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not probe video %s: %s", path, exc)
        return None, None


def _file_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio_rms(path):
    try:
        import soundfile as sf

        data, _ = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(data ** 2)))
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not read narration audio %s: %s", path, exc)
        return None


def run_quality_gates(script_part, images, voice_info, assembly_info, part_index, working_dir="working"):
    """Run every gate for one part. Returns an ordered list of (name, passed, detail)."""
    results = []

    def gate(name, passed, detail):
        results.append((name, bool(passed), detail))
        logger.info("GATE p%d %-36s %s — %s", part_index, name, "PASS" if passed else "FAIL", detail)

    output_path = assembly_info["output_path"]
    duration, resolution = _probe_video(output_path)

    gate("Duration 40-100s", duration is not None and MIN_DURATION <= duration <= MAX_DURATION,
         f"duration = {duration:.1f}s" if duration is not None else "could not probe video")
    gate("Resolution exactly 1080x1920", resolution == EXPECTED_RESOLUTION, f"resolution = {resolution}")
    gate("All part shots rendered", assembly_info["shots_rendered"] == len(script_part["shots"]),
         f"{assembly_info['shots_rendered']} of {len(script_part['shots'])} rendered")

    try:
        ordered = [images["cover"]] + list(images["shots"])
        hashes = [_file_md5(p) for p in ordered]
        dupes = [os.path.basename(ordered[i]) for i in range(1, len(hashes)) if hashes[i] == hashes[i - 1]]
        gate("No two consecutive identical images", not dupes,
             "all consecutive images differ" if not dupes else f"duplicates: {', '.join(dupes)}")
    except Exception as exc:  # noqa: BLE001
        gate("No two consecutive identical images", False, f"could not hash images: {exc}")

    handle_set = bool(os.environ.get("CHANNEL_HANDLE", "").strip())
    gate("Watermark present", handle_set,
         "CHANNEL_HANDLE set" if handle_set else "CHANNEL_HANDLE not set")

    rms = _audio_rms(voice_info["audio_path"])
    gate("Audio non-silent", rms is not None and rms > SILENCE_RMS_THRESHOLD,
         f"RMS = {rms:.6f}" if rms is not None else "audio unreadable")

    narration_window = max(assembly_info["duration"] - assembly_info["cover_duration"], 0.001)
    coverage = assembly_info["caption_seconds"] / narration_window
    gate("Caption coverage >= 85%", coverage >= MIN_CAPTION_COVERAGE,
         f"coverage = {coverage:.0%} of {narration_window:.1f}s")

    try:
        size = os.path.getsize(output_path)
    except OSError:
        size = 0
    gate("File size > 500KB", size > MIN_FILE_SIZE_BYTES, f"size = {size / 1024:.0f}KB")

    _append_report(results, part_index, assembly_info, working_dir)
    return results


def _append_report(results, part_index, assembly_info, working_dir):
    try:
        debug_dir = os.path.join(working_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, "quality_report.txt")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"\nPART {part_index} — {os.path.basename(assembly_info['output_path'])}\n")
            handle.write("=" * 60 + "\n")
            for name, passed, detail in results:
                handle.write(f"[{'PASS' if passed else 'FAIL'}] {name}\n        {detail}\n")
            failed = sum(1 for _, ok, _ in results if not ok)
            handle.write(f"{len(results) - failed}/{len(results)} gates passed\n")
    except OSError as exc:
        logger.warning("Could not write quality report: %s", exc)
