#!/usr/bin/env python3
"""Grimm pipeline orchestrator.

Runs every stage in sequence and owns all failure handling. This process
never lets an unhandled exception escape: every outcome is logged, the debug
trail is always written, and the exit code is the only signal to CI.

Exit codes:
  0 — video produced (the run is a success if and only if a video exists),
      or the story queue is empty (graceful no-op).
  1 — no video produced.
"""

import logging
import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORKING_DIR = "working"

logger = logging.getLogger("grimm.main")


def prepare_working_dir():
    """Clean and recreate the working tree. Files never accumulate across runs."""
    shutil.rmtree(WORKING_DIR, ignore_errors=True)
    for sub in ("images", "audio", "output", "debug"):
        os.makedirs(os.path.join(WORKING_DIR, sub), exist_ok=True)


def setup_logging():
    """Log to stdout and to working/debug/run_log.txt."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    try:
        file_handler = logging.FileHandler(
            os.path.join(WORKING_DIR, "debug", "run_log.txt"), encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not attach run_log.txt handler: %s", exc)


def run():
    """Execute the pipeline. Returns the process exit code."""
    import ingest
    import enrich
    import assets
    import voice
    import assemble
    import validate
    import distribute

    warnings = []

    # Stage 1 — Ingest
    logger.info("=== STAGE 1: INGEST ===")
    story = ingest.pick_next_story()
    if story is None:
        logger.warning("No story to process — exiting gracefully.")
        return 0

    # Stage 2 — Enrich
    logger.info("=== STAGE 2: ENRICH ===")
    script = enrich.generate_script(story, WORKING_DIR)
    if script is None:
        logger.error("No valid script — skipping this run. See working/debug/raw_response.txt")
        return 1

    # Stage 3 — Assets
    logger.info("=== STAGE 3: ASSETS ===")
    images = assets.fetch_all_images(script, WORKING_DIR)
    if images["placeholders"]:
        warnings.append(f"{images['placeholders']} image(s) fell back to a placeholder")

    # Stage 4 — Voice
    logger.info("=== STAGE 4: VOICE ===")
    voice_info = voice.generate_narration(script, WORKING_DIR)
    if voice_info is None:
        logger.error("Narration generation failed — no audio means no video.")
        return 1
    if not voice_info["words"]:
        warnings.append("Whisper produced no word timestamps — captions are missing")

    # Stage 5 — Assemble
    logger.info("=== STAGE 5: ASSEMBLE ===")
    assembly_info = assemble.assemble_video(script, images, voice_info, WORKING_DIR)
    if assembly_info is None:
        logger.error("Assembly failed — no video produced.")
        return 1

    # Stage 6 — Validate
    logger.info("=== STAGE 6: VALIDATE ===")
    try:
        gate_results = validate.run_quality_gates(
            script, images, voice_info, assembly_info, WORKING_DIR
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Quality gates crashed (%s) — video still produced, continuing", exc)
        gate_results = [("Quality gates executed", False, str(exc))]
    failed_gates = [name for name, ok, _ in gate_results if not ok]
    if failed_gates:
        warnings.append(
            "Quality gates failed (inspect before posting): " + ", ".join(failed_gates)
        )

    # Stage 7 — Distribute
    logger.info("=== STAGE 7: DISTRIBUTE ===")
    distribute.write_run_summary(
        story, script, assembly_info, gate_results, warnings, WORKING_DIR
    )

    for warning in warnings:
        logger.warning("RUN WARNING: %s", warning)
    logger.info(
        "Run complete: %s — %s (%.1fs video)",
        script["title"], script["author"], assembly_info["duration"],
    )
    return 0


def main():
    exit_code = 1
    try:
        prepare_working_dir()
        setup_logging()
        exit_code = run()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — nothing may escape main.py unhandled
        details = traceback.format_exc()
        print("FATAL: unhandled exception in pipeline:\n" + details, file=sys.stderr)
        try:
            os.makedirs(os.path.join(WORKING_DIR, "debug"), exist_ok=True)
            with open(
                os.path.join(WORKING_DIR, "debug", "run_log.txt"), "a", encoding="utf-8"
            ) as handle:
                handle.write("\nFATAL unhandled exception:\n" + details + "\n")
        except OSError:
            pass
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
