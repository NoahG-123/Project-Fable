#!/usr/bin/env python3
"""Grimm pipeline orchestrator (V2).

Runs every stage in sequence and owns all failure handling. Never lets an
unhandled exception escape. A story may produce multiple parts; each part is
generated, voiced, assembled and validated independently.

Exit codes:
  0 — at least one part video was produced, or the queue is empty (graceful).
  1 — no video was produced at all.

The story is marked done in stories.json only if EVERY part was produced.
Partial success uploads what exists but leaves the story pending for retry.
"""

import logging
import os
import random
import shutil
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORKING_DIR = "working"

logger = logging.getLogger("grimm.main")


def prepare_working_dir():
    shutil.rmtree(WORKING_DIR, ignore_errors=True)
    for sub in ("images", "audio", "output", "debug"):
        os.makedirs(os.path.join(WORKING_DIR, sub), exist_ok=True)


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    try:
        file_handler = logging.FileHandler(os.path.join(WORKING_DIR, "debug", "run_log.txt"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not attach run_log.txt handler: %s", exc)


def run():
    import ingest
    import enrich
    import assets
    import voice
    import assemble
    import validate
    import distribute

    warnings = []

    # Stage 1 — Ingest (peek only; marked done later if all parts succeed)
    logger.info("=== STAGE 1: INGEST ===")
    story = ingest.pick_next_story()
    if story is None:
        logger.warning("No story to process — exiting gracefully.")
        return 0

    # Stage 2 — Enrich (whole multi-part script in one call)
    logger.info("=== STAGE 2: ENRICH ===")
    script = enrich.generate_script(story, WORKING_DIR)
    if script is None:
        logger.error("No valid script — skipping this run. See working/debug/raw_response.txt")
        return 1

    parts = script.get("scripts", [])
    expected_parts = story.get("parts", 1)
    run_seed = random.randint(0, 2**31 - 1)
    style_ref = assets.style_reference_url()
    logger.info("Run seed %d | style reference: %s", run_seed, style_ref or "<none>")

    part_results = []

    for part_index, part in enumerate(parts, start=1):
        logger.info("========== PART %d of %d ==========", part_index, len(parts))

        # Stage 3 — Assets
        logger.info("=== STAGE 3: ASSETS (part %d) ===", part_index)
        try:
            images = assets.generate_part_images(part, part_index, run_seed, style_ref, WORKING_DIR)
        except Exception as exc:  # noqa: BLE001
            logger.error("Asset generation crashed for part %d: %s — skipping part", part_index, exc)
            continue
        if images["placeholders"]:
            warnings.append(f"Part {part_index}: {images['placeholders']} image(s) fell back to a placeholder")

        # Stage 4 — Voice
        logger.info("=== STAGE 4: VOICE (part %d) ===", part_index)
        voice_info = voice.generate_part_narration(part, part_index, WORKING_DIR)
        if voice_info is None:
            logger.error("Narration failed for part %d — skipping part", part_index)
            warnings.append(f"Part {part_index}: narration generation failed — part skipped")
            continue
        if not voice_info["words"]:
            warnings.append(f"Part {part_index}: no word timestamps — captions missing")

        # Stage 5 — Assemble
        logger.info("=== STAGE 5: ASSEMBLE (part %d) ===", part_index)
        assembly_info = assemble.assemble_part(part, images, voice_info, part_index, WORKING_DIR)
        if assembly_info is None:
            logger.error("Assembly failed for part %d — skipping part", part_index)
            warnings.append(f"Part {part_index}: assembly failed — part skipped")
            continue

        # Stage 6 — Validate
        logger.info("=== STAGE 6: VALIDATE (part %d) ===", part_index)
        try:
            gates = validate.run_quality_gates(part, images, voice_info, assembly_info, part_index, WORKING_DIR)
        except Exception as exc:  # noqa: BLE001
            logger.error("Quality gates crashed for part %d (%s) — video still produced", part_index, exc)
            gates = [("Quality gates executed", False, str(exc))]
        failed = [name for name, ok, _ in gates if not ok]
        if failed:
            warnings.append(f"Part {part_index} gates failed (inspect before posting): {', '.join(failed)}")

        part_results.append({
            "part_index": part_index,
            "assembly": assembly_info,
            "gates": gates,
            "word_count": sum(len(s["narration"].split()) for s in part["shots"]),
        })

    if not part_results:
        logger.error("No parts produced a video — run failed.")
        return 1

    # Stage 7 — Distribute
    logger.info("=== STAGE 7: DISTRIBUTE ===")
    distribute.write_run_summary(story, script, part_results, warnings, WORKING_DIR)

    if len(part_results) == len(parts) == expected_parts:
        ingest.mark_story_done(story["title"], story["author"])
    else:
        warnings.append(
            f"Only {len(part_results)}/{expected_parts} parts produced — story left pending for retry"
        )
        logger.warning("Story NOT marked done — %d of %d parts produced", len(part_results), expected_parts)

    for warning in warnings:
        logger.warning("RUN WARNING: %s", warning)
    logger.info("Run complete: %s — %s (%d part video(s) produced)",
                script.get("title"), script.get("author"), len(part_results))
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
            with open(os.path.join(WORKING_DIR, "debug", "run_log.txt"), "a", encoding="utf-8") as handle:
                handle.write("\nFATAL unhandled exception:\n" + details + "\n")
        except OSError:
            pass
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
