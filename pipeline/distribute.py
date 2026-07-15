"""Stage 7 — Distribute (V2).

Artifacts are uploaded by the GitHub Actions workflow. This stage writes the
run summary — markdown to the debug trail and to $GITHUB_STEP_SUMMARY — across
all parts produced.
"""

import logging
import os

logger = logging.getLogger("grimm.distribute")


def _build_summary(story, script, part_results, warnings):
    title = script.get("title", story.get("title", "?"))
    author = script.get("author", story.get("author", "?"))

    lines = [
        "# 🎬 Daily Fairy Tale Reel",
        "",
        f"**{title}** — {author}",
        "",
        f"- Story length: {story.get('estimated_length', '?')}",
        f"- Parts produced: {len(part_results)} of {story.get('parts', 1)}",
        "",
    ]

    for entry in part_results:
        pi = entry["part_index"]
        assembly = entry["assembly"]
        gates = entry["gates"]
        words = entry["word_count"]
        passed = sum(1 for _, ok, _ in gates if ok)
        lines += [
            f"## Part {pi}",
            "",
            f"- Duration: {assembly['duration']:.1f}s",
            f"- Shots: {assembly['shots_rendered']}",
            f"- Narration words: {words}",
            f"- Quality gates: {passed}/{len(gates)} passed",
            "",
            "| Gate | Result | Detail |",
            "|---|---|---|",
        ]
        for name, ok, detail in gates:
            lines.append(f"| {name} | {'✅' if ok else '❌'} | {detail} |")
        lines.append("")

    if warnings:
        lines += ["## Warnings", ""]
        lines += [f"- ⚠️ {w}" for w in warnings]
        lines.append("")

    lines += ["_Download the `reels-*` artifact and post each part to Instagram in order._", ""]
    return "\n".join(lines)


def write_run_summary(story, script, part_results, warnings, working_dir="working"):
    """Write the run summary. Never raises."""
    try:
        summary = _build_summary(story, script, part_results, warnings)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not build run summary: %s", exc)
        return

    try:
        debug_dir = os.path.join(working_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "run_summary.md"), "w", encoding="utf-8") as handle:
            handle.write(summary)
    except OSError as exc:
        logger.warning("Could not write run_summary.md: %s", exc)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as handle:
                handle.write(summary + "\n")
            logger.info("Run summary written to GitHub job summary")
        except OSError as exc:
            logger.warning("Could not write to GITHUB_STEP_SUMMARY: %s", exc)
    else:
        logger.info("GITHUB_STEP_SUMMARY not set — summary kept in debug trail only")
