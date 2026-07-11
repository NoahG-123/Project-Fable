"""Stage 7 — Distribute.

The artifacts themselves are uploaded by the GitHub Actions workflow. This
stage writes the run summary: a markdown report to the debug trail and to
$GITHUB_STEP_SUMMARY so it appears on the workflow run page.
"""

import logging
import os

logger = logging.getLogger("grimm.distribute")


def _build_summary(story, script, assembly_info, gate_results, warnings):
    total_words = sum(len(shot["narration"].split()) for shot in script["shots"])
    passed = sum(1 for _, ok, _ in gate_results if ok)

    lines = [
        "# 🎬 Daily Fairy Tale Reel",
        "",
        f"**{script['title']}** — {script['author']}",
        "",
        "| | |",
        "|---|---|",
        f"| Story length | {story.get('estimated_length', '?')} |",
        f"| Total shots | {len(script['shots'])} |",
        f"| Video duration | {assembly_info['duration']:.1f}s |",
        f"| Narration word count | {total_words} |",
        f"| Quality gates | {passed}/{len(gate_results)} passed |",
        "",
        "## Quality gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in gate_results:
        lines.append(f"| {name} | {'✅ pass' if ok else '❌ fail'} | {detail} |")

    if warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- ⚠️ {warning}" for warning in warnings]

    lines += [
        "",
        "_Download the `reel-*` artifact from this run and post it to Instagram._",
        "",
    ]
    return "\n".join(lines)


def write_run_summary(story, script, assembly_info, gate_results, warnings, working_dir="working"):
    """Write the run summary. Never raises."""
    try:
        summary = _build_summary(story, script, assembly_info, gate_results, warnings)
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
