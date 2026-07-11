"""Stage 1 — Ingest.

Read stories.json from the repo root, pick the first pending story, mark it
done and write the file back. The GitHub Actions workflow commits the updated
stories.json at the end of the run, so the queue advances automatically.
"""

import copy
import json
import logging
import os

logger = logging.getLogger("grimm.ingest")

STORIES_PATH = "stories.json"

VALID_LENGTHS = ("short", "medium", "long")


def pick_next_story(stories_path=STORIES_PATH):
    """Return the first pending story (marking it done on disk), or None.

    Returns a copy of the story dict with its original "pending" status so
    downstream stages see the story as selected, while the file on disk has
    the entry marked "done".
    """
    try:
        with open(stories_path, "r", encoding="utf-8") as handle:
            stories = json.load(handle)
    except FileNotFoundError:
        logger.error("stories.json not found at %s", os.path.abspath(stories_path))
        return None
    except json.JSONDecodeError as exc:
        logger.error("stories.json is not valid JSON: %s", exc)
        return None
    except OSError as exc:
        logger.error("Could not read stories.json: %s", exc)
        return None

    if not isinstance(stories, list):
        logger.error("stories.json must contain a JSON list, got %s", type(stories).__name__)
        return None

    for entry in stories:
        if not isinstance(entry, dict) or entry.get("status") != "pending":
            continue

        missing = [key for key in ("title", "author", "moral") if not entry.get(key)]
        if missing:
            logger.warning(
                "Skipping malformed story entry (missing %s): %r", ", ".join(missing), entry
            )
            continue

        selected = copy.deepcopy(entry)
        if selected.get("estimated_length") not in VALID_LENGTHS:
            logger.warning(
                "Story %r has invalid estimated_length %r — defaulting to 'medium'",
                selected.get("title"),
                selected.get("estimated_length"),
            )
            selected["estimated_length"] = "medium"

        entry["status"] = "done"
        try:
            with open(stories_path, "w", encoding="utf-8") as handle:
                json.dump(stories, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            logger.error("Could not write updated stories.json: %s", exc)
            return None

        logger.info(
            "Selected story: %s — %s (%s)",
            selected["title"],
            selected["author"],
            selected["estimated_length"],
        )
        return selected

    logger.warning("No pending stories remain in stories.json — nothing to do.")
    return None
