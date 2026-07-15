"""Stage 1 — Ingest (V2).

Read stories.json from the repo root and pick the first pending story. In V2
the story is NOT marked done at selection time — it is only marked done once
every part has been successfully produced (see main.py). The GitHub Actions
workflow commits the updated stories.json at the end of the run.
"""

import copy
import json
import logging

logger = logging.getLogger("grimm.ingest")

STORIES_PATH = "stories.json"

VALID_LENGTHS = ("short", "medium", "long")


def _load(stories_path):
    try:
        with open(stories_path, "r", encoding="utf-8") as handle:
            stories = json.load(handle)
    except FileNotFoundError:
        logger.error("stories.json not found at %s", stories_path)
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
    return stories


def pick_next_story(stories_path=STORIES_PATH):
    """Return the first pending story (peek only — does NOT write), or None.

    Adds a normalised 'parts' count and validated 'estimated_length' to the
    returned copy.
    """
    stories = _load(stories_path)
    if stories is None:
        return None

    for entry in stories:
        if not isinstance(entry, dict) or entry.get("status") != "pending":
            continue

        missing = [key for key in ("title", "author", "moral") if not entry.get(key)]
        if missing:
            logger.warning("Skipping malformed story entry (missing %s): %r", ", ".join(missing), entry)
            continue

        selected = copy.deepcopy(entry)

        if selected.get("estimated_length") not in VALID_LENGTHS:
            logger.warning(
                "Story %r has invalid estimated_length %r — defaulting to 'medium'",
                selected.get("title"), selected.get("estimated_length"),
            )
            selected["estimated_length"] = "medium"

        parts = selected.get("parts", 1)
        if not isinstance(parts, int) or parts < 1 or parts > 3:
            logger.warning(
                "Story %r has invalid parts %r — defaulting to 1", selected.get("title"), parts
            )
            parts = 1
        selected["parts"] = parts

        logger.info(
            "Selected story: %s — %s (%s, %d part%s)",
            selected["title"], selected["author"], selected["estimated_length"],
            parts, "" if parts == 1 else "s",
        )
        return selected

    logger.warning("No pending stories remain in stories.json — nothing to do.")
    return None


def mark_story_done(title, author, stories_path=STORIES_PATH):
    """Mark the matching story done and write stories.json back. Returns bool."""
    stories = _load(stories_path)
    if stories is None:
        return False

    for entry in stories:
        if (
            isinstance(entry, dict)
            and entry.get("title") == title
            and entry.get("author") == author
            and entry.get("status") == "pending"
        ):
            entry["status"] = "done"
            try:
                with open(stories_path, "w", encoding="utf-8") as handle:
                    json.dump(stories, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
            except OSError as exc:
                logger.error("Could not write updated stories.json: %s", exc)
                return False
            logger.info("Marked story done: %s — %s", title, author)
            return True

    logger.warning("Could not find pending story to mark done: %s — %s", title, author)
    return False
