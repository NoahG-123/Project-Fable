"""Stage 2 — Enrich.

Call DeepSeek via OpenRouter with the master prompt, then parse and validate
the JSON script. This is the most critical stage: if the script is malformed
or incomplete the run is skipped, never continued with a partial script.

Parse chain on failure: strip markdown fences -> json.loads -> balanced-brace
extraction of the first {...} block -> one correction message to the model ->
give up. The full raw response is appended to working/debug/raw_response.txt
on every call.
"""

import json
import logging
import os
import re
import time

import requests

logger = logging.getLogger("grimm.enrich")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "deepseek/deepseek-r1"
FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

SLEEP_BETWEEN_CALLS = 4          # seconds between any sequential OpenRouter calls
RATE_LIMIT_BACKOFFS = [6, 12, 24, 48]  # exponential backoff on 429
REQUEST_TIMEOUT = 300            # deepseek-r1 reasoning can be slow

VISUAL_PROMPT_PREFIX = "stickman illustration"

SHOT_RANGES = {"short": (18, 20), "medium": (24, 26), "long": (28, 30)}
WORD_RANGES = {"short": (80, 100), "medium": (110, 130), "long": (130, 160)}
WORD_COUNT_TOLERANCE = 0.25  # hard-fail only outside +/-25% of the strict range

SYSTEM_PROMPT = """You are a script writer for a short-form video channel that retells classic fairy tales and fables in the style of educational infographic channels. Your scripts are written in second person — the viewer is the main character. You narrate directly to them as if telling them their own story.

Your voice is conversational, confident and warm. You talk like a knowledgeable friend who has read everything and finds the story genuinely interesting. You never mock the story or its characters. The humour comes entirely from the natural absurdity of the situations, delivered with dry understatement and perfect timing. You let ridiculous moments land on their own — you do not point at them and say they are funny.

Your sentences vary in length. Short sentences land punches. Longer sentences carry context and momentum. You use present tense throughout. You use light asides and parenthetical observations occasionally — small human moments that make the narration feel alive rather than scripted.

You write for a stickman animation style. Every shot in the video is a single frozen stickman illustration. Your visual prompts describe one frozen moment — like briefing an illustrator on a single panel. They are never sequences or actions in progress.

Your output must always be valid JSON. No markdown. No preamble. No explanation. Only the JSON object."""

USER_PROMPT_TEMPLATE = """Retell the following story as a script for a 60-second Instagram Reel:

Title: {title}
Author: {author}
Moral: {moral}
Estimated Length: {estimated_length}

---

VOICE AND TONE RULES — follow these exactly:

- Write in second person throughout. The viewer is the main character. Use "you" and "your" — never "he", "she" or "they" for the protagonist
- Write in present tense throughout
- Narrate like a knowledgeable friend telling you your own story — confident, warm, conversational
- The very first shot of the story drops the viewer immediately into the world. No preamble. No wind-up. Just start. Like: "You are born the youngest of three brothers in a small woodcutter's family" or "You are a tortoise and a hare has just challenged you to a race and everyone thinks this is hilarious"
- Humour comes from understatement and the natural absurdity of situations. Never write a joke. Never point at something and call it funny. Let the moment land on its own
- Use short sentences when action happens. Use longer sentences to carry context and build momentum
- Occasionally use light asides — small parenthetical observations that feel human and unscripted. Do not overuse them
- Never use crude humour, mockery, sarcasm or incomplete sentences
- The ending states the outcome and moral matter-of-factly — no grand sweeping conclusion, just the natural landing of the story

---

STRUCTURE RULES:

The script has two parts: cover and shots.

Cover:
One shot. No narration except the title and author read aloud, exactly as written. The visual must be the single most dramatic or visually interesting moment of the entire story. It is the thumbnail — it must make someone stop scrolling. Bold, dynamic, immediately readable.

Shots:
The full story told shot by shot. Shot 1 opens immediately — no hook, no preamble, straight into the world. Each shot is one frozen visual moment. Narration per shot is one to two sentences — short, punchy, present tense. The story must flow naturally from shot to shot with clear momentum. Never linger. Never repeat. Keep moving.

The final two shots are the ending. State the outcome plainly in the second to last shot. Land the moral naturally in the final shot — not as a lesson being taught, just as the obvious conclusion any reasonable person would draw. Matter-of-fact. No fanfare.

Shot count by estimated length:
- short: 18 to 20 shots total
- medium: 24 to 26 shots total
- long: 28 to 30 shots total

---

VISUAL PROMPT RULES:

- Every visual prompt describes one single frozen stickman moment — like a storyboard panel
- Always begin with "stickman illustration —"
- White background, bold black lines, minimal detail, simple and clear
- Describe character expressions, positioning and key objects
- The cover visual prompt must feel dynamic and dramatic — the most arresting image of the whole story
- Never describe motion, sequences or actions in progress — only the frozen moment

---

WORD COUNT:
- short stories: 80 to 100 words total narration across all shots including ending
- medium stories: 110 to 130 words total narration across all shots including ending
- long stories: 130 to 160 words total narration across all shots including ending
- The cover has no narration word count — only the title and author read aloud

---

Output this exact JSON structure:

{{
  "title": "string",
  "author": "string",
  "cover": {{
    "title_readout": "string — exactly '[Title] — [Author]'",
    "visual_prompt": "string"
  }},
  "shots": [
    {{
      "shot_number": 1,
      "narration": "string",
      "visual_prompt": "string"
    }}
  ]
}}

The final two entries in the shots array are the ending. Do not add a separate ending key. Do not include any text outside the JSON object."""

CORRECTION_MESSAGE = (
    "Your previous response was not valid JSON. Respond again with ONLY the valid "
    "JSON object in the exact structure requested — no markdown fences, no preamble, "
    "no commentary, nothing outside the JSON object."
)


def _log_raw_response(text, debug_dir, label):
    """Append the full raw response to the debug trail. Never raises."""
    try:
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, "raw_response.txt")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"\n{'=' * 70}\n[{label}]\n{'=' * 70}\n")
            handle.write(text if text is not None else "<no response>")
            handle.write("\n")
    except OSError as exc:
        logger.warning("Could not write raw response to debug trail: %s", exc)


def _call_openrouter(messages, model, api_key, debug_dir, label):
    """Make one chat-completions call. Returns the content string or None.

    Handles 429 with exponential backoff (6s, 12s, 24s, 48s) — never retries
    immediately. Other transient errors get one retry per backoff slot too.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "grimm-pipeline",
        "X-Title": "GrimmPipeline",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    attempts = 1 + len(RATE_LIMIT_BACKOFFS)
    for attempt in range(attempts):
        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            logger.warning("[%s] OpenRouter request failed (attempt %d): %s", label, attempt + 1, exc)
            _log_raw_response(f"REQUEST EXCEPTION: {exc}", debug_dir, f"{label} attempt {attempt + 1}")
            if attempt < len(RATE_LIMIT_BACKOFFS):
                time.sleep(RATE_LIMIT_BACKOFFS[attempt])
                continue
            return None

        _log_raw_response(
            f"HTTP {response.status_code}\n{response.text}",
            debug_dir,
            f"{label} attempt {attempt + 1} (model={model})",
        )

        if response.status_code == 429:
            if attempt < len(RATE_LIMIT_BACKOFFS):
                wait = RATE_LIMIT_BACKOFFS[attempt]
                logger.warning("[%s] 429 from OpenRouter — backing off %ds", label, wait)
                time.sleep(wait)
                continue
            logger.error("[%s] 429 from OpenRouter after all backoffs — giving up", label)
            return None

        if response.status_code != 200:
            logger.error(
                "[%s] OpenRouter returned HTTP %d: %s",
                label, response.status_code, response.text[:500],
            )
            if attempt < len(RATE_LIMIT_BACKOFFS) and response.status_code >= 500:
                time.sleep(RATE_LIMIT_BACKOFFS[attempt])
                continue
            return None

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.error("[%s] Could not extract content from OpenRouter response: %s", label, exc)
            return None

        if not content or not content.strip():
            logger.error("[%s] OpenRouter returned an empty content string", label)
            return None
        return content

    return None


def _strip_markdown_fences(text):
    """Remove ```json ... ``` (or bare ```) fences wrapping the payload."""
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z]*\s*\n?", "", stripped)
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


def _extract_first_json_object(text):
    """Scan for the first balanced {...} block that parses as JSON, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    return None


def _parse_script(raw_content):
    """Parse an LLM response into a dict, or None if unrecoverable."""
    if not raw_content:
        return None

    cleaned = _strip_markdown_fences(raw_content)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        logger.warning("Parsed JSON is a %s, not an object", type(parsed).__name__)
    except json.JSONDecodeError as exc:
        logger.warning("Direct JSON parse failed: %s — trying balanced-brace extraction", exc)

    extracted = _extract_first_json_object(cleaned)
    if isinstance(extracted, dict):
        logger.info("Recovered JSON object via balanced-brace extraction")
        return extracted
    return None


def _repair_visual_prompts(script):
    """Ensure every visual_prompt starts with the stickman prefix.

    The stickman style is the channel's entire identity, so a missing prefix
    is repaired (and logged) rather than failing the whole run over it.
    """
    def repair(prompt, where):
        if not isinstance(prompt, str):
            return prompt
        if prompt.strip().lower().startswith(VISUAL_PROMPT_PREFIX):
            return prompt.strip()
        logger.warning("Repairing %s visual_prompt missing stickman prefix", where)
        return f"stickman illustration — {prompt.strip()}"

    cover = script.get("cover")
    if isinstance(cover, dict) and cover.get("visual_prompt"):
        cover["visual_prompt"] = repair(cover["visual_prompt"], "cover")
    for shot in script.get("shots") or []:
        if isinstance(shot, dict) and shot.get("visual_prompt"):
            shot["visual_prompt"] = repair(
                shot["visual_prompt"], f"shot {shot.get('shot_number', '?')}"
            )
    return script


def validate_script(script, story):
    """Validate the parsed script against every requirement in the brief.

    Returns a list of error strings — empty means the script is usable.
    Every failed check is reported, not just the first.
    """
    errors = []

    if not isinstance(script, dict):
        return ["script is not a JSON object"]

    if not (isinstance(script.get("title"), str) and script["title"].strip()):
        errors.append("'title' is missing or empty")
    if not (isinstance(script.get("author"), str) and script["author"].strip()):
        errors.append("'author' is missing or empty")

    cover = script.get("cover")
    if not isinstance(cover, dict):
        errors.append("'cover' is missing or not an object")
    else:
        if not (isinstance(cover.get("title_readout"), str) and cover["title_readout"].strip()):
            errors.append("cover.title_readout is missing or empty")
        if not (isinstance(cover.get("visual_prompt"), str) and cover["visual_prompt"].strip()):
            errors.append("cover.visual_prompt is missing or empty")
        elif not cover["visual_prompt"].strip().lower().startswith(VISUAL_PROMPT_PREFIX):
            errors.append("cover.visual_prompt does not begin with 'stickman illustration —'")

    shots = script.get("shots")
    if not isinstance(shots, list) or len(shots) < 15:
        errors.append(
            f"'shots' must be a list with at least 15 entries "
            f"(got {len(shots) if isinstance(shots, list) else 'non-list'})"
        )
        return errors  # per-shot checks are meaningless without a usable shots list

    total_words = 0
    for index, shot in enumerate(shots):
        where = f"shots[{index}]"
        if not isinstance(shot, dict):
            errors.append(f"{where} is not an object")
            continue
        if "shot_number" not in shot:
            errors.append(f"{where} is missing 'shot_number'")
        narration = shot.get("narration")
        if not (isinstance(narration, str) and narration.strip()):
            errors.append(f"{where} has empty or missing narration")
        else:
            total_words += len(narration.split())
        visual = shot.get("visual_prompt")
        if not (isinstance(visual, str) and visual.strip()):
            errors.append(f"{where} has empty or missing visual_prompt")
        elif not visual.strip().lower().startswith(VISUAL_PROMPT_PREFIX):
            errors.append(f"{where} visual_prompt does not begin with 'stickman illustration —'")

    length = story.get("estimated_length", "medium")
    low, high = WORD_RANGES.get(length, WORD_RANGES["medium"])
    hard_low = int(low * (1 - WORD_COUNT_TOLERANCE))
    hard_high = int(high * (1 + WORD_COUNT_TOLERANCE))
    if not hard_low <= total_words <= hard_high:
        errors.append(
            f"total narration word count {total_words} is outside the acceptable "
            f"range for '{length}' stories ({hard_low}-{hard_high})"
        )
    elif not low <= total_words <= high:
        logger.warning(
            "Narration word count %d is outside the strict %s range (%d-%d) "
            "but within tolerance — continuing",
            total_words, length, low, high,
        )

    shot_low, shot_high = SHOT_RANGES.get(length, SHOT_RANGES["medium"])
    if not shot_low <= len(shots) <= shot_high:
        logger.warning(
            "Shot count %d is outside the expected %s range (%d-%d) — continuing "
            "since the minimum of 15 is met",
            len(shots), length, shot_low, shot_high,
        )

    return errors


def generate_script(story, working_dir="working"):
    """Generate and validate the video script for a story. Returns dict or None."""
    debug_dir = os.path.join(working_dir, "debug")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set — cannot call OpenRouter")
        return None

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=story["title"],
        author=story["author"],
        moral=story["moral"],
        estimated_length=story["estimated_length"],
    )
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for model_index, model in enumerate((PRIMARY_MODEL, FALLBACK_MODEL)):
        if model_index > 0:
            logger.warning("Primary model failed — falling back to %s", model)
            time.sleep(SLEEP_BETWEEN_CALLS)

        content = _call_openrouter(base_messages, model, api_key, debug_dir, f"{model} initial")
        script = _parse_script(content) if content else None

        if script is None and content:
            logger.warning("Could not parse JSON from %s — sending correction message", model)
            time.sleep(SLEEP_BETWEEN_CALLS)
            correction_messages = base_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": CORRECTION_MESSAGE},
            ]
            content = _call_openrouter(
                correction_messages, model, api_key, debug_dir, f"{model} correction"
            )
            script = _parse_script(content) if content else None

        if script is None:
            logger.error("Model %s produced no parseable JSON script", model)
            continue

        script = _repair_visual_prompts(script)
        errors = validate_script(script, story)
        if errors:
            logger.error("Script from %s failed validation with %d error(s):", model, len(errors))
            for error in errors:
                logger.error("  VALIDATION FAILED: %s", error)
            continue

        total_words = sum(len(s["narration"].split()) for s in script["shots"])
        logger.info(
            "Validated script from %s: %d shots, %d narration words",
            model, len(script["shots"]), total_words,
        )
        return script

    logger.error("All models failed to produce a valid script — skipping run")
    return None
