"""Stage 2 — Enrich (V2).

Call DeepSeek R1 via OpenRouter with the V2 master prompt (third person, past
tense, style-reference voice matching). Parse and validate the multi-part JSON
response. Validation uses WIDE tolerances — the ranges are safety nets that
only reject catastrophically broken output, not strict enforcers.

Parse chain on failure: strip markdown fences -> json.loads -> balanced-brace
extraction of the first {...} block -> one correction message -> fallback model
-> give up. The full raw response is appended to working/debug/raw_response.txt
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
FALLBACK_MODEL = "deepseek/deepseek-chat"

SLEEP_BETWEEN_CALLS = 4          # seconds between any sequential OpenRouter calls
RATE_LIMIT_BACKOFFS = [6, 12, 24, 48]  # exponential backoff on 429
REQUEST_TIMEOUT = 300            # deepseek-r1 reasoning can be slow

VISUAL_PROMPT_PREFIX = "scene:"

# WIDE safety-net ranges (per part) — reject only catastrophically broken output.
WORD_RANGES = {"short": (50, 300), "medium": (50, 500), "long": (50, 700)}

SYSTEM_PROMPT = """You are a script writer for a short-form video channel that retells classic fairy tales and fables in the style of warm, engaging documentary narration. You write in the style of the following example — study it carefully and match its voice, rhythm, pacing, and wit exactly:

---

STYLE REFERENCE — MATCH THIS VOICE PRECISELY:

it's medieval Japan and you're born in 1584 in a small village named Miiamoto Being born in Miiamoto and all your mom obviously gives you the name Benoske Wait what you don't know who your dad is And unfortunately you won't know your mom either because she passes away during childbirth Luckily she had remarried a guy named Shinman Mooney and he's going to be your dad Mooney is like the perfect blend of supremely skilled swordsman and supremely abusive stepfather You don't really like him and you get into a ton of fights with him until he kicks you out of the house when you're six You go to live with your uncle who's pretty chill because he's a Buddhist So you live with your uncle for a few years and you're 12 now And one day you're walking around the town and you see a poster that catches your eye It's an open challenge for a sword duel from a guy named Arma Kihei He's looking to test his skill against anyone who's brave enough to face him You grab a long stick find him and tell him you're down to duel and he accepts The duel begins and Arma draws his blade You circle him with your stick and he charges at you You swing your stick with all your might and you knock his sword to the ground He looks at you with a mixture of confusion and fear in his eyes and on instinct you quickly strike his head and he collapses to the ground You just won your first duel Hooray oh wait He's still alive All right that'll do it Now you just won your first duel Hooray Dueling is pretty fun But your uncle didn't really like you fighting and wants you to avoid violence and seek enlightenment Your uncle is cool so you listen to him until this guy named Tatashima Akiyama challenges you to a duel when you're 15 And you go and kill him too You decide that your life is probably not going to be the one your uncle wished for you And so you change your name to Miiamoto Mousashi and leave his home to go on a quest to become a great warrior

---

CRITICAL DIFFERENCES FROM THE REFERENCE:
- Write in THIRD PERSON. The viewer is not the character. Use "he", "she", "they" for characters — never "you"
- Write in PAST TENSE throughout
- The reference above uses second person — yours does not. Everything else about the voice, rhythm, wit, and pacing — match exactly.

Your output must always be valid JSON. No markdown. No preamble. No explanation. Only the JSON object."""

USER_PROMPT_TEMPLATE = """Retell the following story as a script for one or more 60-90 second Instagram Reels:

Title: {title}
Author: {author}
Moral: {moral}
Estimated Length: {estimated_length}
Number of Parts: {parts}

---

VOICE AND TONE — follow exactly:

- Third person, past tense throughout
- Match the wit, rhythm, pacing and warmth of the style reference in the system prompt exactly
- Let absurd situations land naturally — never point at them or call them funny
- Use short sentences for action. Longer sentences for context and momentum
- Occasional dry asides are good — do not overuse them
- Never use crude humour, mockery, or incomplete sentences
- The ending of each part lands naturally — no grand speeches

---

PARTS SYSTEM:

If parts is 1: write one complete self-contained script.

If parts is 2 or 3: split the story at natural dramatic tension points — not arbitrarily. Each part must:
- End at a genuine cliffhanger or moment of unresolved tension
- Open with one sentence recapping where the previous part left off (except Part 1)
- Feel satisfying on its own while leaving the viewer wanting the next part
- Be clearly labelled in the title_readout as "Part 1", "Part 2" etc.

---

STRUCTURE:

Each part has a cover and shots.

Cover:
- No narration except title_readout spoken aloud
- title_readout format: "[Title] — [Author]" for Part 1, "[Title] Part 2 — [Author]" for subsequent parts
- visual_prompt must capture the single most dramatic or visually interesting moment of that part
- Must make someone stop scrolling — bold, dynamic, immediately readable

Shots:
- Shot 1 opens immediately into the story world — no preamble, no wind-up
- Each shot is one frozen visual moment
- One to two sentences of narration per shot
- Natural momentum throughout — never linger, never repeat
- Final two shots are the ending — outcome first, then moral lands naturally

Shot count per part:
- short (1 part only): 18 to 22 shots
- medium per part: 20 to 24 shots
- long per part: 22 to 26 shots

---

VISUAL PROMPT RULES:

Every visual_prompt has two parts separated by " | ":

scene: [background description] | character: [character description and pose]

Example:
"scene: warm forest clearing, large oak tree, golden afternoon light, illustrated storybook style, muted earthy tones | character: young stickman boy standing at base of tree, looking up with wide curious eyes, one hand reaching toward a glowing object inside the hollow"

Rules:
- Always use this exact format with the pipe separator
- Scene describes the background environment only
- Character describes who is present, their pose, and their expression
- Both must be specific and visual — no abstract descriptions
- Muted earthy tones, clean line work, hand-drawn storybook aesthetic on every prompt
- Never describe motion or sequences — only the single frozen moment

---

WORD COUNT — these are targets, not hard limits:
- short: aim for 150 to 200 words total narration
- medium: aim for 200 to 280 words per part
- long: aim for 250 to 320 words per part

Count your words carefully. If you exceed the target by more than 50 words, trim before outputting.

---

OUTPUT JSON STRUCTURE:

For a single part story:
{{
  "title": "string",
  "author": "string",
  "parts": 1,
  "scripts": [
    {{
      "part": 1,
      "cover": {{
        "title_readout": "string",
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
  ]
}}

For multi-part stories, the scripts array contains one object per part with the same structure.

Do not include any text outside the JSON object."""

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
    immediately. Server errors get one retry per backoff slot too.
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
            debug_dir, f"{label} attempt {attempt + 1} (model={model})",
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
            logger.error("[%s] OpenRouter returned HTTP %d: %s", label, response.status_code, response.text[:500])
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
    """Ensure every visual_prompt starts with 'scene:'. Repair (and log) if not."""
    def repair(prompt, where):
        if not isinstance(prompt, str):
            return prompt
        clean = prompt.strip()
        if clean.lower().startswith(VISUAL_PROMPT_PREFIX):
            return clean
        logger.warning("Repairing %s visual_prompt missing 'scene:' prefix", where)
        if "|" in clean:
            return f"scene: {clean}"
        return f"scene: {clean} | character: stickman figure, storybook style, muted earthy tones"

    for part in script.get("scripts") or []:
        if not isinstance(part, dict):
            continue
        cover = part.get("cover")
        if isinstance(cover, dict) and cover.get("visual_prompt"):
            cover["visual_prompt"] = repair(cover["visual_prompt"], f"part {part.get('part', '?')} cover")
        for shot in part.get("shots") or []:
            if isinstance(shot, dict) and shot.get("visual_prompt"):
                shot["visual_prompt"] = repair(
                    shot["visual_prompt"],
                    f"part {part.get('part', '?')} shot {shot.get('shot_number', '?')}",
                )
    return script


def validate_script(script, story):
    """Validate the parsed multi-part script. Returns a list of error strings.

    Empty list means usable. Word-count ranges are WIDE safety nets — they only
    catch catastrophically broken output.
    """
    errors = []

    if not isinstance(script, dict):
        return ["script is not a JSON object"]

    if not (isinstance(script.get("title"), str) and script["title"].strip()):
        errors.append("'title' is missing or empty")
    if not (isinstance(script.get("author"), str) and script["author"].strip()):
        errors.append("'author' is missing or empty")

    scripts = script.get("scripts")
    if not isinstance(scripts, list) or not scripts:
        errors.append("'scripts' must be a non-empty list of parts")
        return errors

    length = story.get("estimated_length", "medium")
    low, high = WORD_RANGES.get(length, WORD_RANGES["medium"])

    for pi, part in enumerate(scripts):
        where = f"scripts[{pi}]"
        if not isinstance(part, dict):
            errors.append(f"{where} is not an object")
            continue

        cover = part.get("cover")
        if not isinstance(cover, dict):
            errors.append(f"{where}.cover is missing or not an object")
        else:
            if not (isinstance(cover.get("title_readout"), str) and cover["title_readout"].strip()):
                errors.append(f"{where}.cover.title_readout is missing or empty")
            cvp = cover.get("visual_prompt")
            if not (isinstance(cvp, str) and cvp.strip()):
                errors.append(f"{where}.cover.visual_prompt is missing or empty")
            elif not cvp.strip().lower().startswith(VISUAL_PROMPT_PREFIX):
                errors.append(f"{where}.cover.visual_prompt does not begin with 'scene:'")

        shots = part.get("shots")
        if not isinstance(shots, list) or len(shots) < 15:
            errors.append(
                f"{where}.shots must be a list with at least 15 entries "
                f"(got {len(shots) if isinstance(shots, list) else 'non-list'})"
            )
            continue

        total_words = 0
        for si, shot in enumerate(shots):
            sw = f"{where}.shots[{si}]"
            if not isinstance(shot, dict):
                errors.append(f"{sw} is not an object")
                continue
            if "shot_number" not in shot:
                errors.append(f"{sw} is missing 'shot_number'")
            narration = shot.get("narration")
            if not (isinstance(narration, str) and narration.strip()):
                errors.append(f"{sw} has empty or missing narration")
            else:
                total_words += len(narration.split())
            visual = shot.get("visual_prompt")
            if not (isinstance(visual, str) and visual.strip()):
                errors.append(f"{sw} has empty or missing visual_prompt")
            elif not visual.strip().lower().startswith(VISUAL_PROMPT_PREFIX):
                errors.append(f"{sw} visual_prompt does not begin with 'scene:'")

        if not low <= total_words <= high:
            errors.append(
                f"{where} narration word count {total_words} is outside the wide "
                f"safety range for '{length}' parts ({low}-{high})"
            )

    return errors


def generate_script(story, working_dir="working"):
    """Generate and validate the V2 multi-part script. Returns dict or None."""
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
        parts=story.get("parts", 1),
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
            content = _call_openrouter(correction_messages, model, api_key, debug_dir, f"{model} correction")
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

        n_parts = len(script["scripts"])
        total_shots = sum(len(p.get("shots", [])) for p in script["scripts"])
        logger.info("Validated script from %s: %d part(s), %d total shots", model, n_parts, total_shots)
        return script

    logger.error("All models failed to produce a valid script — skipping run")
    return None
