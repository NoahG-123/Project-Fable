# Project Fable — Grimm Pipeline (V2)

A fully automated Instagram Reels pipeline. Every day it retells one classic
fairy tale or fable (Brothers Grimm, Aesop, Hans Christian Andersen) as one or
more 60–90 second vertical videos: style-anchored storybook illustrations,
kinetic captions, and warm third-person documentary narration.

The channel is faceless — the illustrations and narrator voice are the entire
identity. The pipeline runs autonomously on GitHub Actions. The only human
touchpoint is downloading the artifact and posting to Instagram.

## How it works

Seven stages, orchestrated by `pipeline/main.py`:

1. **Ingest** — picks the first `"pending"` story from `stories.json`. Multi-part
   stories (`"parts": 2` or `3`) produce one video per part. The story is marked
   `"done"` only after every part is produced.
2. **Enrich** — DeepSeek R1 via OpenRouter writes the multi-part script as JSON
   (third person, past tense), strictly validated with wide safety-net tolerances.
3. **Assets** — for each shot, Pollinations generates a **background** and a
   **character sprite** separately, composited with Pillow. Every call is anchored
   to your `assets/style_reference.png`.
4. **Voice** — Chatterbox TTS runs locally on the runner's CPU to narrate each
   part; `whisper-timestamped` extracts word-level caption timing.
5. **Assemble** — MoviePy + FFmpeg build each part's 1080×1920 30fps reel: static
   cover card, still images cut to the narration, kinetic pill captions, watermark.
6. **Validate** — eight quality gates per part → `working/debug/quality_report.txt`.
7. **Distribute** — a per-part run summary lands on the workflow run page.

## Setup

### 1. Add your style reference

Place `style_reference.png` in the `assets/` folder and commit it to `main`. See
[`assets/README.md`](assets/README.md). It is sent to Pollinations on every image
call so all visuals stay on-style. The raw URL is built automatically from the
`GITHUB_REPOSITORY` variable — nothing to configure.

### 2. Create the secrets

**Settings → Secrets and variables → Actions**:

| Secret | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (paid tier recommended) |
| `CHANNEL_HANDLE` | Yes | Instagram handle, e.g. `@grimmtales` — rendered as the watermark |
| `VOICE_REFERENCE_PATH` | No | Path (in the repo) to a voice reference WAV for cloning |
| `MUSIC_BED_PATH` | No | Path (in the repo) to an optional background music file (mixed at −18dB) |

Missing optional secrets are fine — the run warns and continues.

The workflow then runs daily at 07:00 UTC.

## Adding stories to the queue

Append entries to `stories.json`:

```json
{
  "title": "The Golden Goose",
  "author": "Brothers Grimm",
  "moral": "Kindness is rewarded in unexpected ways",
  "estimated_length": "medium",
  "parts": 2,
  "status": "pending"
}
```

- `estimated_length`: `short`, `medium`, or `long`
- `parts`: `1` (Aesop fables), `2` (most Grimm/HCA), or `3` (the longest stories)

Stories are consumed top to bottom — the first `"pending"` entry wins.

## Triggering a manual run

**Actions → Daily Fairy Tale Reel → Run workflow**. Same pipeline as the schedule.

## Downloading the videos

Open the workflow run → **Artifacts**:

- `reels-<run id>` — every part's `part_N_reel.mp4` (kept 7 days). Post them in order.
- `debug-<run id>` — raw LLM responses, quality report, image fetch log, run log (3 days)

Check the run summary on the same page before posting.

## Troubleshooting

| Symptom | Where to look | Likely cause |
|---|---|---|
| Run fails at Enrich | `debug` → `raw_response.txt` | Malformed LLM JSON, exhausted OpenRouter credits, or a validation failure (the log names the exact failed check) |
| Voice stage slow | `run_log.txt` | Chatterbox TTS runs on the runner's CPU — model load + per-shot generation is inherently slow with no GPU; this is expected, not an error |
| Placeholder frames in the video | `image_fetch_log.txt` | Pollinations timed out on those shots — placeholders were substituted so the run could finish |
| Visuals drift from your style | — | Confirm `assets/style_reference.png` is committed to `main` and the raw URL resolves |
| No captions / coverage gate fails | `run_log.txt` | Whisper produced no word timestamps; video still produced, inspect before posting |
| `reels-*` artifact missing but job "succeeded" | job log | Story queue is empty — add pending stories |
| Only some parts produced | job summary | A part failed mid-way; the story stays pending and is retried next run |

### Notes

- A run is **successful if at least one part video is produced**. A story is only
  marked done when **all** its parts are produced.
- The `working/` directory is recreated from scratch every run and is not committed.
- The stories.json commit uses `[skip ci]`, so advancing the queue never loops.
