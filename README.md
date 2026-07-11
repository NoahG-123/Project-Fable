# Project Fable — Grimm Pipeline

A fully automated Instagram Reels pipeline. Every day it retells one classic
fairy tale or fable (Brothers Grimm, Aesop, Hans Christian Andersen) as a
60-second vertical video: stickman illustrations, kinetic captions, and a
confident second-person narrator — "you are the tortoise, and a hare has just
challenged you to a race."

The pipeline runs entirely on GitHub Actions. The only human touchpoint is
downloading the finished video artifact and posting it to Instagram.

## How it works

Each run walks through seven stages (`pipeline/main.py` orchestrates):

1. **Ingest** — picks the first `"pending"` story from `stories.json`, marks it
   `"done"` (the workflow commits the change back, so the queue advances daily).
2. **Enrich** — DeepSeek R1 via OpenRouter writes the shot-by-shot script as
   JSON, which is strictly validated before anything else runs.
3. **Assets** — Pollinations.ai generates one stickman illustration per shot
   (free API, no key; one seed per run for visual consistency).
4. **Voice** — Chatterbox TTS narrates the full script on CPU;
   whisper-timestamped extracts word-level timestamps.
5. **Assemble** — MoviePy + FFmpeg build the 1080x1920 30fps reel: cover card,
   still images timed to the narration, kinetic pill captions, watermark.
6. **Validate** — eight quality gates (duration, resolution, caption coverage,
   silence check, …) written to `working/debug/quality_report.txt`.
7. **Distribute** — a run summary lands on the workflow run page.

## Setup

1. Create the repository secrets below (**Settings → Secrets and variables →
   Actions**).
2. That's it — the workflow in `.github/workflows/daily_reel.yml` runs daily at
   07:00 UTC.

### Secrets

| Secret | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for DeepSeek access |
| `CHANNEL_HANDLE` | Yes | Instagram handle, e.g. `@grimmtales` — rendered as the watermark |
| `MUSIC_BED_PATH` | No | Path to an optional background music file in the repo (mixed at -18dB) |
| `CHATTERBOX_VOICE_REFERENCE` | No | Path to a voice reference audio file in the repo for voice cloning |

Missing optional secrets are fine — the run warns and continues.

## Adding stories to the queue

Append entries to `stories.json`:

```json
{
  "title": "The Golden Goose",
  "author": "Brothers Grimm",
  "moral": "Kindness is rewarded in unexpected ways",
  "estimated_length": "medium",
  "status": "pending"
}
```

`estimated_length` is one of `short` (Aesop fables, ~18-20 shots), `medium`
(~24-26 shots), or `long` (~28-30 shots). Stories are consumed top to bottom —
the first `"pending"` entry wins. When the queue runs dry the pipeline logs a
warning and exits without producing a video.

## Triggering a manual run

**Actions → Daily Fairy Tale Reel → Run workflow**. The `workflow_dispatch`
trigger runs the exact same pipeline as the daily schedule.

## Downloading the video

Open the workflow run, scroll to **Artifacts**, and download:

- `reel-<run id>` — the finished `reel.mp4` (kept 7 days)
- `debug-<run id>` — full debug trail: raw LLM responses, quality report,
  image fetch log, run log (kept 3 days)

Check the run summary on the same page for the quality gate results before
posting.

## Troubleshooting

| Symptom | Where to look | Likely cause |
|---|---|---|
| Run fails at Enrich | `debug` artifact → `raw_response.txt` | Malformed LLM JSON, exhausted OpenRouter credits, or a validation failure (the log names the exact failed check) |
| White frames with shot numbers in the video | `image_fetch_log.txt` | Pollinations.ai timed out on those shots — placeholders were substituted so the run could finish |
| No captions / caption coverage gate fails | `run_log.txt` | Whisper produced no word timestamps; the video is still produced, inspect before posting |
| Run is very slow | — | Normal: Chatterbox TTS on CPU is the slow stage; the job allows 120 minutes. First run also downloads model weights (`torch` is large) |
| `reel-*` artifact missing but job "succeeded" | job log | Story queue is empty — add pending stories to `stories.json` |
| Robotic/wrong voice | — | Check `CHATTERBOX_VOICE_REFERENCE` points to a real file in the repo; otherwise the default voice is used |
| Quality gate failures | `quality_report.txt` | Gates never block the artifact — they flag it. Inspect the video before posting |

### Notes

- A run counts as **successful if and only if `working/output/reel.mp4` exists**.
  Quality gate failures produce warnings, not crashes.
- The `working/` directory is recreated from scratch on every run and is not
  committed.
- The stories.json commit uses `[skip ci]`, so advancing the queue never
  triggers a loop.
