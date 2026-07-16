# Style Reference

Place your `style_reference.png` file in this directory **before the first run**.

This image is a detailed animation style guide — character designs, expressions,
poses, colour palette, and background examples. On every image generation call,
the pipeline sends it to Pollinations as a visual reference (`&image=...`) so
every background and character sprite is anchored to the exact look you want.

## How it is used

The pipeline builds the reference URL dynamically from the `GITHUB_REPOSITORY`
environment variable that GitHub Actions provides automatically:

```
https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/main/assets/style_reference.png
```

For this to work the file **must be committed and tracked by git** (it is — this
folder is not gitignored). After you add or replace `style_reference.png`, commit
and push it to `main`.

## Requirements

- Filename must be exactly `style_reference.png`
- PNG format
- Committed to the `main` branch so the raw URL resolves publicly

If the file is missing, image generation still runs — just without the style
anchor, so results will drift from your intended look.
