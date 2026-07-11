"""All brand constants for the Grimm pipeline.

Every colour, font, size and layout value used anywhere in the pipeline
lives here. Never hardcode these values in other modules.
"""

import os

# Canvas
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
FPS = 30

# Colours
BACKGROUND = "#FFFFFF"    # white — matches stickman illustration style
INK = "#1A1A1A"           # near-black for text
ACCENT = "#E83B2A"        # bold red — caption emphasis, title accents
CAPTION_BG = "#000000"    # caption pill background (apply at 70% opacity)
CAPTION_TEXT = "#FFFFFF"  # caption primary text
CAPTION_BG_OPACITY = 0.70

# Typography — logical names (ImageMagick style) plus resolved file paths for PIL
FONT_TITLE = "DejaVu-Sans-Bold"    # cover title
FONT_CAPTION = "DejaVu-Sans-Bold"  # kinetic captions
FONT_HANDLE = "DejaVu-Sans"        # watermark

_FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
    "/usr/share/fonts/TTF",
    os.path.expanduser("~/.fonts"),
]


def _find_font(filename):
    for directory in _FONT_DIRS:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None


FONT_TITLE_PATH = _find_font("DejaVuSans-Bold.ttf")
FONT_CAPTION_PATH = _find_font("DejaVuSans-Bold.ttf")
FONT_HANDLE_PATH = _find_font("DejaVuSans.ttf")

# Sizes
TITLE_FONT_SIZE = 72
CAPTION_FONT_SIZE = 58
HANDLE_FONT_SIZE = 32

# Layout
CAPTION_Y_RATIO = 0.72          # captions at 72% of frame height
WATERMARK_MARGIN = 32           # pixels from edge
CAPTION_MAX_WIDTH_RATIO = 0.80  # captions max 80% of frame width


def hex_to_rgb(hex_colour):
    """'#E83B2A' -> (232, 59, 42)"""
    value = hex_colour.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
