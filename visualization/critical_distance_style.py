"""Presentation constants for the critical-distance visualization.

All purely cosmetic values (colors, fonts, sizes) live here so
critical_distance.py stays focused on logic. The severity thresholds that
*decide* which color applies stay in critical_distance.py — only the resulting
colors are here.
"""

# Robot collision spheres in their normal (non-critical) state.
SPHERE_DEFAULT_COLOR = (0, 200, 255)

# Severity colors (RGB) for the highlighted critical pair: safe / warn / danger.
SEVERITY_SAFE = (40, 160, 40)
SEVERITY_WARN = (240, 150, 0)
SEVERITY_DANGER = (200, 0, 0)

# Emoji dots for the live label, matching the severity tiers.
DOT_SAFE = "🟢"
DOT_WARN = "🟠"
DOT_DANGER = "🔴"

# Gap-ruler line in the 3-D scene.
GAP_LINE_WIDTH = 6.0

# --- Timeline figure -------------------------------------------------------
TIMELINE_ASPECT = 0.6

# d_crit curve.
CURVE_COLOR = "rgb(30,30,30)"
CURVE_WIDTH = 2

# Danger bands behind the curve (penetration / danger / warn / safe).
BAND_PENETRATION = "rgba(200,0,0,0.30)"
BAND_DANGER = "rgba(220,60,60,0.18)"
BAND_WARN = "rgba(240,150,0,0.15)"
BAND_SAFE = "rgba(40,160,40,0.12)"

# Dotted line at d_crit = 0.
ZERO_LINE_COLOR = "rgb(120,120,120)"
ZERO_LINE_WIDTH = 1

# Diamond marking the trajectory-wide minimum.
MARKER_SIZE = 7
MARKER_COLOR = "rgb(200,0,0)"
MARKER_SYMBOL = "diamond"

# Vertical playback cursor.
CURSOR_COLOR = "rgb(30,90,200)"
CURSOR_WIDTH = 2

# Title / subtitle text.
TITLE_FONT_SIZE = 14
SUBTITLE_FONT_SIZE = 11
SUBTITLE_COLOR = "rgb(90,90,90)"

# Plot frame.
MARGIN = {"l": 55, "r": 15, "t": 55, "b": 35}
PLOT_BGCOLOR = "white"
