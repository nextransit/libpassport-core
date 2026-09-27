#!/usr/bin/env python3
"""Theme regression scan for PassportGUI (tests/gui.py + tests/gui_ctk.py).

Asserts:
  dark mode  -> zero light-coloured surfaces (mean RGB > 225) anywhere in the
                widget tree, and ttk style elements resolve dark;
  light mode -> zero dark-coloured surfaces (mean RGB < 60), styles light.

Usage:
    python gui_theme_scan.py [gui|gui_ctk]

Exit code 0 = PASS, 1 = FAIL. Must run on a real display (macOS ok).
"""
import sys
from pathlib import Path

THRESH_LIGHT = 225.0   # mean RGB above this counts as a light surface
THRESH_DARK = 60.0     # mean RGB below this counts as a dark surface

# style names checked against ttk.Style().lookup().
# Foreground colours are excluded: tab text etc. is drawn on a themed
# background and its light/dark polarity is a design choice, not a leak.
STYLE_ELEMENTS = {
    "Data.Treeview": ("background", "fieldbackground"),
    "Data.Treeview.Heading": ("background",),
    "TNotebook": ("background",),
    "TNotebook.Tab": ("background",),
    "TLabelframe": ("background",),
    "Field.TEntry": ("fieldbackground",),
}

COLOR_KEYS = ("bg", "background", "highlightbackground", "fg_color",
              "button_color", "hover_color")


def mean_rgb(hex_color):
    try:
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return (int(h[0:2], 16) + int(h[2:4], 16) + int(h[4:6], 16)) / 3.0
    except (ValueError, IndexError):
        return None


def walk(widget, out, path="ROOT", depth=0):
    import tkinter as tk
    if depth > 12:
        return
    if isinstance(widget, tk.Menu):
        return
    name = type(widget).__name__
    # customtkinter 6.x draws its rounded corners on internal CTkCanvas
    # widgets whose tk `bg` attribute keeps the creation-time colour even
    # after a theme switch (the surface is redrawn via create_rectangle).
    # Sub-pixel internal widgets (1x1 labels, 1px highlight borders) are
    # rendering noise, not surfaces — skip both.
    if name == "CTkCanvas":
        return
    try:
        wd = widget.winfo_width()
        ht = widget.winfo_height()
    except tk.TclError:
        return
    if wd < 4 or ht < 4:
        return
    for key in COLOR_KEYS:
        try:
            val = widget.cget(key)
        except (tk.TclError, ValueError):
            continue
        if isinstance(val, str) and val.startswith("#"):
            out.append((path, wd, ht, key, val))
    for child in widget.winfo_children():
        walk(child, out, f"{path}/{type(child).__name__}", depth + 1)


def style_values(app, style):
    res = {}
    for name, attrs in STYLE_ELEMENTS.items():
        for attr in attrs:
            v = style.lookup(name, attr)
            res[f"{name}.{attr}"] = v
    return res


def run_checks(mod_name, theme, want_light):
    import importlib
    mod = importlib.import_module(mod_name)
    app = mod.PassportGUI()
    app._switch_theme(theme)
    app.update()
    app.update_idletasks()
    import tkinter.ttk as ttk
    style = ttk.Style()

    colors = []
    walk(app, colors)
    bad = []
    for path, w, h, key, val in colors:
        m = mean_rgb(val)
        if m is None:
            continue
        if (want_light and m < THRESH_DARK) or (not want_light and m > THRESH_LIGHT):
            bad.append((path, w, h, key, val, round(m, 1)))

    sv = style_values(app, style)
    bad_style = []
    for k, v in sv.items():
        m = mean_rgb(v)
        if m is None:
            continue
        if (want_light and m < THRESH_DARK) or (not want_light and m > THRESH_LIGHT):
            bad_style.append((k, v, round(m, 1)))
    app.destroy()
    return bad, bad_style


def main():
    mod_name = sys.argv[1] if len(sys.argv) > 1 else "gui"
    if mod_name not in ("gui", "gui_ctk"):
        print(f"unknown module {mod_name}")
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # A saved pref would make the FIRST constructed app start in that
    # theme and the first _switch_theme call would be a no-op flip.
    (Path(__file__).resolve().parent / ".gui_theme.json").unlink(missing_ok=True)
    failed = 0
    for theme, want_light, label in (
            ("dark", False, "dark mode  → 0 light surfaces"),
            ("light", True, "light mode → 0 dark surfaces")):
        bad, bad_style = run_checks(mod_name, theme, want_light)
        if bad or bad_style:
            failed += 1
            print(f"[FAIL] {mod_name}: {label}")
            for b in bad[:20]:
                print(f"  widget {b[0]} {b[1]}x{b[2]} {b[3]}={b[4]} (mean {b[5]})")
            for b in bad_style[:10]:
                print(f"  style  {b[0]} = {b[1]} (mean {b[2]})")
            if len(bad) > 20:
                print(f"  ... and {len(bad) - 20} more widgets")
        else:
            print(f"[PASS] {mod_name}: {label}")
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())