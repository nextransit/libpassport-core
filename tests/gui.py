#!/usr/bin/env python3
"""passport_test_gui -- Tkinter GUI that exercises all 4 modules.

Tabs (in order):

  1. MRZ OCR 测试       -- corpus-driven per-case benchmark (>=100 cases),
                          traditional vs CNN, side-by-side with image preview.
  2. NFC 读卡           -- mock R-APDU full eMRTD flow over JSON scripts.
  3. MRZ 解码           -- text TD3 -> fields / check digits.
  4. 防伪验证           -- MRZ + sample -> AC report.
  5. 照片比对           -- two image files -> SSIM similarity.

All four CLI tools must already be built in their respective build/
directories. Missing tools are reported in the status bar.
"""
from __future__ import annotations
import os, subprocess, sys, threading, time, json, re, hashlib, tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# High-DPI awareness must be set before Tk initialises (Windows only;
# macOS/Retina is handled natively). Prevents blurry controls/table
# text on scaled displays.
if sys.platform.startswith("win"):
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

try:
    import ttkbootstrap as ttk
    from ttkbootstrap import Window
    if not hasattr(ttk, "PanedWindow"):
        ttk.PanedWindow = ttk.Panedwindow
    HAS_TTKB = True
except ImportError:
    from tkinter import ttk
    HAS_TTKB = False

ROOT = Path(__file__).resolve().parent.parent

def tool_path(name, sub_dirs=("1_mrz_decode", "2_anticounterfeit",
                              "3_face_recognition", "4_nfc_reader")):
    """Locate a built CLI under */build/."""
    for sub in sub_dirs:
        cand = ROOT / sub / "build" / name
        if cand.exists():
            return cand
    return ROOT / sub_dirs[0] / "build" / name

MRZ_TOOL  = tool_path("mrz_tool")
AC_TOOL   = tool_path("ac_tool")
FACE_TOOL = tool_path("face_tool")
GEN_SAMP  = tool_path("gen_sample")
NFC_TOOL  = tool_path("nfc_tool")

# OCR back-end display names, kept in one place so the method picker,
# the summary rows, and the detail rows all stay consistent.
METHOD_LABELS = {
    "traditional": "传统模板",
    "cnn":         "轻量CNN",
    "tesseract":   "Tesseract",
    "paddle":      "PaddleOCR",
}
METHOD_ORDER = ["traditional", "cnn", "tesseract", "paddle"]

FACE_DATA_DIR = ROOT / "3_face_recognition" / "data"
FACE_SAMPLES = {
    "face_a (合成, 128×128)": FACE_DATA_DIR / "face_a.ppm",
    "face_b (合成, 128×128)": FACE_DATA_DIR / "face_b.ppm",
    "face_c (合成, 128×128)": FACE_DATA_DIR / "face_c.ppm",
}

SAMPLE_PASSPORT_PNG = ROOT / "tests" / "assets" / "passport_sample.png"

SAMPLE_MRZ = (
    "P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO6908061F9406236ZE184226B<<<<<18\n"
)

# ---------------------------------------------------------------------------
# Cyber-tech theme system (.impeccable.md)
#
# Two brand themes are registered into ttkbootstrap so every ttk widget
# (Notebook, buttons, progress bars, Treeviews, scrollbars, entries) follows
# the palette in one switch:
#
#   cyber-dark  -- deep space blue-black (#0a0e14) + neon cyan accent
#   cyber-light -- cool paper white + electronic blue accent
#
# Semantic colours (success green / fail red / in-flight violet) keep their
# MEANING across both modes while each mode gets harmonised values. The
# "tk" dict extends the 16 ttkbootstrap keys with tokens for the non-ttk
# widgets (canvas, Text tags, chips, CTA buttons) that need manual recolour.
# ---------------------------------------------------------------------------
THEME_PREF_FILE = Path(__file__).resolve().parent / ".gui_theme.json"

CYBER_THEMES = {
    "cyber-dark": {
        "mode": "dark",
        "colors": {
            "primary":   "#00e5ff",
            "secondary": "#2e86ff",
            "success":   "#2ee6a8",
            "info":      "#2e86ff",
            "warning":   "#ffb454",
            "danger":    "#ff5c7a",
            "light":     "#1c2530",
            "dark":      "#060a10",
            "bg":        "#0a0e14",
            "fg":        "#dce3ec",
            "selectbg":  "#0f2a33",
            "selectfg":  "#7df3ff",
            "border":    "#21262D",
            "inputfg":   "#e8eef6",
            "inputbg":   "#111827",
            "active":    "#16202c",
        },
        "tk": {
            "card": "#121620", "code": "#0a0e14",
            "dim": "#8b98a9", "violet": "#b388ff",
            "cta_bg": "#1F6FEB", "cta_fg": "#ffffff", "cta_hover": "#3a8bff",
            "ghost_bg": "#30363D", "ghost_fg": "#C9D1D9",
            "ghost_hover": "#3a434d", "ghost_border": "#30363D",
            "canvas": "#121620", "canvas_border": "#21262D",
            "chip_idle":  ("#00e5ff", "#0f2a33"),
            "chip_ok":    ("#2ee6a8", "#0f3325"),
            "chip_run":   ("#b388ff", "#251a3d"),
            "chip_fail":  ("#ff5c7a", "#3a1620"),
            "chip_dim":   ("#6a737d", "#1a1d22"),
            "tag_ok":     ("#0d3a2a", "#7eeac2"),
            "tag_fail":   ("#43131f", "#ffaabb"),
            "tag_warn":   ("#3d2f08", "#ffd479"),
            "tag_match":  ("#2ee6a8", None),
            "tag_mm":     ("#ffaabb", "#43131f"),
            "tag_plain":  ("#c9d4e0", "#0a0e14"),
        },
    },
    "cyber-light": {
        "mode": "light",
        "colors": {
            "primary":   "#0b5cad",
            "secondary": "#2e86ff",
            "success":   "#027a48",
            "info":      "#2e86ff",
            "warning":   "#b58a00",
            "danger":    "#b42318",
            "light":     "#ffffff",
            "dark":      "#14213a",
            "bg":        "#f4f6fa",
            "fg":        "#1a2433",
            "selectbg":  "#d5ecf7",
            "selectfg":  "#0b3a52",
            "border":    "#d0d5dd",
            "inputfg":   "#111827",
            "inputbg":   "#ffffff",
            "active":    "#e9edf3",
        },
        "tk": {
            "card": "#ffffff", "code": "#ffffff",
            "dim": "#5c6773", "violet": "#6c2bd9",
            "cta_bg": "#0b5cad", "cta_fg": "#ffffff", "cta_hover": "#2e86ff",
            "ghost_bg": "#ffffff", "ghost_fg": "#0b5cad",
            "ghost_hover": "#e9edf3", "ghost_border": "#b9d3ea",
            "canvas": "#eef1f6", "canvas_border": "#9fc3dd",
            "chip_idle":  ("#0b5cad", "#e6f0fa"),
            "chip_ok":    ("#027a48", "#d1fadf"),
            "chip_run":   ("#6c2bd9", "#ede4ff"),
            "chip_fail":  ("#b42318", "#fee4e2"),
            "chip_dim":   ("#667085", "#eaecf0"),
            "tag_ok":     ("#c8e6c9", "#1b5e20"),
            "tag_fail":   ("#ffcdd2", "#b71c1c"),
            "tag_warn":   ("#fff3cd", "#8a6d00"),
            "tag_match":  ("#027a48", None),
            "tag_mm":     ("#b42318", "#ffe4e6"),
            "tag_plain":  ("#1a2433", "#ffffff"),
        },
    },
}


def _load_theme_pref():
    try:
        return json.loads(THEME_PREF_FILE.read_text()).get("theme")
    except Exception:
        return None


def _save_theme_pref(name):
    try:
        THEME_PREF_FILE.write_text(json.dumps({"theme": name}))
    except Exception:
        pass


class _Tooltip:
    """Hover tooltip (overrideredirect Toplevel) showing a full path."""

    def __init__(self, widget):
        self.widget = widget
        self.tip = None
        self.text = ""
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")

    def set_text(self, text):
        self.text = text or ""

    def _enter(self, _e):
        if not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f"+{x}+{y}")
            tk.Label(self.tip, text=self.text, bg="#21262D", fg="#C9D1D9",
                     font=("Menlo", 9), padx=8, pady=4,
                     justify="left").pack()
        except tk.TclError:
            self.tip = None

    def _leave(self, _e):
        if self.tip is not None:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None


class _ThemedButton(tk.Frame):
    """Flat accent button macOS aqua cannot ignore.

    tk.Button on macOS ignores -background on its native bezel (and the
    workaround matrix of relief/bd/highlight flags is version-fragile),
    so the face is a tk.Frame + tk.Label — both always honour bg/fg.
    Supports the Button API subset the GUI uses: configure(text=/state=),
    cget(text=/state=)."""

    def __init__(self, master, text, command, kind, gui, padx, pady):
        super().__init__(master, bd=0, highlightthickness=0)
        self._gui = gui
        self._cmd = command
        self._kind = kind
        self._state = "normal"
        self._hover = False
        self._lbl = tk.Label(self, text=text, padx=padx, pady=pady,
                             font=gui._pal["font_ui_bold"], cursor="hand2")
        self._lbl.pack()
        self._lbl.bind("<Button-1>", self._on_click)
        self._lbl.bind("<Enter>", lambda e: self._set_hover(True))
        self._lbl.bind("<Leave>", lambda e: self._set_hover(False))
        self.refresh()

    def _on_click(self, _e=None):
        if self._state == "normal" and self._cmd:
            self._cmd()

    def _set_hover(self, on):
        self._hover = on
        self.refresh()

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        if "text" in kw:
            self._lbl.configure(text=kw.pop("text"))
        if "state" in kw:
            self._state = kw.pop("state")
        if kw:
            super().configure(**kw)
        self.refresh()

    def cget(self, key):
        if key == "text":
            return self._lbl.cget("text")
        if key == "state":
            return self._state
        return super().cget(key)

    def refresh(self):
        """Re-colour from the live palette (hover/disabled aware)."""
        pal = self._gui._pal
        prim = self._kind == "primary"
        bg = pal["cta_bg"] if prim else pal["ghost_bg"]
        fg = pal["cta_fg"] if prim else pal["ghost_fg"]
        ring = bg if prim else pal["ghost_border"]
        if self._state == "disabled":
            bg, fg, ring = pal["ghost_bg"], pal["dim"], pal["ghost_border"]
        elif self._hover:
            bg = pal["cta_hover"] if prim else pal["ghost_hover"]
        try:
            # NOTE: super(), not self.configure — configure() funnels
            # through refresh() and would recurse.
            super().configure(bg=bg, highlightbackground=ring,
                              highlightcolor=ring, highlightthickness=1)
            self._lbl.configure(bg=bg, fg=fg)
        except tk.TclError:
            pass


class PassportGUI(Window if HAS_TTKB else tk.Tk):
    def __init__(self, themename=None):
        if HAS_TTKB:
            # Construct on a valid stock theme first; the brand themes are
            # registered right after and _switch_theme() applies the
            # persisted choice (they cannot be registered before the Tk
            # root exists).
            super().__init__(themename="darkly")
        else:
            super().__init__()
        self._theme_name = "darkly"
        self.title("PASSPORT TEST BENCH 护照机综合测试台 v2.4")
        self.geometry("1280x780")
        self._set_window_icon()
        self._register_cyber_themes()
        self._build_menu()
        self._nfc_key_entries = {}
        self._nfc_dg1_lbs = {}
        self._nfc_prog_lbs = {}
        self._cta_buttons = []      # tk accent buttons (manually themed)
        self._build_header()
        # Restore the persisted theme (or the brand default) before any
        # tab widget exists; every builder then reads self._pal.
        self._switch_theme(themename or _load_theme_pref()
                           or ("cyber-dark" if HAS_TTKB else "darkly"))
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        # Tab order: Locator (1) -> OCR (2) -> NFC (3) -> MRZ (4) -> AC (5) -> face (6).
        self.tab_loc  = ttk.Frame(nb)
        self.tab_ocr  = ttk.Frame(nb)
        self.tab_nfc  = ttk.Frame(nb)
        self.tab_mrz  = ttk.Frame(nb)
        self.tab_ac   = ttk.Frame(nb)
        self.tab_face = ttk.Frame(nb)
        nb.add(self.tab_loc,  text="护照定位")
        nb.add(self.tab_ocr,  text="MRZ OCR 测试")
        nb.add(self.tab_nfc,  text="NFC 读卡")
        nb.add(self.tab_mrz,  text="MRZ 解码")
        nb.add(self.tab_ac,   text="防伪验证")
        nb.add(self.tab_face, text="照片比对")
        self._build_loc_tab()
        self._build_ocr_tab()
        self._build_nfc_tab()
        self._build_mrz_tab()
        self._build_ac_tab()
        self._build_face_tab()
        # Root window background: with no ttkbootstrap the Tk root defaults
        # to the macOS system window colour (light grey in light appearance)
        # which would flood the whole cockpit with a light backdrop.
        self.configure(bg=self._pal["bg"])
        self._refresh_tk_theme()
        self._apply_text_tag_palette()
        self.status = tk.StringVar(value=self._status_text())
        sbar = ttk.Frame(self)
        sbar.pack(fill="x", side="bottom")
        self._status_bar = sbar
        self.ocr_status_var = tk.StringVar(value="就绪")
        ttk.Label(sbar, textvariable=self.ocr_status_var, anchor="w",
                  padding=4).pack(side="left")
        self.ocr_progress = ttk.Progressbar(sbar, length=180,
                                            mode="determinate", maximum=100)
        self.ocr_progress.pack(side="left", padx=8, pady=2)
        # Visual-pass 2.0: status bar becomes a chip strip. One ttk.Label
        # per tool, each carries its own colour and dot, replacing the
        # previous flat "mrz_tool=OK ac_tool=OK ..." string.
        chip_frame = ttk.Frame(sbar)
        chip_frame.pack(side="right", fill="x")
        self._status_chip_labels = {}
        for label, path in (("MRZ", MRZ_TOOL), ("AC", AC_TOOL),
                            ("FACE", FACE_TOOL),
                            ("GEN", GEN_SAMP), ("NFC", NFC_TOOL)):
            ok = path.exists()
            lb = ttk.Label(
                chip_frame, text=f" ● {label} ",
                anchor="center", padding=(6, 2),
                font=self._pal["font_ui_bold"])
            lb.pack(side="right", padx=2, pady=2)
            self._status_chip_labels[label] = (lb, ok)
        # Keep the legacy self.status StringVar so any pre-existing binding
        # does not break; update _refresh_tk_theme() to mirror the chip.
        ttk.Label(sbar, textvariable=self.status, anchor="e",
                  padding=(6, 2)).pack(side="right")
        # Keep preview references so Tk doesn't garbage-collect them.
        self._preview_imgs = {}

    def _status_text(self):
        parts = []
        for label, path in (("mrz_tool", MRZ_TOOL), ("ac_tool", AC_TOOL),
                            ("face_tool", FACE_TOOL),
                            ("gen_sample", GEN_SAMP),
                            ("nfc_tool", NFC_TOOL)):
            parts.append(f"{label}={'OK' if path.exists() else 'MISSING'}")
        return "  ".join(parts)

    def _apply_theme_proof_styles(self):
        """Pin readable colors on input widgets. With ttkbootstrap the
        palette comes from the active theme (st.colors); otherwise fall back
        to a dark-aware field/text scheme (no ttkbootstrap in .venv-ctk:
        the old hardcoded #ffffff fallback left white Treeviews in dark
        mode)."""
        st = ttk.Style()
        dark = not self._is_light()
        try:
            c = st.colors
            if isinstance(c, dict):
                FIELD = c.get("inputbg", "#ffffff")
                TEXT = c.get("inputfg", "#111827")
                HEAD = c.get("active", "#e9edf3")
                SEL = c.get("primary", "#0b5cad")
            else:
                FIELD = getattr(c, "inputbg", "#ffffff")
                TEXT = getattr(c, "inputfg", "#111827")
                HEAD = getattr(c, "active", "#e9edf3")
                SEL = getattr(c, "primary", "#0b5cad")
        except AttributeError:
            if dark:
                FIELD, TEXT, HEAD, SEL = "#14181e", "#d5dae2", "#161F30", "#1F6FEB"
            else:
                FIELD, TEXT, HEAD, SEL = "#f4f6fa", "#1a2433", "#FFFFFF", "#0b5cad"
        # Pin palette-derived border so Labelframe/Card uses a faint line.
        border = (self._pal.get("border") if hasattr(self, "_pal") else "#3A3B3C")
        if not HAS_TTKB:
            # macOS aqua ignores Heading/radiobutton backgrounds; clam
            # lets every style element below render from our palette.
            try:
                st.theme_use("clam")
            except tk.TclError:
                pass
        try:
            st.configure("Data.Treeview", background=FIELD,
                         fieldbackground=FIELD, foreground=TEXT,
                         bordercolor=SEL, lightcolor=SEL, darkcolor=SEL,
                         rowheight=26)
            st.map("Data.Treeview",
                   background=[("selected", SEL)],
                   foreground=[("selected", "#ffffff")])
            st.configure("Data.Treeview.Heading", background=HEAD,
                         foreground=TEXT, padding=(6, 4), relief="flat")
            st.map("Data.Treeview.Heading", background=[("active", HEAD)])
            # Card-style container: replace the chunky native ridge with a
            # 1px flat border tinted to the palette border colour. Also pin
            # the container background: on macOS TLabelframe/TFrame default
            # to the SYSTEM window colour (light grey in light appearance),
            # which would punch big light panels into the dark cockpit.
            st.configure("TLabelframe", background=FIELD, bordercolor=border,
                         borderwidth=1, relief="solid")
            st.configure("TLabelframe.Label", background=FIELD,
                         foreground=TEXT, padding=(6, 2))
            st.configure("TFrame", background=FIELD)
            st.configure("TPanedwindow", background=FIELD,
                         sashwidth=3, sashpad=0, sashrelief="flat")
            st.configure("Field.TEntry", fieldbackground=FIELD,
                         foreground=TEXT, insertcolor=TEXT, bordercolor=border)
            st.map("Field.TEntry",
                   background=[("readonly", FIELD), ("disabled", HEAD)],
                   foreground=[("readonly", TEXT), ("disabled", TEXT)])
            st.configure("Field.TCombobox", fieldbackground=FIELD,
                         foreground=TEXT, bordercolor=border)
            st.map("Field.TCombobox",
                   fieldbackground=[("readonly", FIELD)],
                   foreground=[("readonly", TEXT)])
            # Notebook: theme the tab strip + content area so the deep
            # cockpit background is not broken by native light tabs.
            st.configure("TNotebook", background=FIELD, borderwidth=0)
            st.configure("TNotebook.Tab", background=HEAD, foreground=TEXT,
                         padding=(10, 4))
            st.map("TNotebook.Tab",
                   background=[("selected", FIELD), ("active", HEAD)],
                   foreground=[("selected", TEXT)])
            st.configure("TRadiobutton", background=FIELD, foreground=TEXT,
                         indicatorbackground=FIELD, bordercolor=border)
            st.map("TRadiobutton",
                   background=[("active", HEAD)],
                   foreground=[("active", TEXT), ("disabled", "#7c7c7c")],
                   indicatorbackground=[("selected", SEL)])
            st.configure("TCheckbutton", background=FIELD, foreground=TEXT,
                         indicatorbackground=FIELD, bordercolor=border)
            st.map("TCheckbutton",
                   background=[("active", HEAD)],
                   foreground=[("active", TEXT), ("disabled", "#7c7c7c")],
                   indicatorbackground=[("selected", SEL)])
            st.configure("TScrollbar", background=HEAD, troughcolor=FIELD,
                         arrowcolor=TEXT, bordercolor=border, relief="flat")
            st.configure("Vertical.TScrollbar", gripcount=0, background=HEAD,
                         troughcolor=FIELD, arrowcolor=TEXT, bordercolor=border,
                         relief="flat")
            st.configure("Horizontal.TScrollbar", gripcount=0, background=HEAD,
                         troughcolor=FIELD, arrowcolor=TEXT, bordercolor=border,
                         relief="flat")
            st.configure("TEntry", fieldbackground=FIELD, foreground=TEXT,
                         insertcolor=TEXT, bordercolor=border)
            st.configure("TCombobox", fieldbackground=FIELD, foreground=TEXT,
                         bordercolor=border)
        except Exception:
            pass

    def _register_cyber_themes(self):
        """Register the two brand themes with ttkbootstrap so every ttk
        widget class (Notebook, Button, Progressbar, Treeview, ...) is
        restyled by a single theme_use() call."""
        if not HAS_TTKB:
            return
        try:
            from ttkbootstrap.style import ThemeDefinition
        except Exception:
            return
        st = ttk.Style()
        for name, spec in CYBER_THEMES.items():
            if name in st.theme_names():
                continue
            try:
                st.register_theme(ThemeDefinition(name, spec["colors"],
                                                  mode=spec["mode"]))
            except Exception:
                pass

    def _switch_theme(self, name):
        # Friendly aliases so the menu / header toggle can offer
        # "light" / "dark" without memorising theme names.
        if name in ("dark", "dark_default"):
            name = "cyber-dark"
        elif name in ("light", "light_default"):
            name = "cyber-light"
        try:
            ttk.Style().theme_use(name)
        except tk.TclError:
            pass
        self._theme_name = name
        self._pal = self._theme_palette()
        self._apply_theme_proof_styles()
        self._apply_text_tag_palette()
        self._refresh_tk_theme()
        self._refresh_header()
        self._update_theme_menu_label()
        _save_theme_pref(name)

    def _is_light(self):
        spec = CYBER_THEMES.get(self._theme_name)
        if spec is not None:
            return spec["mode"] != "dark"
        return self._theme_name in ("cosmo", "journal", "flatly", "superhero")

    def _toggle_theme(self):
        self._switch_theme("cyber-dark" if self._is_light() else "cyber-light")

    def _update_theme_menu_label(self):
        viewm = getattr(self, "_viewm", None)
        if viewm is None:
            return
        label = "🌙  深色主题" if not self._is_light() else "☀️  浅色主题"
        try:
            viewm.entryconfig(self._theme_toggle_idx, label=label)
        except tk.TclError:
            pass

    # ---------------- brand header + theme toggle ----------------
    def _build_header(self):
        """Slim instrument-style header: letterspaced brand caption on
        the left, a segmented dark/light switch on the right. Purely
        additive — the tab layout below is untouched."""
        h = tk.Frame(self, bd=0, highlightthickness=0)
        h.pack(fill="x", side="top")
        self._header = h
        self._header_title = tk.Label(
            h, text="P A S S P O R T   T E S T   B E N C H",
            font=self._pal["font_title"] if hasattr(self, "_pal")
            else ("TkDefaultFont", 12, "bold"), anchor="w")
        self._header_title.pack(side="left", padx=(14, 8), pady=7)
        self._header_sub = tk.Label(h, text="护照机测试台 · MRZ / NFC / 防伪 / 人脸",
                                    anchor="w")
        self._header_sub.pack(side="left", pady=7)
        seg = tk.Frame(h, bd=0, highlightthickness=0)
        seg.pack(side="right", padx=10, pady=5)
        self._seg_frame = seg
        self._seg_toggle = tk.Label(seg, text="☀️ 浅色", cursor="hand2")
        self._seg_toggle.pack(side="left", padx=1, ipadx=10, ipady=2)
        self._seg_toggle.bind("<Button-1>", lambda e: self._toggle_theme())
        # Provisional palette: _switch_theme (which owns the final one)
        # runs after the header exists.
        if not hasattr(self, "_pal"):
            self._pal = self._theme_palette()
        self._refresh_header()

    def _refresh_header(self):
        h = getattr(self, "_header", None)
        if h is None:
            return
        pal = self._pal
        spec = CYBER_THEMES.get(self._theme_name)
        dark = spec["mode"] == "dark" if spec else \
            self._theme_name in ("darkly", "cyborg")
        try:
            h.configure(bg=pal["bg"])
            self._seg_frame.configure(bg=pal["bg"])
            self._header_title.configure(bg=pal["bg"], fg=pal["fg"])
            self._header_sub.configure(bg=pal["bg"], fg=pal["dim"])
            seg = self._seg_toggle
            seg.configure(text=("🌙 深色" if dark else "☀️ 浅色"),
                          bg=pal["cta_bg"], fg=pal["cta_fg"])
        except tk.TclError:
            pass

    # ---------------- themed accent buttons ----------------
    def _cta(self, parent, text, cmd, kind="primary", padx=14, pady=4):
        """Flat accent button (tk.Button so bg/fg render on macOS aqua)
        with hover feedback and automatic re-theming on switch."""
        pal = self._pal
        if kind == "primary":
            bg, fg = pal["cta_bg"], pal["cta_fg"]
        else:
            bg, fg = pal["ghost_bg"], pal["ghost_fg"]
        b = _ThemedButton(parent, text=text, command=cmd, kind=kind,
                          gui=self, padx=padx, pady=pady)
        b._cta_kind = kind
        self._cta_buttons.append(b)
        return b

    def _cta_recolor(self, b=None, hover=False):
        targets = [b] if b is not None else list(getattr(self, "_cta_buttons", []))
        for w in targets:
            if hasattr(w, "refresh"):
                if b is w:
                    w._hover = hover
                w.refresh()

    # ---------------- Text/Canvas tag palette ----------------
    def _apply_text_tag_palette(self):
        """Re-colour every Text-widget tag (diff highlight, PASS/FAIL
        rows, NFC trace) from the palette. Called at build time and on
        every theme switch."""
        pal = self._pal
        def cfg(w, tag, spec):
            if w is None:
                return
            try:
                kw = {"foreground": spec["fg"]}
                if spec["bg"]:
                    kw["background"] = spec["bg"]
                w.tag_configure(tag, **kw)
            except tk.TclError:
                pass
        cfg(getattr(self, "ocr_detail", None), "OK", pal["tag_ok"])
        cfg(getattr(self, "ocr_detail", None), "FAIL", pal["tag_fail"])
        cfg(getattr(self, "ocr_detail", None), "PASS", pal["tag_ok"])
        cfg(getattr(self, "ocr_detail", None), "SUSPECT", pal["tag_warn"])
        cfg(getattr(self, "ocr_detail", None), "REJECT", pal["tag_fail"])
        cfg(getattr(self, "ocr_summary", None), "row_ok", pal["tag_ok"])
        cfg(getattr(self, "ocr_summary", None), "row_fail", pal["tag_fail"])
        cfg(getattr(self, "ocr_diff_text", None), "match", pal["tag_match"])
        cfg(getattr(self, "ocr_diff_text", None), "mm", pal["tag_mm"])
        cfg(getattr(self, "ocr_diff_text", None), "okline", pal["tag_match"])
        con = pal["console"]
        for w, tag, spec in (
                (getattr(self, "ocr_diff_text", None), "hdr",
                 {"bg": None, "fg": con["pad"]}),
                (getattr(self, "ocr_diff_text", None), "lab",
                 {"bg": None, "fg": con["pad"]}),
                (getattr(self, "ocr_diff_text", None), "pad",
                 {"bg": None, "fg": con["pad"]}),
                (getattr(self, "ocr_diff_text", None), "error",
                 {"bg": con["error_bg"], "fg": con["error_fg"]})):
            cfg(w, tag, spec)
        cfg(getattr(self, "nfc_trace", None), "sw_ok",
            {"bg": None, "fg": "#3FB950"})
        cfg(getattr(self, "nfc_trace", None), "sw_bad",
            {"bg": None, "fg": "#F85149"})
        # AC verdict tags (same tag names reused across report widgets).
        for name in ("ac_out", "ac_report"):
            w = getattr(self, name, None)
            if w is None:
                continue
            cfg(w, "PASS", pal["tag_ok"])
            cfg(w, "SUSPECT", pal["tag_warn"])
            cfg(w, "REJECT", pal["tag_fail"])

    def _theme_palette(self):
        """Semantic color palette for the active theme (tk widgets)."""
        # ---- Cyber-tech brand themes: straight from the token table ----
        spec = CYBER_THEMES.get(self._theme_name)
        if HAS_TTKB and spec is not None:
            c, t = spec["colors"], spec["tk"]
        else:
            c = t = None
        if spec is not None:
            dark = spec["mode"] == "dark"
        else:
            dark = (not HAS_TTKB) or self._theme_name in ("darkly", "cyborg")
        # Mono font: Menlo on macOS, Consolas on Windows; falls back to TkFixedFont.
        if sys.platform == "darwin":
            mono_family = "Menlo"
        elif sys.platform.startswith("win"):
            mono_family = "Consolas"
        else:
            mono_family = "DejaVu Sans Mono"
        # UI font: SF Pro Text / PingFang SC on macOS, Segoe UI on Windows.
        if sys.platform == "darwin":
            ui_family = ".AppleSystemUIFont"
        elif sys.platform.startswith("win"):
            ui_family = "Segoe UI"
        else:
            ui_family = "TkDefaultFont"
        if t is not None:
            return {
                "bg": c["bg"], "fg": c["fg"],
                "field": c["inputbg"], "field_fg": c["inputfg"],
                "primary": c["primary"], "accent": t["cta_bg"],
                "ok": t["chip_ok"][0], "err": t["chip_fail"][0],
                "warn": t["chip_run"][0], "violet": t["violet"],
                "card": t["card"], "code": t["code"],
                "border": c["border"], "dim": t["dim"],
                "canvas": t["canvas"], "canvas_border": t["canvas_border"],
                "cta_bg": t["cta_bg"], "cta_fg": t["cta_fg"],
                "cta_hover": t["cta_hover"],
                "ghost_bg": t["ghost_bg"], "ghost_fg": t["ghost_fg"],
                "ghost_hover": t["ghost_hover"],
                "ghost_border": t["ghost_border"],
                "chip": {
                    "idle":  {"fg": t["chip_idle"][0],  "bg": t["chip_idle"][1]},
                    "ok":    {"fg": t["chip_ok"][0],    "bg": t["chip_ok"][1]},
                    "run":   {"fg": t["chip_run"][0],   "bg": t["chip_run"][1]},
                    "fail":  {"fg": t["chip_fail"][0],  "bg": t["chip_fail"][1]},
                    "dim":   {"fg": t["chip_dim"][0],   "bg": t["chip_dim"][1]},
                },
                "tag_ok":    {"bg": t["tag_ok"][0],    "fg": t["tag_ok"][1]},
                "tag_fail":  {"bg": t["tag_fail"][0],  "fg": t["tag_fail"][1]},
                "tag_warn":  {"bg": t["tag_warn"][0],  "fg": t["tag_warn"][1]},
                "tag_match": {"bg": None,              "fg": t["tag_match"][0]},
                "tag_mm":    {"bg": t["tag_mm"][0],    "fg": t["tag_mm"][1]},
                "tag_plain": {"bg": t["tag_plain"][1], "fg": t["tag_plain"][0]},
                "console": {
                    "bg": "#0B0F19" if dark else "#FFFFFF",
                    "fg": "#E2E8F0" if dark else "#1a2433",
                    "header": "#38BDF8" if dark else "#0b5cad",
                    "pad": "#475569" if dark else "#94a3b8",
                    "error_bg": "#EF4444" if dark else "#B42318",
                    "error_fg": "#FFFFFF",
                },
                "metric": {
                    "cases": "#E6EDF3" if dark else "#0b5cad",
                    "mean": "#38BDF8" if dark else "#0b8fb0",
                    "p95": "#38BDF8" if dark else "#0b5cad",
                    "pass": "#3FB950" if dark else "#027a48",
                    "l1": "#3FB950" if dark else "#027a48",
                    "l2": "#3FB950" if dark else "#027a48",
                    "full": "#3FB950" if dark else "#B45309",
                },
                "font_ui": (ui_family, 10),
                "font_ui_bold": (ui_family, 10, "bold"),
                "font_title": (ui_family, 12, "bold"),
                "font_mono": (mono_family, 10),
                "font_mono_sm": (mono_family, 9),
            }
        # ---- Legacy ttkbootstrap presets: derive from the theme ----
        st = ttk.Style()
        try:
            cc = st.colors
            if isinstance(cc, dict):
                bg = cc.get("bg", "#14181e" if dark else "#f4f6fa")
                fg = cc.get("fg", "#d5dae2" if dark else "#1a2433")
                field = cc.get("inputbg", "#1f242c" if dark else "#FFFFFF")
                field_fg = cc.get("inputfg", "#e6ebf2" if dark else "#1a2433")
                primary = cc.get("primary", "#0b5cad")
            else:
                bg = getattr(cc, "bg", "#14181e" if dark else "#f4f6fa")
                fg = getattr(cc, "fg", "#d5dae2" if dark else "#1a2433")
                field = getattr(cc, "inputbg", "#1f242c" if dark else "#FFFFFF")
                field_fg = getattr(cc, "inputfg", "#e6ebf2" if dark else "#1a2433")
                primary = getattr(cc, "primary", "#0b5cad")
        except AttributeError:
            if dark:
                bg, fg, field, field_fg, primary = ("#14181e", "#d5dae2",
                                                    "#1f242c", "#e6ebf2", "#0b5cad")
            else:
                bg, fg, field, field_fg, primary = ("#f4f6fa", "#1a2433",
                                                    "#FFFFFF", "#1a2433", "#0b5cad")
        try:
            cc = ttk.Style().colors
            active_c = (cc.get("active", "#e9edf3") if isinstance(cc, dict)
                        else getattr(cc, "active", "#e9edf3"))
        except AttributeError:
            active_c = "#e9edf3"
        accent = "#00e5ff" if dark else primary
        # Visual-pass 2.0 semantic tokens, kept for the stock presets.
        if dark:
            border = "#21262D"
            card = "#121620"
            code = "#0B0F19"
            chips = {"idle": ("#00F5FF", "#0f2a33"), "ok": ("#10B981", "#0f3325"),
                     "run": ("#b388ff", "#251a3d"), "fail": ("#EF4444", "#3a1620"),
                     "dim": ("#6a737d", "#1a1d22")}
            tags = {"ok": ("#0d3a2a", "#7eeac2"), "fail": ("#43131f", "#ffaabb"),
                    "warn": ("#3d2f08", "#F59E0B"), "match": "#10B981",
                    "mm": ("#EF4444", "#FFFFFF"), "plain": ("#c9d4e0", "#0a0e14")}
        else:
            border = "#D0D5DD"
            card = "#F2F4F7"
            code = "#FFFFFF"
            chips = {"idle": ("#0b5cad", "#E6F0FA"), "ok": ("#027a48", "#D1FADF"),
                     "run": ("#6c2bd9", "#EDE4FF"), "fail": ("#B42318", "#FEE4E2"),
                     "dim": ("#666666", "#EAECF0")}
            tags = {"ok": ("#c8e6c9", "#1b5e20"), "fail": ("#ffcdd2", "#b71c1c"),
                    "warn": ("#fff3cd", "#8a6d00"), "match": "#027a48",
                    "mm": ("#b42318", "#ffe4e6"), "plain": ("#1a2433", "#ffffff")}
        return {
            "bg": bg, "fg": fg, "field": field, "field_fg": field_fg,
            "primary": primary, "accent": accent,
            "ok": chips["ok"][0], "err": chips["fail"][0],
            "warn": chips["run"][0], "violet": chips["run"][0],
            "card": card, "code": code, "border": border,
            "dim": "#9aa4b2" if dark else "#5c6773",
            "canvas": card, "canvas_border": border,
            "cta_bg": "#1F6FEB" if dark else primary, "cta_fg": "#ffffff",
            "cta_hover": "#3a8bff" if dark else "#2b7ed6",
            "ghost_bg": "#30363D" if dark else field,
            "ghost_fg": "#C9D1D9" if dark else accent,
            "ghost_hover": "#3a434d" if dark else active_c,
            "ghost_border": "#30363D" if dark else border,
            "chip": {k: {"fg": v[0], "bg": v[1]} for k, v in chips.items()},
            "tag_ok":    {"bg": tags["ok"][0],    "fg": tags["ok"][1]},
            "tag_fail":  {"bg": tags["fail"][0],  "fg": tags["fail"][1]},
            "tag_warn":  {"bg": tags["warn"][0],  "fg": tags["warn"][1]},
            "tag_match": {"bg": None,             "fg": tags["match"]},
            "tag_mm":    {"bg": tags["mm"][0],    "fg": tags["mm"][1]},
            "tag_plain": {"bg": tags["plain"][1], "fg": tags["plain"][0]},
            "console": {
                "bg": "#0B0F19" if dark else "#FFFFFF",
                "fg": "#E2E8F0" if dark else "#1a2433",
                "header": "#38BDF8" if dark else "#0b5cad",
                "pad": "#475569" if dark else "#94a3b8",
                "error_bg": "#EF4444" if dark else "#B42318",
                "error_fg": "#FFFFFF",
            },
            "metric": {
                "cases": "#E6EDF3" if dark else "#0b5cad",
                "mean": "#38BDF8" if dark else "#0b8fb0",
                "p95": "#38BDF8" if dark else "#0b5cad",
                "pass": "#3FB950" if dark else "#027a48",
                "l1": "#3FB950" if dark else "#027a48",
                "l2": "#3FB950" if dark else "#027a48",
                "full": "#3FB950" if dark else "#B45309",
            },
            "font_ui": (ui_family, 10),
            "font_ui_bold": (ui_family, 10, "bold"),
            "font_title": (ui_family, 12, "bold"),
            "font_mono": (mono_family, 10),
            "font_mono_sm": (mono_family, 9),
        }

    def _refresh_tk_theme(self):
        """Re-color tk (non-ttk) widgets to match the active theme."""
        pal = self._pal
        chip = pal["chip"]
        # Text/code widgets: bg = field, fg = field_fg (already contrast-safe).
        for n in ("loc_info", "nfc_inspect", "nfc_adv_text",
                  "mrz_text", "mrz_out", "ac_out", "face_out"):
            w = getattr(self, n, None)
            if w is not None:
                try:
                    w.configure(bg=pal["field"], fg=pal["field_fg"],
                                highlightbackground=pal["border"],
                                highlightcolor=pal["accent"],
                                highlightthickness=1)
                except tk.TclError:
                    pass
        w = getattr(self, "loc_canvas", None)
        if w is not None:
            try:
                w.configure(bg=pal["canvas"],
                            highlightbackground=pal["canvas_border"],
                            highlightcolor=pal["accent"])
            except tk.TclError:
                pass
        # ---- Visual-pass 2.0: chip-style badges need explicit bg/fg ----
        # Right-top corner empty white square fix: nfc_badge starts blank but
        # was inheriting the platform default bg; pin to window bg + dim chip.
        w = getattr(self, "nfc_badge", None)
        if w is not None:
            try:
                w.configure(bg=pal["bg"], fg=chip["dim"]["fg"],
                            font=pal["font_ui_bold"])
            except tk.TclError:
                pass
        # BAC state-machine chips: reset every known badge to the dim chip
        # palette; _nfc_show() will recolour them on the next render pass.
        for _title, _var, lb in getattr(self, "_nfc_states", []):
            try:
                lb.configure(bg=pal["bg"], fg=chip["idle"]["fg"],
                             font=pal["font_ui_bold"])
            except tk.TclError:
                pass
        # DG1 / SOD / read-progress labels live in dicts of tk.Label refs.
        for d in ("_nfc_dg1_lbs", "_nfc_prog_lbs"):
            dd = getattr(self, d, None) or {}
            for w in dd.values():
                try:
                    w.configure(bg=pal["bg"], fg=pal["fg"],
                                font=pal["font_mono_sm"])
                except tk.TclError:
                    pass
        for attr in ("_nfc_sod_sha_lb", "_nfc_sod_rsa_lb"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.configure(bg=pal["bg"], fg=pal["fg"],
                                font=pal["font_mono_sm"])
                except tk.TclError:
                    pass
        # Status-bar tool chips: OK=green dot, MISSING=red dot, both with
        # contrast-safe chip backgrounds that survive light/dark themes.
        for _label, (lb, ok) in getattr(self, "_status_chip_labels", {}).items():
            try:
                if ok:
                    lb.configure(foreground=chip["ok"]["fg"],
                                 background=chip["ok"]["bg"])
                else:
                    lb.configure(foreground=chip["fail"]["fg"],
                                 background=chip["fail"]["bg"])
            except tk.TclError:
                pass
        # Themed CTA buttons (tk.Button accents need manual recolour).
        self._cta_recolor()
        w = getattr(self, "ocr_preview_label", None)
        if w is not None:
            try:
                w.configure(background=pal["canvas"],
                            highlightbackground=pal["border"])
            except tk.TclError:
                pass
        ml = getattr(self, "ocr_preview_meta_lb", None)
        if ml is not None:
            try:
                ml.configure(bg=pal["card"], fg=pal["fg"])
            except tk.TclError:
                pass
        for d in ("_nfc_key_entries", "_nfc_dg1_lbs", "_nfc_prog_lbs"):
            dd = getattr(self, d, None) or {}
            for w in dd.values():
                try:
                    w.configure(bg=pal["field"], fg=pal["field_fg"],
                                readonlybackground=pal["field"])
                except tk.TclError:
                    pass
        # Diff terminal (cockpit console) — re-theme from palette.
        con = pal["console"]
        for w, bg, fg in ((getattr(self, "_diff_term", None), pal["card"], None),
                          (getattr(self, "_diff_tbar", None), pal["card"], con["header"])):
            if w is not None:
                try:
                    kw = {"bg": bg}
                    if fg:
                        kw["fg"] = fg
                    w.configure(**kw)
                except tk.TclError:
                    pass
        t = getattr(self, "_diff_term", None)
        if t is not None:
            try:
                t.configure(highlightbackground=pal["border"])
            except tk.TclError:
                pass
        w = getattr(self, "ocr_diff_text", None)
        if w is not None:
            try:
                w.configure(bg=con["bg"], fg=con["fg"],
                            highlightbackground=pal["border"])
            except tk.TclError:
                pass
        # KPI metric cards — re-theme frames + labels.
        for cell, ttl, val, role in getattr(self, "_metric_cells", []):
            try:
                cell.configure(bg=pal["card"], highlightbackground=pal["border"])
                ttl.configure(foreground=pal["dim"])
                val.configure(foreground=pal["metric"][role])
            except tk.TclError:
                pass
        # Root window + loc-page MRZ status chip — set once at construction
        # is not enough; re-theme them on every switch so no large light
        # (or dark) surface survives a theme toggle.
        try:
            self.configure(bg=pal["bg"])
        except tk.TclError:
            pass
        w = getattr(self, "loc_mrz_chip", None)
        if w is not None:
            try:
                w.configure(bg=chip["dim"]["bg"], fg=chip["dim"]["fg"])
            except tk.TclError:
                pass

    def _set_window_icon(self):
        """Use the app icon (tests/assets/app_icon.png) for the window
        title bar / dock. Falls back gracefully if the asset is missing."""
        try:
            icon_path = ROOT / "tests" / "assets" / "app_icon.png"
            if icon_path.exists():
                self.iconphoto(True,
                               tk.PhotoImage(file=str(icon_path)),
                               tk.PhotoImage(file=str(
                                   ROOT / "tests" / "assets" / "app_icon_64.png")),
                               tk.PhotoImage(file=str(
                                   ROOT / "tests" / "assets" / "app_icon_32.png")))
        except tk.TclError:
            pass  # headless environment

    def _build_menu(self):
        menubar = tk.Menu(self)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="生成样本图片", command=self._menu_gen_sample)
        filem.add_separator()
        filem.add_command(label="退出", command=self.destroy)
        menubar.add_cascade(label="文件", menu=filem)
        viewm = tk.Menu(menubar, tearoff=0)
        self._viewm = viewm
        self._theme_toggle_idx = 0
        viewm.add_command(label="☀️  浅色主题", command=self._toggle_theme)
        viewm.add_separator()
        advm = tk.Menu(viewm, tearoff=0)
        for name in ("darkly", "cyborg", "superhero", "cosmo",
                     "journal", "flatly"):
            advm.add_command(label=name,
                             command=lambda n=name: self._switch_theme(n))
        viewm.add_cascade(label="其它 (ttkbootstrap)", menu=advm)
        menubar.add_cascade(label="视图", menu=viewm)
        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="关于",
                          command=lambda: messagebox.showinfo(
                              "关于",
                              "护照机测试机 GUI\n"
                              "驱动 4 个纯 C 模块:\n"
                              "  1_mrz_decode\n"
                              "  2_anticounterfeit\n"
                              "  3_face_recognition\n"
                              "  4_nfc_reader\n"
                              "  6_mrz_ocr (CNN)\n"
                              "  7_security"))
        menubar.add_cascade(label="帮助", menu=helpm)
        self.config(menu=menubar)

    # ---------------- OCR tab ----------------
    def _build_ocr_tab(self):
        pal = self._pal
        from pathlib import Path as _Path
        import sys as _sys
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        f = self.tab_ocr

        # ---- Control card: methods | filter | search | primary CTA ----
        ctrl = ttk.LabelFrame(f, text="控制面板")
        ctrl.pack(fill="x", padx=4, pady=(4, 0))
        row1 = ttk.Frame(ctrl); row1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(row1, text="模式:").pack(side="left")
        self.ocr_photo_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="图片直接识别（assets/pic 护照照片）",
                        variable=self.ocr_photo_mode,
                        command=self._ocr_mode_switch).pack(side="left", padx=2)
        ttk.Separator(row1, orient="vertical").pack(side="left",
                                                    fill="y", padx=6)
        ttk.Label(row1, text="方案:").pack(side="left")
        self.ocr_methods = tk.StringVar(value=",".join(METHOD_ORDER))
        for key in METHOD_ORDER:
            ttk.Radiobutton(row1, text=METHOD_LABELS[key],
                            variable=self.ocr_methods,
                            value=key).pack(side="left", padx=2)
        self.ocr_method_both = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="同时跑全部方案",
                        variable=self.ocr_method_both,
                        command=self._ocr_method_toggle).pack(side="left", padx=6)
        ttk.Separator(row1, orient="vertical").pack(side="left",
                                                    fill="y", padx=8)
        ttk.Label(row1, text="过滤:").pack(side="left")
        self.ocr_filter_var = tk.StringVar(value="all")
        for label, key in [("全部", "all"), ("干净", "clean"), ("带噪", "noisy")]:
            ttk.Radiobutton(row1, text=label, variable=self.ocr_filter_var,
                            value=key,
                            command=self._ocr_apply_filter).pack(side="left", padx=2)
        style = ttk.Style()
        if not style.theme_use():
            style.theme_use("default")
        # Accent CTA: brand-filled flat button via the themed factory
        # (tk.Button, not ttk, so bg/fg render on macOS aqua).
        btn_frame = ttk.Frame(row1)
        btn_frame.pack(side="right", padx=4)
        self.ocr_run_btn = self._cta(
            btn_frame, "⚡ 开始测试", self._ocr_run, padx=12, pady=4)
        self.ocr_run_btn.pack(side="right", padx=4, pady=2)
        # Direct-image recognition: pick a JPG/PNG file, render its
        # thumbnail in the right pane, then call mrz_ocr_tool and
        # mrz_tool and stream the result through the diff pane. This
        # is the "I just want to OCR one image" path; the corpus bench
        # stays the "stress test many cases" path.
        self.ocr_pick_btn = self._cta(
            btn_frame, "🖼 直接识别图片", self._ocr_pick_and_recognize, kind="ghost")
        self.ocr_pick_btn.pack(side="right", padx=4, pady=2)

        row2 = ttk.Frame(ctrl); row2.pack(fill="x", padx=8, pady=(0, 6))
        self._cta(row2, "全选",
                  lambda: self._ocr_select_all(True), kind="ghost").pack(side="left", padx=2)
        self._cta(row2, "反选",
                  lambda: self._ocr_select_all(False), kind="ghost").pack(side="left", padx=2)
        self._cta(row2, "清空选择",
                  self._ocr_clear_sel, kind="ghost").pack(side="left", padx=2)
        ttk.Label(row2, text="搜索案例 ID:").pack(side="left", padx=(14, 2))
        self.ocr_search_var = tk.StringVar()
        search = ttk.Entry(row2, textvariable=self.ocr_search_var, width=30,
                           style="Field.TEntry")
        search.pack(side="left")
        search.bind("<KeyRelease>", self._ocr_apply_search)

        # ---- Body: 3-pane splitter (case list | metrics+summary+detail | sample) ----
        body = ttk.PanedWindow(f, orient="horizontal")
        body.pack(fill="both", expand=True, padx=4, pady=4)
        self._ocr_body = body

        # Left = case list.
        left = ttk.Frame(body); body.add(left, weight=1)
        ttk.Label(left, text="案例集列表（点击查看原图，多选后开始测试）").pack(anchor="w")
        self.ocr_case_list = ttk.Treeview(left,
            columns=("id", "scale", "noise", "skew"),
            show="headings", selectmode="extended", height=22,
            style="Data.Treeview")
        for c, w in [("id", 240), ("scale", 70), ("noise", 70), ("skew", 70)]:
            self.ocr_case_list.heading(c, text=c)
            self.ocr_case_list.column(c, width=w, anchor="w")
        ysb = ttk.Scrollbar(left, orient="vertical",
                            command=self.ocr_case_list.yview)
        self.ocr_case_list.configure(yscrollcommand=ysb.set)
        self.ocr_case_list.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.ocr_case_list.bind("<<TreeviewSelect>>", self._ocr_on_case_select)
        # Double-click opens the full-resolution viewer; single-click
        # keeps the existing thumbnail preview in the right pane.
        self.ocr_case_list.bind("<Double-Button-1>", self._ocr_open_original)
        # Photo mode: single-click on a list row immediately runs
        # process_image() on that single photo, so the user gets the
        # full pipeline (locate -> OCR -> decode -> grade) without
        # having to also press ▶ 开始识别.
        self.ocr_case_list.bind("<Button-1>", self._ocr_on_list_click)

        # Middle = metrics card + summary + detail.
        middle = ttk.Frame(body); body.add(middle, weight=2)
        self._build_ocr_metrics(middle)
        ttk.Label(middle, text="方案指标对比").pack(anchor="w")
        self.ocr_summary = ttk.Treeview(middle,
            columns=("method", "n", "ok", "okp", "ms_avg",
                     "l1_acc", "l2_acc", "full_match"),
            show="headings", height=3)
        for c, anc in [("method", "w"), ("n", "center"), ("ok", "center"),
                       ("okp", "center"), ("ms_avg", "center"),
                       ("l1_acc", "center"), ("l2_acc", "center"),
                       ("full_match", "center")]:
            self.ocr_summary.heading(c, text=c)
            self.ocr_summary.column(c, width=80, anchor=anc)
        # Theme-proof: Data.Treeview style is set globally in __init__.
        self.ocr_summary.configure(style="Data.Treeview")
        self.ocr_summary.tag_configure("row_ok",
                                      background="#e8f5e9",
                                      foreground="#0f5132")
        self.ocr_summary.tag_configure("row_fail",
                                      background="#ffebee",
                                      foreground="#a00d20")
        self.ocr_summary.pack(fill="x")

        ttk.Label(middle, text="详细结果（点击行 → 右侧单样本透视）").pack(anchor="w", pady=(6, 0))
        self.ocr_detail = ttk.Treeview(middle,
            columns=("id", "method", "ok", "ms", "l1", "l2"),
            show="headings", height=14, style="Data.Treeview")
        for c, anc in [("id", "w"), ("method", "w"), ("ok", "center"),
                       ("ms", "center"), ("l1", "center"), ("l2", "center")]:
            self.ocr_detail.heading(c, text=c)
            if c == "id":
                self.ocr_detail.column(c, width=120, anchor=anc)
            elif c == "method":
                self.ocr_detail.column(c, width=90, anchor=anc)
            elif c in ("l1", "l2"):
                self.ocr_detail.column(c, width=140, anchor=anc, stretch=True)
            else:
                self.ocr_detail.column(c, width=56, anchor=anc)
        self.ocr_detail.tag_configure("OK", background="#c8e6c9",
                                      foreground=self._pal["ok"])
        self.ocr_detail.tag_configure("FAIL", background="#ffcdd2",
                                      foreground=self._pal["err"])
        ysb2 = ttk.Scrollbar(middle, orient="vertical",
                             command=self.ocr_detail.yview)
        xsb2 = ttk.Scrollbar(middle, orient="horizontal",
                             command=self.ocr_detail.xview)
        self.ocr_detail.configure(yscrollcommand=ysb2.set,
                                  xscrollcommand=xsb2.set)
        self.ocr_detail.pack(side="top", fill="both", expand=True)
        ysb2.pack(side="right", fill="y")
        xsb2.pack(side="bottom", fill="x")
        self.ocr_detail.bind("<<TreeviewSelect>>", self._ocr_on_detail_select)
        self.ocr_detail.bind("<Double-Button-1>", self._ocr_open_original_from_detail)
        self._ocr_detail_map = {}

        # Right = single-sample perspective (image + char-level diff).
        preview = ttk.Frame(body); body.add(preview, weight=1)
        ttk.Label(preview, text="单样本透视").pack(anchor="w")
        self.ocr_preview_meta = tk.StringVar(value="（在左侧选择案例，或点击明细行）")
        self.ocr_preview_meta_lb = tk.Label(
            preview, textvariable=self.ocr_preview_meta,
            bg=pal["card"], fg=pal["fg"], wraplength=380, justify="left",
            anchor="w", font=("TkDefaultFont", 9))
        self.ocr_preview_meta_lb.pack(anchor="w", fill="x")
        self._ocr_meta_fullpath = ""
        self._ocr_meta_tip = _Tooltip(self.ocr_preview_meta_lb)
        self.ocr_preview_label = tk.Label(preview, bg=pal["canvas"],
                                          fg=pal["fg"], anchor="center",
                                          highlightthickness=1,
                                          highlightbackground=pal["border"])
        self.ocr_preview_label.pack(fill="both", expand=True)
        self.ocr_preview_label.bind("<Configure>", self._update_preview_image)
        ttk.Label(preview, text="识别结果 · 字符级比对 (GT vs Pred)").pack(anchor="w")
        con = pal["console"]
        term = tk.Frame(preview, bg=pal["card"], highlightthickness=1,
                        highlightbackground=pal["border"])
        term.pack(fill="both", expand=True)
        self._diff_term = term
        tbar = tk.Label(term, text="  DIGITAL COMPARATOR // 字符级检视",
                        font=("Menlo", 9, "bold"), bg=pal["card"],
                        fg=con["header"], anchor="w", pady=3)
        tbar.pack(fill="x")
        self._diff_tbar = tbar
        self.ocr_diff_text = tk.Text(term, height=11, width=48,
                                     font=("Menlo", 10), wrap="none",
                                     bg=con["bg"], fg=con["fg"], relief="flat",
                                     padx=8, pady=6,
                                     highlightthickness=1,
                                     highlightbackground=pal["border"])
        self.ocr_diff_text.tag_configure("hdr", font=("Menlo", 9, "bold"),
                                         foreground=con["pad"])
        self.ocr_diff_text.tag_configure("lab", foreground=con["pad"])
        self.ocr_diff_text.tag_configure("match", foreground=pal["tag_match"]["fg"],
                                         font=("Menlo", 10, "bold"))
        self.ocr_diff_text.tag_configure("pad", foreground=con["pad"])
        self.ocr_diff_text.tag_configure("mm", background=con["error_bg"],
                                         foreground=con["error_fg"],
                                         font=("Menlo", 10, "bold"))
        self.ocr_diff_text.tag_configure("error", background=con["error_bg"],
                                         foreground=con["error_fg"],
                                         font=("Menlo", 10, "bold"))
        self.ocr_diff_text.tag_configure("okline",
                                         foreground=pal["tag_match"]["fg"],
                                         font=("Menlo", 10))
        d_y = ttk.Scrollbar(term, orient="vertical",
                            command=self.ocr_diff_text.yview)
        d_x = ttk.Scrollbar(term, orient="horizontal",
                            command=self.ocr_diff_text.xview)
        self.ocr_diff_text.configure(yscrollcommand=d_y.set,
                                     xscrollcommand=d_x.set)
        self.ocr_diff_text.pack(side="left", fill="both", expand=True)
        d_y.pack(side="right", fill="y")
        d_x.pack(side="bottom", fill="x")

        # Load corpus.
        self._ocr_corpus = {"records": []}
        self._ocr_case_order = []
        self._ocr_load_corpus()
        self.after(100, self._ocr_set_sashes)

    def _ocr_load_corpus(self):
        """(Re)fill the case list with corpus cases (batch-test mode)."""
        import sys as _sys
        from pathlib import Path as _Path
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        for iid in self.ocr_case_list.get_children():
            self.ocr_case_list.delete(iid)
        self._ocr_case_order = []
        corpus_path = _Path(CORPUS_JSON)
        self._ocr_corpus = {"records": []}
        if corpus_path.exists():
            data = json.loads(corpus_path.read_text())
            for rec in data["records"]:
                tag = "noisy" if rec["noise"] > 0 else "clean"
                iid = rec["id"]
                self.ocr_case_list.insert("", "end", iid=iid,
                    values=(rec["id"], rec["scale"], rec["noise"], rec["skew"]),
                    tags=(tag,))
                self._ocr_case_order.append(iid)
            self._ocr_corpus = data

    def _ocr_mode_switch(self):
        """Toggle between corpus batch-test and passport-photo recognition."""
        if self.ocr_photo_mode.get():
            self.ocr_run_btn.configure(text="⏳ 识别中…")
            from passport_pipeline import PIC_DIR as _PIC_DIR
            for iid in self.ocr_case_list.get_children():
                self.ocr_case_list.delete(iid)
            self._ocr_case_order = []
            pics = [p.name for p in sorted(_PIC_DIR.glob("*"))
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm")]
            for name in pics:
                self.ocr_case_list.insert("", "end", iid=name,
                                          values=(name, "", "", ""),
                                          tags=("clean",))
                self._ocr_case_order.append(name)
        else:
            self.ocr_run_btn.configure(text="⚡ 开始测试")
            self._ocr_load_corpus()

    def _build_loc_tab(self):
        """Passport image OCR region locator: photo / data / MRZ (standalone tab).
        Built from tests/passport_locator.py (pure PIL+numpy).
        """
        pal = self._pal
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        from passport_locator import locate_regions, annotate  # type: ignore
        self._loc_locator = locate_regions
        self._loc_annotate = annotate

        f = self.tab_loc

        top = ttk.LabelFrame(f, text="护照图片 OCR 区域定位（照片 / 数据区 / MRZ 区）")
        top.pack(fill="x", padx=4, pady=(4, 0))

        row = ttk.Frame(top); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="样本:").pack(side="left")
        self.loc_sample = tk.StringVar()
        from passport_pipeline import PIC_DIR as _PIC  # type: ignore
        _pic_names = sorted(p.name for p in _PIC.glob("*")
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm"))
        self.loc_samples = [
            "合成护照样本 (TD3 整页)",
            "corpus MRZ 条带 (img_0001_b0_v0)",
        ] + [f"assets/pic: {n}" for n in _pic_names]
        cb = ttk.Combobox(row, textvariable=self.loc_sample,
                          values=self.loc_samples, state="readonly", width=30,
                          style="Field.TCombobox")
        cb.pack(side="left", padx=2)
        cb.current(0)
        cb.bind("<<ComboboxSelected>>", self._loc_on_sample_selected)
        self.loc_sample_path = tk.StringVar()
        ttk.Button(row, text="▤ 浏览文件…", command=self._loc_pick).pack(side="left", padx=4)
        self.loc_run_btn = self._cta(row, "⚡ 定位区域", self._loc_run,
                                     padx=12, pady=3)
        self.loc_run_btn.pack(side="left", padx=4)
        # MRZ-only quick action: locate MRZ band, crop it, feed it to
        # mrz_tool, and dump the parsed fields + check digits on the
        # right-hand info pane. Ghost styling so the primary CTA stays
        # the single strongest affordance in the row.
        self.loc_mrz_btn = self._cta(row, "🆔 MRZ 识别", self._loc_mrz_recognize,
                                     kind="ghost", padx=12, pady=3)
        self.loc_mrz_btn.pack(side="left", padx=4)
        self.loc_status = ttk.Label(row, text="")
        self.loc_status.pack(side="left", padx=8)
        # MRZ result chip: PASS green / FAIL red, sits at the top of the
        # info pane so it stays in the user's natural reading path.
        self._loc_mrz_chip_var = tk.StringVar(value="  IDLE  ")

        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=4, pady=(4, 6))
        self.loc_canvas = tk.Canvas(body, width=760, height=440,
                                    bg=pal["canvas"], highlightthickness=1,
                                    highlightbackground=pal["canvas_border"],
                                    highlightcolor=pal["accent"])
        self.loc_canvas.pack(side="left", fill="both", expand=True)
        # Side panel: MRZ result chip at the top, detail Text below.
        side = ttk.Frame(body)
        side.pack(side="left", padx=(6, 0), fill="y")
        pal = self._pal
        chip = pal["chip"]["dim"]
        self.loc_mrz_chip = tk.Label(
            side, textvariable=self._loc_mrz_chip_var,
            bg=chip["bg"], fg=chip["fg"],
            font=pal["font_ui_bold"], padx=10, pady=3)
        self.loc_mrz_chip.pack(anchor="w", pady=(0, 4))
        self.loc_info = tk.Text(side, width=52, height=20, font=("Menlo", 9),
                                state="disabled", wrap="none",
                                bg=pal["field"], fg=pal["field_fg"],
                                relief="flat", bd=0, highlightthickness=1,
                                highlightbackground=pal["border"],
                                highlightcolor=pal["accent"])
        self.loc_info.pack(fill="both", expand=True)

    def _loc_pick(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.png *.ppm *.bmp *.jpg"), ("All", "*.*")])
        if path:
            self.loc_sample_path.set(path)
            self.loc_sample.set(f"自定义: {os.path.basename(path)}")
            try:
                self._loc_display_image(path, "已选择文件，点“定位区域”开始定位")
            except Exception as e:
                self.loc_status.config(text=f"加载失败: {e}", foreground=self._pal["err"])

    def _loc_resolve_path(self):
        path = self.loc_sample_path.get()
        if path and os.path.exists(path):
            return path
        key = self.loc_sample.get()
        if key.startswith("corpus"):
            from ocr_bench_runner import CORPUS_JSON  # type: ignore
            return str(Path(CORPUS_JSON).parent / "img_0001_b0_v0.ppm")
        if key.startswith("assets/pic:"):
            from passport_pipeline import PIC_DIR as _PIC  # type: ignore
            name = key.split(":", 1)[1].strip()
            p = _PIC / name
            if p.exists():
                return str(p)
        return str(SAMPLE_PASSPORT_PNG)

    def _loc_display_image(self, path, note=""):
        """Show the (unannotated) image in the locator canvas."""
        from PIL import Image, ImageTk
        img = Image.open(path).convert("RGB")
        disp = img.copy()
        disp.thumbnail((max(self.loc_canvas.winfo_width() - 4, 320),
                        max(self.loc_canvas.winfo_height() - 4, 220)))
        self._loc_imgtk = ImageTk.PhotoImage(disp)
        self.loc_canvas.delete("all")
        self.loc_canvas.create_image(2, 2, image=self._loc_imgtk, anchor="nw")
        self.loc_canvas.create_text(4, self.loc_canvas.winfo_height() - 4,
                                    text=f"{img.width}×{img.height}",
                                    anchor="sw", fill=self._pal["dim"],
                                    font=("Menlo", 8))
        if note:
            self.loc_status.config(text=note, foreground=self._pal["dim"])

    def _loc_on_sample_selected(self, _evt=None):
        try:
            path = self._loc_resolve_path()
            self._loc_display_image(path, "已显示样本图片，点“定位区域”开始定位")
        except Exception as e:
            self.loc_status.config(text=f"加载失败: {e}", foreground=self._pal["err"])

    def _loc_set_text(self, txt):
        self.loc_info.configure(state="normal")
        self.loc_info.delete("1.0", "end")
        self.loc_info.insert("1.0", txt)
        self.loc_info.configure(state="disabled")

    def _loc_run(self):
        try:
            from PIL import Image, ImageTk
            path = self._loc_resolve_path()
            img = Image.open(path).convert("RGB")
            loc = self._loc_locator(img)
            disp = self._loc_annotate(img, loc, scale=1.0)
            disp.thumbnail((max(self.loc_canvas.winfo_width() - 4, 320),
                            max(self.loc_canvas.winfo_height() - 4, 220)))
            self._loc_imgtk = ImageTk.PhotoImage(disp)
            self.loc_canvas.delete("all")
            self.loc_canvas.create_image(2, 2, image=self._loc_imgtk, anchor="nw")
            self.loc_canvas.create_text(4, self.loc_canvas.winfo_height() - 4,
                                        text=f"{img.width}×{img.height}",
                                        anchor="sw", fill=self._pal["dim"],
                                    font=("Menlo", 8))
            names = {"photo": "照片区", "data": "数据区", "mrz": "MRZ 区"}
            txt = [f"样本: {os.path.basename(path)}  ({img.width}×{img.height})", ""]
            ok = 0
            for k, name in names.items():
                r = loc.get(k)
                if r:
                    ok += 1
                    txt.append(f"[{name}]  x={r['x']} y={r['y']} w={r['w']} h={r['h']}"
                               f"   置信度 {r['confidence']:.2f}")
                    txt.append(f"    依据: {r['evidence']}")
                else:
                    txt.append(f"[{name}]  未定位")
            self.loc_status.config(text=f"完成：{ok}/3 区域", foreground=self._pal["ok"])
            self._loc_set_text("\n".join(txt))
        except Exception as e:
            self.loc_status.config(text=f"定位失败: {e}", foreground=self._pal["err"])
            self._loc_set_text(f"错误: {e}")

    @staticmethod
    def _synth_hint(conf1, conf2, decode_ok):
        """Return a warning if the OCR confidence is low *and* mrz_tool
        decode rejected the result. Both signals together are the only
        reliable indicator that the image is genuinely problematic; a
        confident result is always trusted even if the character set
        happens to look uniform."""
        if decode_ok:
            return ""
        if max(conf1, conf2) >= 70:
            return ""
        return (f"[!] OCR 置信率低（conf1={conf1}%, conf2={conf2}%）"
                f"且 mrz_tool decode 拒绝。这是 OCR 模型在该样本上的局限，"
                f"不是 GUI bug。常见原因：合成样本 / 低分辨率照片 /"
                f"MRZ 区定位过窄。可视情况：用更高分辨率的真人护照图替换样本。")

    @staticmethod
    def _ocr_garbage_hint(line1: str, line2: str):
        """OCR sometimes returns rc=0 / "OK" yet the character string
        is nonsense (e.g. "AAAAAAMM...") because mrz_ocr_tool has
        no language model: it classifies each glyph independently and
        falls back to the most-frequent training classes (A, M, W, V,
        L, J, Z). Detect this by checking whether the OCR output is a
        strong concentration of those training-frequent glyphs.
        """
        if not line1 or not line2:
            return ""
        from collections import Counter
        combined = (line1 + line2).strip()
        if len(combined) < 44:
            return ""
        counts = Counter(combined)
        # Training-frequent glyph classes in the OCR's bitmap font
        # make_sample_passport and most real MRZ fonts.
        frequent = set("AMWVSLJKZ")
        freq_total = sum(c for k, c in counts.items() if k in frequent)
        # Real MRZ mixes these with the rest of the alphabet. OCR
        # garbage strings hover near 90% on this set.
        freq_ratio = freq_total / len(combined)
        if freq_ratio >= 0.55:
            return (f"[!] OCR 输出字符分布异常（A/M/W/V/S/L/J/Z 类字符"
                    f"占比 {freq_ratio*100:.0f}%），高概率是 OCR 模型"
                    f"在没有语言模型时的占位字符，不是真实 MRZ。"
                    f"建议：换用更高分辨率或更多训练样本的 OCR 模型。"
                    f"本样本的真实 ICAO 字符（如 'P<CZSPECIMEN...'）无法从"
                    f"当前 OCR 输出中恢复。")
        return ""

    # ---- MRZ-only quick action (locator tab) -------------------------
    def _loc_mrz_recognize(self):
        """Locate the MRZ band on the currently selected image, crop
        it, hand it to mrz_tool, and display the parsed fields plus a
        pass/fail chip on the right-hand info pane.
        """
        pal = self._pal
        chip = pal["chip"]
        def _chip(text, kind):
            self._loc_mrz_chip_var.set(text)
            c = chip[kind]
            try:
                self.loc_mrz_chip.configure(bg=c["bg"], fg=c["fg"])
            except tk.TclError:
                try:
                    self.loc_mrz_chip.configure(
                        background=c["bg"], foreground=c["fg"])
                except tk.TclError:
                    pass
        try:
            self.loc_mrz_btn.configure(state="disabled")
            self.loc_status.config(text="MRZ 识别中…", foreground=self._pal["dim"])
            _chip("  RUN  ", "run")
            from PIL import Image
            # Drive the bottom status bar through each pipeline stage so
            # long OCR/decode runs stay visibly active (single-shot path
            # previously never touched the progress bar).
            self._set_status_bar("定位 MRZ 区…", 10)
            self.update_idletasks()
            path = self._loc_resolve_path()
            img = Image.open(path).convert("RGB")
            loc = self._loc_locator(img)
            r = loc.get("mrz")
            if not r:
                _chip("  FAIL  ", "fail")
                self._set_status_bar("MRZ 定位失败", 0)
                self.loc_status.config(
                    text="未定位到 MRZ 区", foreground=self._pal["err"])
                self._loc_set_text(
                    f"样本: {os.path.basename(path)}\n"
                    f"无法定位 MRZ 区。请先点击“定位区域”确认区域，"
                    f"或换一张更清晰的护照图。")
                return
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            crop = img.crop((x, y, x + w, y + h))
            # Real-photo MRZ lines are ~12px tall while the OCR pipeline
            # was trained at 36-72px; upscale low bands so the glyphs land
            # in-distribution (else output degenerates to garbage).
            from passport_pipeline import auto_upscale
            _ups = auto_upscale(crop)
            if _ups > 1:
                crop = crop.resize((crop.width * _ups, crop.height * _ups),
                                   Image.LANCZOS)
            import tempfile
            with tempfile.NamedTemporaryFile(
                    suffix=".ppm", delete=False) as tf:
                tmp_path = Path(tf.name)
            try:
                crop.save(tmp_path)
                # Pipeline:
                #   1) mrz_ocr_tool reads the cropped PPM and prints
                #      result.line1 / line2 / conf1 / conf2 / ok.
                #   2) mrz_tool only accepts the 44-char strings, not
                #      PPM, so feed it the OCR result to get the parsed
                #      fields and check digits.
                self._set_status_bar("MRZ OCR 识别中…", 50)
                self.update_idletasks()
                ocr_proc = subprocess.run(
                    [str(ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool"),
                     str(tmp_path)],
                    text=True, capture_output=True, timeout=20)
                ocr_kv = {}
                for ln in (ocr_proc.stdout or "").splitlines():
                    ln = ln.strip()
                    if not ln or ":" not in ln:
                        continue
                    k, _, v = ln.partition(":")
                    ocr_kv[k.strip()] = v.strip()
                line1 = ocr_kv.get("result.line1", "")
                line2 = ocr_kv.get("result.line2", "")
                # The OCR binary uses `result.ok` as a status field: it
                # is the literal string "OK" on success, or an English
                # error message ("could not split into two lines",
                # "image too small", ...) on failure. Treat anything
                # other than "OK" as a hard OCR failure and skip
                # mrz_tool decode entirely so we don't surface the
                # cryptic `bad MRZ length` secondary error.
                raw_ok = (ocr_kv.get("result.ok", "") or "").strip()
                ok = raw_ok.upper() == "OK" and bool(line1) and bool(line2)
                if not ok:
                    # When OCR failed, line1/line2 are empty. We pass
                    # raw_ok to mrz_tool decode as the (empty) MRZ so
                    # its own failure message is informative, but the
                    # real cause is the OCR step.
                    line1 = ""
                    line2 = ""
                # Surface mrz_ocr_tool's stderr in the report so the user
                # can see why the OCR step failed (empty / bad image).
                self._loc_last_ocr_err = (ocr_proc.stderr or "").strip()
                if line1 and line2:
                    # mrz_tool decode accepts a single MRZ string of two
                    # newline-joined lines, not two positional args.
                    self._set_status_bar("校验 MRZ 解码…", 85)
                    self.update_idletasks()
                    decode_proc = subprocess.run(
                        [str(MRZ_TOOL), "decode",
                         f"{line1}\n{line2}"],
                        text=True, capture_output=True, timeout=10)
                else:
                    # OCR returned no MRZ text. Surface a clear explanation
                    # so the user understands why decode was skipped and
                    # can decide to swap in a higher-resolution sample.
                    decode_proc = subprocess.CompletedProcess(
                        args=[], returncode=2, stdout="",
                        stderr=("mrz_ocr_tool 未识别出 MRZ 两行 "
                                f"(line1={line1!r}, line2={line2!r}). "
                                "常见原因：合成样本 / 低分辨率照片 / "
                                "MRZ 区被裁切过窄。"))
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            # Parse mrz_tool decode output. mrz_tool uses the same
            # key=value protocol for fields and check digits.
            kv = dict(ocr_kv)  # start with OCR results; overlay decoded
            for ln in (decode_proc.stdout or "").splitlines():
                ln = ln.strip()
                if not ln or ":" not in ln:
                    continue
                k, _, v = ln.partition(":")
                kv[k.strip()] = v.strip()
            _chip(f"  {"PASS" if ok else "FAIL"}  ",
                  "ok" if ok else "fail")
            self.loc_status.config(
                text=("MRZ ✓ 通过" if ok else "MRZ ✗ 失败"),
                foreground=self._pal["ok"] if ok else self._pal["err"])
            if ok:
                self._set_status_bar("MRZ 识别完成", 100, done=1, total=1)
            else:
                self._set_status_bar("MRZ 识别失败（详见报告）", 0)
            # Render a structured report in the info pane.
            lines = [
                f"样本 : {os.path.basename(path)}",
                f"区域 : x={x} y={y} w={w} h={h}   "
                f"置信度 {r['confidence']:.2f}",
                f"依据 : {r.get('evidence', '-')}",
                "",
                f"mrz_ocr_tool rc : {ocr_proc.returncode}   "
                f"mrz_tool rc : {decode_proc.returncode}",
                "",
                "mrz_ocr_tool [stderr] (only shown when non-empty):",
                (ocr_proc.stderr.strip() or "(none)"),
                # Surface OCR confidence so the user can see whether
                # 'OK' actually means a confident recognition. Treat
                # <70% as WARN, <40% as FAIL even when the tool says OK.
                f"result.ok   : {kv.get('result.ok', '-')}   "
                f"conf1={kv.get('result.conf1', '-')}%   "
                f"conf2={kv.get('result.conf2', '-')}%",
                "" if decode_proc.returncode == 0 else
                "[!] mrz_tool decode 返回非 0 (说明 OCR 出的两行不是合法 ICAO 字符，"
                "常见原因：合成样本 / 低分辨率 / MRZ 区定位过窄)",
                "",
                self._synth_hint(int(kv.get("result.conf1", "0") or 0),
                                   int(kv.get("result.conf2", "0") or 0),
                                   decode_proc.returncode == 0),
                # Even when OCR says "OK", the output characters may
                # be garbage (mrz_ocr_tool has no language model so
                # it can output A/M/W/V/L/J/Z placeholders that look
                # like ICAO chars but make no ICAO sense).
                self._ocr_garbage_hint(line1, line2),
                f"result.name : {kv.get('result.name', '-')}",
                f"result.doc  : {kv.get('result.doc',  '-')}",
                f"result.nat  : {kv.get('result.nat',  '-')}",
                f"result.dob  : {kv.get('result.dob',  '-')}",
                f"result.exp  : {kv.get('result.exp',  '-')}",
                f"result.sex  : {kv.get('result.sex',  '-')}",
                "",
                f"line1 (44)  : {kv.get('result.line1', '-')}",
                f"line2 (44)  : {kv.get('result.line2', '-')}",
                "",
                f"check_digit : "
                f"{kv.get('result.check_doc',  '-')} / "
                f"{kv.get('result.check_dob',  '-')} / "
                f"{kv.get('result.check_exp',  '-')} / "
                f"{kv.get('result.check_comp', '-')}",
                "",
                "--- raw mrz_ocr_tool output ---",
                ocr_proc.stdout or "",
            ]
            if ocr_proc.stderr:
                lines += ["", "[stderr]", ocr_proc.stderr]
            lines += ["",
                "--- raw mrz_tool decode output ---",
                decode_proc.stdout or ""]
            if decode_proc.stderr:
                lines += ["", "[stderr]", decode_proc.stderr]
            self._loc_set_text("\n".join(lines))
            # Refresh the annotated preview so the MRZ box is visible.
            try:
                disp = self._loc_annotate(img, loc, scale=1.0)
                disp.thumbnail((max(self.loc_canvas.winfo_width() - 4, 320),
                                max(self.loc_canvas.winfo_height() - 4, 220)))
                from PIL import ImageTk
                self._loc_imgtk = ImageTk.PhotoImage(disp)
                self.loc_canvas.delete("all")
                self.loc_canvas.create_image(
                    2, 2, image=self._loc_imgtk, anchor="nw")
            except Exception:
                pass
        except Exception as e:
            _chip("  FAIL  ", "fail")
            self._set_status_bar(f"MRZ 失败: {e}", 0)
            self.loc_status.config(text=f"MRZ 失败: {e}",
                                   foreground=self._pal["err"])
            self._loc_set_text(f"错误: {e}")
        finally:
            try:
                self.loc_mrz_btn.configure(state="normal")
            except tk.TclError:
                pass

    # ---- direct-image recognition (jpg/png on demand) ---------------
    def _ocr_pick_and_recognize(self):
        """Open a file picker for a single JPG/PNG/PPM, render its
        thumbnail in the right pane, then run mrz_ocr_tool + mrz_tool
        and stream the parsed fields + chip status into the diff pane."""
        path = filedialog.askopenfilename(
            title="选择护照图片直接识别",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.ppm *.bmp"),
                       ("All", "*.*")])
        if not path:
            return
        # Render thumbnail first so the user has immediate visual feedback.
        try:
            from PIL import Image
            self._preview_img_pil = Image.open(path).convert("RGB")
            self._preview_w, self._preview_h = self._preview_img_pil.size
            self._preview_img_path = Path(path)
            self._preview_last_pm = {}  # no OCR result yet
            self._update_preview_image()
        except Exception as e:
            messagebox.showerror("加载失败", f"无法读取 {path}: {e}")
            return
        # Reset the diff pane and stream a "running" header.
        self.ocr_diff_text.configure(state="normal")
        self.ocr_diff_text.delete("1.0", "end")
        self.ocr_diff_text.insert(
            "end", f"样本 : {os.path.basename(path)}\n", ("hdr",))
        self.ocr_diff_text.insert("end", "状态 : 运行中…\n\n", ("lab",))
        self.ocr_diff_text.configure(state="disabled")
        self.ocr_preview_meta.set(
            f"{os.path.basename(path)}  ({self._preview_w}×"
            f"{self._preview_h})")
        self._ocr_meta_fullpath = str(path)
        self._ocr_meta_tip.set_text(self._ocr_meta_fullpath)
        self.ocr_pick_btn.configure(state="disabled")
        try:
            self.ocr_run_btn.configure(state="disabled")
        except tk.TclError:
            pass

        def worker():
            err = None
            kv = {}
            proc_ms = 0.0
            self.after(0, lambda: self._set_status_bar(
                "MRZ OCR 识别中…", 50, done=0, total=1))
            try:
                from pathlib import Path as _P
                t0 = time.time()
                proc = subprocess.run(
                    [str(ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool"), path],
                    text=True, capture_output=True, timeout=20)
                proc_ms = (time.time() - t0) * 1000
                for ln in (proc.stdout or "").splitlines():
                    ln = ln.strip()
                    if not ln or ":" not in ln:
                        continue
                    k, _, v = ln.partition(":")
                    kv[k.strip()] = v.strip()
            except Exception as e:  # noqa: BLE001
                err = e
            self.after(0, lambda: self._ocr_render_picked_result(
                path, kv, proc_ms, err))

        threading.Thread(target=worker, daemon=True).start()

    def _ocr_render_picked_result(self, path, kv, proc_ms, err):
        """Paint the picked-image OCR result into the diff pane and
        enable the picker / run buttons again."""
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        chip = self._pal["chip"]
        if err:
            self._set_status_bar(f"识别失败: {err}", 0)
        else:
            self._set_status_bar("识别完成", 100, done=1, total=1)
        ok = kv.get("result.ok", "FAIL").upper() == "OK"
        c1 = int(kv.get("result.conf1", "0") or 0)
        c2 = int(kv.get("result.conf2", "0") or 0)
        # Confidence header (parity with corpus diff pane).
        line1_conf = (f"line1 conf={c1}% ("
                      f"{'PASS' if c1 >= 70 else 'WARN' if c1 >= 40 else 'FAIL'})")
        line2_conf = (f"line2 conf={c2}% ("
                      f"{'PASS' if c2 >= 70 else 'WARN' if c2 >= 40 else 'FAIL'})")
        ok_tag = "okline" if ok else "lab"
        tw.insert("end",
                  f"样本  : {os.path.basename(path)}\n", "hdr")
        tw.insert("end",
                  f"耗时  : {proc_ms:.0f} ms\n", "lab")
        tw.insert("end",
                  f"判定  : {'PASS  ✅' if ok else 'FAIL  ❌'}\n", ok_tag)
        tw.insert("end",
                  f"{line1_conf}    {line2_conf}\n\n", "lab")
        # Parsed fields + raw stdout/stderr for forensic look.
        fields = [
            ("line1",  kv.get("result.line1", "-")),
            ("line2",  kv.get("result.line2", "-")),
            ("name",   kv.get("result.name",  "-")),
            ("doc",    kv.get("result.doc",   "-")),
            ("nat",    kv.get("result.nat",   "-")),
            ("dob",    kv.get("result.dob",   "-")),
            ("exp",    kv.get("result.exp",   "-")),
            ("sex",    kv.get("result.sex",   "-")),
            ("band",   kv.get("band.x band.y band.w band.h", "-")),
        ]
        tw.insert("end", "字段解析:\n", "hdr")
        for k, v in fields:
            tw.insert("end", f"  {k:6} : {v}\n", "lab")
        if err is not None:
            tw.insert("end", "\n[error]\n", "mm")
            tw.insert("end", f"{err}\n", "lab")
        tw.configure(state="disabled")
        # Persist the parsed result so the next single-sample preview /
        # confidence header picks it up. Whole-image OCR means the band
        # coordinates are absolute (no crop offset).
        cid = f"_picked:{os.path.basename(path)}"
        self._ocr_last = getattr(self, "_ocr_last", {})
        self._ocr_last[cid] = {
            "cnn": {
                "line1": kv.get("result.line1", ""),
                "line2": kv.get("result.line2", ""),
                "conf1": c1, "conf2": c2,
                "band":  kv.get("band.x band.y band.w band.h", ""),
                "crop_offset": (0, 0),
                "ok":    ok, "ms": proc_ms,
            }
        }
        # Re-render the right-side preview so the region overlay uses
        # the freshly fetched band/conf.
        self._preview_last_pm = self._ocr_last[cid]["cnn"]
        try:
            self._update_preview_image()
        except tk.TclError:
            pass
        # Restore buttons.
        try:
            self.ocr_pick_btn.configure(state="normal")
        except tk.TclError:
            pass
        try:
            self.ocr_run_btn.configure(state="normal")
        except tk.TclError:
            pass

    def _ocr_set_sashes(self):
        body = getattr(self, "_ocr_body", None)
        if body is None:
            return
        total = body.winfo_width()
        if total < 120:
            return
        w_left = max(240, int(total * 0.22))
        w_right = max(400, int(total * 0.30))
        body.sashpos(0, w_left)
        body.sashpos(1, max(w_left + 60, total - w_right))

    def _build_ocr_metrics(self, parent):
        card = ttk.LabelFrame(parent, text="总体指标")
        card.pack(fill="x", pady=(0, 4))
        self.ocr_metrics = {}
        self._metric_cells = []
        colors = self._pal["metric"]
        specs = [("cases", "案例数"), ("mean", "Mean耗时"),
                 ("p95", "P95耗时"), ("pass", "Pass率"),
                 ("l1", "平均L1"), ("l2", "平均L2"),
                 ("full", "全匹配")]
        units = {"mean": "ms", "p95": "ms", "pass": "%",
                 "l1": "%", "l2": "%"}
        for i in range(len(specs)):
            card.grid_columnconfigure(i, weight=1, uniform="metric_col")
        for i, (key, label) in enumerate(specs):
            cell = tk.Frame(card, bg=self._pal["card"], highlightthickness=1,
                            highlightbackground=self._pal["border"])
            cell.grid(row=0, column=i, sticky="nsew", padx=3, pady=4)
            ttl = ttk.Label(cell, text=label, foreground=self._pal["dim"],
                            font=("TkDefaultFont", 9))
            ttl.pack(anchor="w", padx=8, pady=(4, 0))
            var = tk.StringVar(value="--")
            vrow = tk.Frame(cell, bg=self._pal["card"])
            vrow.pack(anchor="w", padx=8, pady=(0, 4), fill="x")
            val = ttk.Label(vrow, textvariable=var,
                            foreground=colors[key],
                            font=("Menlo", 15, "bold"))
            val.pack(side="left")
            if key in units:
                ttk.Label(vrow, text=units[key], foreground="#6E7681",
                          font=("Segoe UI", 9)).pack(
                    side="left", anchor="s", padx=(3, 0), pady=(0, 2))
            self.ocr_metrics[key] = var
            self._metric_cells.append((cell, ttl, val, key))

    def _fit_columns(self, tree, pad=16, minw=50, cap=340):
        """Auto-fit Treeview column widths from heading + content."""
        import tkinter.font as tkfont
        f = tkfont.nametofont("TkDefaultFont")
        cols = list(tree["columns"])
        for idx, c in enumerate(cols):
            w = f.measure(tree.heading(c, "text"))
            for iid in tree.get_children():
                v = tree.item(iid, "values")
                if idx < len(v):
                    w = max(w, f.measure(str(v[idx])))
            tree.column(c, width=min(cap, max(minw, w + pad)))

    def _set_status_bar(self, msg, pct=None, done=None, total=None):
        """Update the bottom status bar with text + Progressbar.

        pct=None -> indeterminate spinning bar (no percentage chip)
        pct=0..100 -> determinate bar with a percentage chip
        done/total -> also paint "d/t" into the chip when available
        """
        self.ocr_status_var.set(msg)
        chip = self._pal["chip"]
        if pct is None:
            self.ocr_progress.configure(mode="indeterminate")
            self.ocr_progress.start(12)
            self._ocr_paint_progress_chip("  ...  ", chip["dim"]["bg"],
                                          chip["dim"]["fg"])
        else:
            pct = max(0.0, min(100.0, float(pct)))
            self.ocr_progress.stop()
            self.ocr_progress.configure(mode="determinate", value=pct)
            if pct >= 100:
                kind = "ok"
            elif pct >= 40:
                kind = "idle"
            else:
                kind = "run"
            counter = ""
            if done is not None and total is not None and total > 0:
                counter = f" {done}/{total}  {pct:.0f}% "
            else:
                counter = f" {pct:.0f}% "
            label = "  DONE  " if pct >= 100 else "  RUN  "
            self._ocr_paint_progress_chip(label + counter,
                                          chip[kind]["bg"],
                                          chip[kind]["fg"])

    def _ocr_paint_progress_chip(self, text, bg, fg):
        """Repaint the progress chip for both ttk and CTK backends."""
        parent = getattr(self, "_status_bar", None)
        if parent is None:
            return
        existing = getattr(self, "_progress_chip", None)
        if existing is not None:
            try:
                existing.destroy()
            except tk.TclError:
                pass
            self._progress_chip = None
        if not text:
            return
        try:
            lbl = tk.Label(parent, text=text, bg=bg, fg=fg,
                           font=self._pal["font_ui_bold"], padx=6, pady=1)
            lbl.pack(side="left", padx=4)
            self._progress_chip = lbl
        except tk.TclError:
            pass

    def _ocr_method_toggle(self):
        if self.ocr_method_both.get():
            self.ocr_methods.set(",".join(METHOD_ORDER))
        else:
            m = self.ocr_methods.get()
            if "," in m:
                self.ocr_methods.set(m.split(",")[0])

    def _ocr_select_all(self, on: bool):
        all_ids = list(self.ocr_case_list.get_children())
        if on:
            self.ocr_case_list.selection_set(all_ids)
        else:
            self.ocr_case_list.selection_remove(all_ids)

    def _ocr_clear_sel(self):
        self.ocr_case_list.selection_remove(
            list(self.ocr_case_list.get_children()))

    def _ocr_apply_filter(self):
        f = self.ocr_filter_var.get()
        all_ids = list(self.ocr_case_list.get_children())
        self.ocr_case_list.selection_remove(all_ids)
        if f == "all":
            return
        for iid in all_ids:
            tags = self.ocr_case_list.item(iid, "tags")
            if f in tags:
                self.ocr_case_list.selection_add(iid)

    def _ocr_apply_search(self, _evt=None):
        q = self.ocr_search_var.get().strip().lower()
        tree = self.ocr_case_list
        ids = self._ocr_case_order
        if not q:
            for iid in ids:
                if iid not in tree.get_children():
                    tree.move(iid, "", len(tree.get_children()))
            return
        for iid in ids:
            if iid in tree.get_children():
                tree.detach(iid)
        for iid in ids:
            if q in iid.lower():
                tree.move(iid, "", len(tree.get_children()))

    # ---- single-sample preview: image + char-level diff ----
    def _load_preview(self, cid, rec):
        from pathlib import Path as _P
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        corpus_dir = _P(CORPUS_JSON).parent
        img_path = corpus_dir / rec["image"]
        self.ocr_preview_meta.set(
            f"id={cid}  scale={rec['scale']}  noise={rec['noise']}  "
            f"skew={rec['skew']}")
        self._ocr_meta_fullpath = str(img_path)
        self._ocr_meta_tip.set_text(self._ocr_meta_fullpath)
        # Stash the absolute path so the viewer / region overlay can find it.
        self._preview_img_path = img_path
        try:
            from PIL import Image
            self._preview_img_pil = Image.open(img_path).convert("RGB")
            self._preview_w, self._preview_h = self._preview_img_pil.size
            # Cache last-known band / conf so the diff pane can show
            # them alongside the per-char comparison. Whole-image bench
            # so the band is already in absolute coordinates.
            last = getattr(self, "_ocr_last", {}).get(cid, {}) or {}
            pm = last.get("cnn") or last.get("traditional") or {}
            if pm and "crop_offset" not in pm:
                pm = dict(pm)
                pm["crop_offset"] = (0, 0)
            self._preview_last_pm = pm
            self._update_preview_image()
        except Exception:
            self._preview_img_pil = None
            self.ocr_preview_label.configure(image="",
                text=f"(无法预览 {img_path.name})")

    def _update_preview_image(self, _evt=None):
        im = getattr(self, "_preview_img_pil", None)
        if im is None:
            return
        from PIL import Image, ImageDraw, ImageTk
        # Use the requested display size; if the label hasn't been laid
        # out yet (rare under real Tk, common in headless probes) fall
        # back to the label's requested width/height or 320x180 so the
        # region overlay still paints. Wrap winfo_* calls defensively.
        w, h = 0, 0
        for src in (lambda: (self.ocr_preview_label.winfo_width(),
                             self.ocr_preview_label.winfo_height()),
                    lambda: (self.ocr_preview_label.winfo_reqwidth(),
                             self.ocr_preview_label.winfo_reqheight())):
            try:
                w, h = src()
                break
            except tk.TclError:
                continue
        if w <= 1 or h <= 1:
            w, h = 320, 180
        iw, ih = self._preview_w, self._preview_h
        # Scale-thumbnail keeping aspect ratio so the photo fits the
        # preview pane. The Tk label uses the actual displayed size
        # (w x h) minus a small margin; we map image -> display coords
        # uniformly.
        margin = 4
        tw, th = max(1, w - margin), max(1, h - margin)
        scale = min(tw / iw, th / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox = (w - dw) // 2
        oy = (h - dh) // 2
        # Resize the original pixels to the displayed size; we then
        # paint the overlay rectangles directly in pixel space (dw x dh)
        # which simplifies the mapping vs the original iw x ih space.
        im2 = im.resize((dw, dh), Image.LANCZOS)
        try:
            overlay = im2.copy()
            drw = ImageDraw.Draw(overlay, "RGBA")
            # HUD corner brackets (cyan) before the region boxes.
            _hu, _l = (0, 245, 255, 255), 14
            drw.line((0, _l, 0, 0, _l, 0), fill=_hu, width=3)
            drw.line((dw - _l, 0, dw, 0, dw, _l), fill=_hu, width=3)
            drw.line((0, dh - _l, 0, dh, _l, dh), fill=_hu, width=3)
            drw.line((dw - _l, dh, dw, dh, dw, dh - _l), fill=_hu, width=3)
            # Region rectangles (proportional to displayed image):
            #   photo  = left 25%, top 10%, width 25%, height 70%
            #   data   = top 10%, left 25%, width 50%, height 25%
            #   mrz    = bottom 30%, full width (overrides data on the
            #            bottom edge, which is the real passport layout)
            # Colours match .impeccable.md neon palette.
            # A pre-cropped MRZ strip (e.g. a PPM fed to "直接识别图片")
            # contains only the MRZ band: painting the speculative
            # proportional PHOTO/DATA boxes over it would fabricate
            # bogus regions. Full-page passport images keep the
            # proportional layout boxes.
            is_strip = ih < 260 or (iw / ih) > 3.0
            if is_strip:
                photo_box = data_box = None
            else:
                photo_box = (int(dw * 0.04), int(dh * 0.08),
                             int(dw * 0.30), int(dh * 0.78))
                data_box  = (int(dw * 0.30), int(dh * 0.08),
                             int(dw * 0.96), int(dh * 0.40))
            mrz_box   = (int(dw * 0.04), int(dh * 0.62),
                         int(dw * 0.96), int(dh * 0.96))
            # If the OCR tool returned a band, snap MRZ box to it. The
            # band may live in crop-local coordinates (passport_pipeline
            # runs OCR on the cropped MRZ band), so add crop_offset.
            pm_box = getattr(self, "_preview_last_pm", {}) or {}
            band_raw = pm_box.get("band", "")
            cx, cy = pm_box.get("crop_offset", (0, 0)) or (0, 0)
            if band_raw:
                try:
                    bx, by, bw, bh = (int(float(x)) for x in band_raw.split())
                    # Convert crop-local -> absolute -> preview pixels.
                    abx, aby = bx + cx, by + cy
                    sx = dw / iw; sy = dh / ih
                    mrz_box = (int(abx * sx), int(aby * sy),
                               int((abx + bw) * sx), int((aby + bh) * sy))
                except Exception:
                    pass
            # Emerald / amber / cyan (cockpit palette); translucent alpha
            # so the underlying image stays visible.
            if photo_box:
                drw.rectangle(photo_box, outline=(16, 185, 129, 255), width=2)
            if data_box:
                drw.rectangle(data_box,  outline=(245, 158, 11, 255), width=2)
            drw.rectangle(mrz_box,   outline=(0, 245, 255, 255), width=2)
            # Small chip tags so the user can identify each region.
            def tag(box, label, fill):
                x0, y0, x1, y1 = box
                pad = 2
                tag_w = max(28, len(label) * 7 + 8)
                ty = max(0, y0 - 12)
                drw.rectangle((x0, ty, x0 + tag_w, ty + 12),
                              fill=fill)
                drw.text((x0 + 4, ty + 1), label, fill=(0, 0, 0, 255))
            if photo_box:
                tag(photo_box, "PHOTO", (16, 185, 129, 255))
            if data_box:
                tag(data_box,  "DATA",  (245, 158, 11, 255))
            tag(mrz_box,   "MRZ",   (0, 245, 255, 255))
            photo = ImageTk.PhotoImage(overlay)
        except Exception:
            try:
                from PIL import ImageTk
                photo = ImageTk.PhotoImage(im2)
            except Exception:
                return
        self._preview_imgs["_current"] = photo
        self.ocr_preview_label.configure(image=photo, text="")
        # Stash the on-screen rectangle so other widgets (diff pane)
        # can reference consistent proportions if needed.
        self._preview_layout = {"scale": scale, "ox": ox, "oy": oy,
                                "dw": dw, "dh": dh,
                                "photo": photo_box, "data": data_box,
                                "mrz": mrz_box}

    def _render_diff_block(self, tw, label, gt, pred):
        """Append one aligned GT/Pred block with per-char highlighting."""
        n = max(len(gt), len(pred))
        gt = gt.ljust(n)
        pred = pred.ljust(n)
        tw.insert("end", label + "\n", "hdr")
        tw.insert("end", "GT  : ", "lab")
        for i in range(n):
            if gt[i] == pred[i]:
                tag = "pad" if gt[i] == "<" else "match"
            else:
                tag = "mm"
            tw.insert("end", gt[i], tag)
        tw.insert("end", "\nPRED: ", "lab")
        mism = []
        for i in range(n):
            if gt[i] != pred[i]:
                mism.append(i)
                tw.insert("end", pred[i], "error")
            else:
                tw.insert("end", pred[i], "pad" if pred[i] == "<" else "match")
        tw.insert("end", "\n     ")
        if mism:
            for i in range(n):
                tw.insert("end", "^" if i in mism else " ", "error")
            tw.insert("end", "\n")
        else:
            tw.insert("end", "（全部匹配）\n", "okline")

    def _show_diff(self, blocks):
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        # Confidence header: one line per block, mirroring the layout
        # expected by the PPM benchmark output (conf1 / conf2 0..100).
        pm = getattr(self, "_preview_last_pm", None) or {}
        for label, gt, pred in blocks:
            tag = "okline" if gt == pred else "lab"
            conf = pm.get("conf1") if "line1" in label else (
                   pm.get("conf2") if "line2" in label else None)
            if conf is None:
                tw.insert("end", f"{label}  conf=n/a\n", tag)
            else:
                grade = "PASS" if conf >= 70 else ("WARN" if conf >= 40 else "FAIL")
                tw.insert("end",
                          f"{label}  conf={conf}% ({grade})\n", tag)
            tw.insert("end", "\n", "lab")
        for label, gt, pred in blocks:
            self._render_diff_block(tw, label, gt, pred)
        tw.configure(state="disabled")

    def _show_gt_only(self, rec):
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        # Show last-known confidence (if any) above the GT lines so the
        # user can compare expected vs predicted structure immediately.
        pm = getattr(self, "_preview_last_pm", None) or {}
        c1 = pm.get("conf1")
        c2 = pm.get("conf2")
        if c1 is not None or c2 is not None:
            tw.insert("end", "最近一次识别置信率:\n", "hdr")
            tw.insert("end",
                f"  line1 conf={c1 if c1 is not None else 'n/a'}%   "
                f"line2 conf={c2 if c2 is not None else 'n/a'}%\n",
                "lab")
            tw.insert("end", "\n", "lab")
        tw.insert("end", "line1 (GT)\n", "hdr")
        tw.insert("end", rec["line1"] + "\n", "match")
        tw.insert("end", "line2 (GT)\n", "hdr")
        tw.insert("end", rec["line2"] + "\n", "match")
        tw.insert("end", "（运行测试后在此显示 Pred 与字符级比对）", "lab")
        tw.configure(state="disabled")

    def _ocr_on_list_click(self, _evt=None):
        """Photo-mode single-click: run the full pipeline for the
        selected photo and stream results to the right pane. No-op
        for corpus cases (the corpus bench needs ⚡ 开始测试)."""
        if not self.ocr_photo_mode.get():
            return
        sel = self.ocr_case_list.selection()
        if not sel:
            return
        name = sel[0]
        # Avoid re-running if the same row is clicked twice in a row.
        last = getattr(self, "_ocr_last_one", None)
        if last == name:
            return
        self._ocr_last_one = name
        self.ocr_run_btn.configure(state="disabled")
        # Render the image immediately so the right pane shows
        # something while we wait for the worker.
        from passport_pipeline import PIC_DIR
        path = PIC_DIR / name
        try:
            from PIL import Image
            pil = Image.open(path).convert("RGB")
            self._preview_img_pil = pil
            self._preview_w, self._preview_h = pil.size
            self._preview_img_path = path
            self._preview_last_pm = {}
            self.ocr_preview_meta.set(f"{name}")
            self._ocr_meta_fullpath = str(path)
            self._ocr_meta_tip.set_text(self._ocr_meta_fullpath)
            self._update_preview_image()
        except Exception as e:
            messagebox.showerror("加载失败", f"{path}: {e}")
            return
        self.ocr_diff_text.configure(state="normal")
        self.ocr_diff_text.delete("1.0", "end")
        self.ocr_diff_text.insert("end",
            f"样本 : {name}\n状态 : 运行中…", ("lab",))
        self.ocr_diff_text.configure(state="disabled")

        def worker():
            try:
                from passport_pipeline import process_image, CNN
                rec = process_image(path, tool=CNN)
            except Exception as e:
                self.after(0, lambda e=e:
                           self._ocr_render_single_photo_failed(name, e))
                return
            self.after(0, lambda rec=rec:
                       self._ocr_render_single_photo_result(name, rec))

        threading.Thread(target=worker, daemon=True).start()

    def _ocr_render_single_photo_result(self, name, rec):
        """Insert the single-photo result row and select it so the
        right pane repaints with the photo + region overlay + MRZ
        details. Mirrors what batch mode does after _ocr_populate_photos."""
        # Ensure _ocr_last['photo'] is populated so _ocr_on_detail_select
        # can find the rec.
        self._ocr_last = getattr(self, "_ocr_last", {})
        existing = self._ocr_last.get("photo", [])
        # Replace any previous entry for the same file (re-run scenario).
        self._ocr_last["photo"] = [r for r in existing if r.get("file") != name] + [rec]
        # Insert a detail row tagged with the grade so the chip colour
        # matches the verdict (PASS green / SUSPECT amber / REJECT red).
        grade = rec.get("grade", "REJECT")
        iid = name
        if not self.ocr_detail.exists(iid):
            self.ocr_detail.insert("", "end", iid=iid, tags=(grade,),
                values=(name, grade,
                        "✓" if rec.get("verified") else "—",
                        f"{rec.get('ms', 0):.0f}",
                        rec.get("reason") or ""))
        else:
            self.ocr_detail.item(iid, tags=(grade,),
                values=(name, grade,
                        "✓" if rec.get("verified") else "—",
                        f"{rec.get('ms', 0):.0f}",
                        rec.get("reason") or ""))
        self._ocr_detail_map[iid] = ("photo", name)
        self.ocr_detail.selection_set(iid)
        self.ocr_detail.focus(iid)
        # Re-render the right pane (thumbnail + overlay + diff text).
        self._ocr_on_detail_select()
        self.ocr_run_btn.configure(state="normal")

    def _ocr_render_single_photo_failed(self, name, err):
        """Surface worker errors in the right pane and re-enable the
        run button so the user can try again."""
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        tw.insert("end", f"样本 : {name}\n", "hdr")
        tw.insert("end", f"异常 : {err}\n", "mm")
        tw.configure(state="disabled")
        try:
            self.ocr_run_btn.configure(state="normal")
        except tk.TclError:
            pass

    def _ocr_on_case_select(self, _evt=None):
        sel = self.ocr_case_list.selection()
        if not sel:
            return
        cid = sel[0]
        rec = next((r for r in self._ocr_corpus["records"]
                    if r["id"] == cid), None)
        if not rec:
            return
        self._load_preview(cid, rec)
        last = getattr(self, "_ocr_last", {})
        blocks = []
        for m in ("traditional", "cnn"):
            r = last.get(cid, {}).get(m)
            if r:
                label = METHOD_LABELS[m]
                blocks.append((f"line1 [{label}]", rec["line1"],
                               r.get("line1", "")))
                blocks.append((f"line2 [{label}]", rec["line2"],
                               r.get("line2", "")))
        if blocks:
            self._show_diff(blocks)
        else:
            self._show_gt_only(rec)

    # ---- Full-size image viewer (double-click on case / detail row) ----
    def _ocr_open_original_from_detail(self, _evt=None):
        """Resolve the corpus record for the currently selected detail
        row and pop the full-size viewer. Detail rows are keyed by
        (case_id, method); the case_id is the first 8 chars of the row
        iid (or the prefix before "::")."""
        sel = self.ocr_detail.selection()
        if not sel:
            return
        iid = sel[0]
        cid = iid.split("::", 1)[0] if "::" in iid else iid[:8]
        rec = next((r for r in self._ocr_corpus["records"]
                    if r["id"] == cid), None)
        if rec is None:
            return
        self._ocr_open_original(cid, rec)

    def _ocr_open_original(self, cid=None, rec=None):
        """Open a top-level window showing the original (unresized)
        image with zoom controls (Fit / 1x / 2x / 4x) and a chip bar
        reporting case metadata. Mirrors the CustomTkinter preview."""
        if rec is None or cid is None:
            sel = self.ocr_case_list.selection()
            if not sel:
                return
            cid = sel[0]
            rec = next((r for r in self._ocr_corpus["records"]
                        if r["id"] == cid), None)
            if rec is None:
                return
        from pathlib import Path as _P
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        img_path = _P(CORPUS_JSON).parent / rec["image"]
        if not img_path.exists():
            messagebox.showerror("原图缺失", f"找不到图片文件:\n{img_path}")
            return
        try:
            from PIL import Image as _PILImage
        except Exception:
            messagebox.showerror("依赖缺失",
                                 "原图查看器需要 Pillow: pip install pillow")
            return
        try:
            full_pil = _PILImage.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法读取 {img_path.name}: {e}")
            return
        win = tk.Toplevel(self)
        win.title(f"原图 · {cid}  ·  {img_path.name}")
        pal = self._pal
        try:
            win.configure(bg=pal["bg"])
        except tk.TclError:
            pass
        iw, ih = full_pil.size
        try:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        except Exception:
            sw, sh = 1440, 900
        ww = min(max(iw + 80, 720), int(sw * 0.9))
        wh = min(max(ih + 140, 540), int(sh * 0.9))
        win.geometry(f"{ww}x{wh}")

        # ---- Top metadata bar ----
        top = ttk.Frame(win)
        top.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(top, text=f"  原图 · {cid}  ",
                  foreground=pal.get("accent", pal["primary"]),
                  font=("TkDefaultFont", 11, "bold")).pack(
                      side="left", padx=(0, 8))
        chip = pal["chip"]["idle"]
        for label, key in (("scale", "scale"), ("noise", "noise"),
                           ("skew", "skew"), ("channel", "channel")):
            v = rec.get(key, "-")
            ttk.Label(top, text=f"  {label}={v}  ",
                      foreground=chip["fg"], background=chip["bg"],
                      font=("Menlo", 9, "bold")).pack(side="left", padx=4)
        ttk.Label(top, text=f"  {iw}x{ih}  ",
                  foreground="#888",
                  font=("Menlo", 9)).pack(side="right", padx=8)

        # ---- Zoom controls ----
        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=8, pady=4)
        status_var = tk.StringVar(value="zoom 1.0x")
        ttk.Label(bar, textvariable=status_var,
                  foreground="#888",
                  font=("Menlo", 9)).pack(side="right", padx=8)
        zoom_state = {"factor": 1.0}

        # ---- Canvas + scrollbars ----
        cf = ttk.Frame(win, relief="solid", borderwidth=1)
        cf.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        canvas = tk.Canvas(cf, bg=pal["code"], highlightthickness=0)
        xscroll = ttk.Scrollbar(cf, orient="horizontal",
                                command=canvas.xview)
        yscroll = ttk.Scrollbar(cf, orient="vertical",
                                command=canvas.yview)
        canvas.configure(xscrollcommand=xscroll.set,
                         yscrollcommand=yscroll.set)
        xscroll.pack(side="bottom", fill="x")
        yscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        photo_ref = {"img": None}

        def render(factor):
            factor = max(0.1, min(factor, 8.0))
            zoom_state["factor"] = factor
            status_var.set(f"zoom {factor:.2f}x")
            zw = max(1, int(iw * factor))
            zh = max(1, int(ih * factor))
            from PIL import Image as _PI, ImageTk as _IT
            if factor == 1.0:
                photo = _IT.PhotoImage(full_pil)
            else:
                method = _PI.LANCZOS if factor < 1.0 else _PI.BICUBIC
                photo = _IT.PhotoImage(
                    full_pil.resize((zw, zh), method))
            photo_ref["img"] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.configure(scrollregion=(0, 0, zw, zh))

        def fit_to_window():
            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 1)
            ch = max(canvas.winfo_height(), 1)
            render(min(cw / iw, ch / ih))

        def set_zoom(f):
            render(f)

        for label, value in (("Fit", None), ("1x", 1.0),
                             ("2x", 2.0), ("4x", 4.0)):
            cmd = fit_to_window if value is None else (lambda v=value: set_zoom(v))
            ttk.Button(bar, text=label, command=cmd, width=6).pack(
                side="left", padx=2)

        def on_wheel(evt):
            if evt.state & 0x4 or evt.state & 0x0008:
                delta = evt.delta if hasattr(evt, "delta") else 0
                step = 1.1 if delta > 0 else 1 / 1.1
                render(zoom_state["factor"] * step)
                return "break"
            if getattr(evt, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(evt, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            else:
                d = -1 if evt.delta > 0 else 1
                canvas.yview_scroll(d * 3, "units")
            return "break"
        canvas.bind("<MouseWheel>", on_wheel)
        canvas.bind("<Button-4>", on_wheel)
        canvas.bind("<Button-5>", on_wheel)

        canvas.update_idletasks()
        fit_to_window()

    def _gt_for(self, name):
        """Return (line1, line2) GT for a real photo if hand-labeled."""
        try:
            p = Path(__file__).parent / "real_gt.json"
            if not p.exists():
                return None
            d = json.loads(p.read_text(encoding="utf-8"))
            g = (d.get("gt") or {}).get(name)
            if g and g.get("line1") and g.get("line2"):
                return (g["line1"], g["line2"])
        except Exception:
            pass
        return None

    def _ocr_on_detail_select(self, _evt=None):
        sel = self.ocr_detail.selection()
        if not sel:
            return
        kind, name = self._ocr_detail_map[sel[0]]
        if kind == "photo":
            from passport_pipeline import PIC_DIR
            path = PIC_DIR / name
            self.ocr_preview_meta.set(f"{name}")
            self._ocr_meta_fullpath = str(path)
            self._ocr_meta_tip.set_text(self._ocr_meta_fullpath)
            rec = next((r for r in getattr(self, "_ocr_last", {}).get("photo", [])
                        if r.get("file") == name), None)
            # Render the photo + region overlay (PHOTO/DATA/MRZ boxes)
            # so the user can see exactly where the locator placed the
            # three regions on the actual image.
            try:
                from PIL import Image
                pil = Image.open(path).convert("RGB")
                # Annotate with neon rectangles when we have a locate
                # result; otherwise show the plain image.
                loc = (rec or {}).get("loc") if rec else None
                if loc and self._loc_annotate is not None:
                    try:
                        pil_show = self._loc_annotate(pil, loc, scale=1.0)
                    except Exception:
                        pil_show = pil
                else:
                    pil_show = pil
                self._preview_img_pil = pil_show
                self._preview_w, self._preview_h = pil_show.size
                self._preview_img_path = path
                # Cache confidence header from the rec (set by
                # passport_pipeline: result.conf1 / conf2 if present).
                # The OCR was run on the cropped MRZ band, so the band
                # coordinates are local to the crop. Recover the crop
                # offset from the OCR record so the overlay can place
                # the MRZ rectangle in the correct absolute location.
                crop = ((rec or {}).get("ocr") or {}).get("crop") or {}
                try:
                    crop_offset = (int(crop.get("x", 0)),
                                   int(crop.get("y", 0)))
                except Exception:
                    crop_offset = (0, 0)
                self._preview_last_pm = {
                    "conf1": (rec or {}).get("conf1", 0) or 0,
                    "conf2": (rec or {}).get("conf2", 0) or 0,
                    "band":  (rec or {}).get("band", "") or "",
                    "line1": (rec or {}).get("line1", "") or "",
                    "line2": (rec or {}).get("line2", "") or "",
                    "crop_offset": crop_offset,
                }
                self._update_preview_image()
            except Exception as e:
                self.ocr_diff_text.configure(state="normal")
                self.ocr_diff_text.delete("1.0", "end")
                self.ocr_diff_text.insert("end",
                    f"无法加载 {path}: {e}", ("mm",))
                self.ocr_diff_text.configure(state="disabled")
                return
            # Render the structured report in the diff pane.
            tw = self.ocr_diff_text
            tw.configure(state="normal")
            tw.delete("1.0", "end")
            tw.insert("end", f"样本  : {name}\n", "hdr")
            tw.insert("end", f"路径  : {path}\n", "lab")
            if rec is None:
                tw.insert("end", "\n（未找到识别结果，请先点击 ⚡ 开始识别）",
                          "lab")
                tw.configure(state="disabled")
                return
            grade = rec.get("grade", "?")
            ok_chip = "okline" if grade == "PASS" else "mm"
            tw.insert("end",
                f"判定  : {'PASS  ✅' if grade == 'PASS' else grade + '  ⚠️'}\n",
                ok_chip)
            tw.insert("end",
                f"耗时  : {rec.get('ms', 0):.0f} ms\n", "lab")
            tw.insert("end",
                f"OCR校验: {'通过' if rec.get('verified') else '未通过'}\n",
                "lab")
            tw.insert("end", "\n护照定位 (photo / data / mrz):\n", "hdr")
            loc = rec.get("loc") or {}
            for k_zh, k_en in (("照片区", "photo"),
                               ("数据区", "data"),
                               ("MRZ 区", "mrz")):
                r = loc.get(k_en)
                if r:
                    tw.insert("end",
                        f"  {k_zh}: x={r['x']} y={r['y']} "
                        f"w={r['w']} h={r['h']}   "
                        f"置信度 {r['confidence']:.2f}\n", "lab")
                else:
                    tw.insert("end", f"  {k_zh}: 未定位\n", "mm")
            tw.insert("end", "\nMRZ 识别结果:\n", "hdr")
            fields = rec.get("fields") or {}
            if fields:
                for k_zh, k_en in (
                        ("姓名", "name"), ("证件号", "doc"),
                        ("国籍", "nat"), ("出生", "dob"),
                        ("有效期", "exp"), ("性别", "sex")):
                    val = fields.get(k_en, "-")
                    tw.insert("end", f"  {k_zh}: {val}\n", "lab")
            else:
                tw.insert("end",
                    "  （passport_pipeline 未返回解析字段）\n", "lab")
            c1 = rec.get("conf1", 0) or 0
            c2 = rec.get("conf2", 0) or 0
            tw.insert("end", "\n置信率:\n", "hdr")
            tw.insert("end", f"  line1 conf={c1}% "
                f"({'PASS' if c1 >= 70 else 'WARN' if c1 >= 40 else 'FAIL'})\n",
                "lab")
            tw.insert("end", f"  line2 conf={c2}% "
                f"({'PASS' if c2 >= 70 else 'WARN' if c2 >= 40 else 'FAIL'})\n",
                "lab")
            gt = self._gt_for(name)
            l1 = rec.get("line1", "") or ""
            l2 = rec.get("line2", "") or ""
            if gt:
                self._render_diff_block(tw, "Line1 咬合 (GT vs Pred)", gt[0], l1)
                self._render_diff_block(tw, "Line2 咬合 (GT vs Pred)", gt[1], l2)
            else:
                tw.insert("end", f"\nLine1 (44): {l1}\n", "lab")
                tw.insert("end", f"Line2 (44): {l2}\n", "lab")
            tw.insert("end", f"\n说明: {rec.get('reason', '')}\n", "lab")
            tw.configure(state="disabled")
            return
        cid, m = kind, name
        rec = next((r for r in self._ocr_corpus["records"]
                    if r["id"] == cid), None)
        if rec is None:
            return
        self._load_preview(cid, rec)
        r = getattr(self, "_ocr_last", {}).get(cid, {}).get(m, {})
        label = METHOD_LABELS.get(m, m)
        self._show_diff([(f"line1 [{label}]", rec["line1"], r.get("line1", "")),
                         (f"line2 [{label}]", rec["line2"], r.get("line2", ""))])

    def _ocr_request_stop(self):
        self.ocr_stop_flag = True

    def _ocr_run(self):
        if self.ocr_photo_mode.get():
            self._ocr_run_photos()
            return
        selected = list(self.ocr_case_list.selection())
        if self.ocr_method_both.get():
            methods = list(METHOD_ORDER)
        else:
            methods = [self.ocr_methods.get()]
        methods = [m for m in methods if m]
        if not methods:
            messagebox.showerror("错误", "请选择至少一种方案")
            return
        if not selected:
            messagebox.showerror("错误", "请选择至少一个测试案例")
            return
        self.ocr_stop_flag = False
        for r in self.ocr_detail.get_children():
            self.ocr_detail.delete(r)
        for r in self.ocr_summary.get_children():
            self.ocr_summary.delete(r)
        self._ocr_detail_map = {}
        for v in self.ocr_metrics.values():
            v.set("--")
        self.ocr_run_btn.configure(state="disabled")
        self._set_status_bar(f"准备运行 {len(selected)} 个案例 ...", None)

        def worker():
            from pathlib import Path as _P
            runner = _P(__file__).resolve().parent / "ocr_bench_runner.py"
            argv = [sys.executable, str(runner), "--progress",
                    ",".join(methods), *selected]
            t0 = time.monotonic()
            try:
                proc = subprocess.Popen(argv, text=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                lines = []
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    if time.monotonic() - t0 > 900:
                        proc.kill()
                        raise TimeoutError("运行超时 (900s)")
                    if line.startswith("@@PROGRESS@"):
                        parts = line.split("@")
                        try:
                            done, total = int(parts[2]), int(parts[3])
                        except (IndexError, ValueError):
                            continue
                        self.after(0, lambda d=done, t=total:
                                   self._set_status_bar(
                                       f"正在测试 {d}/{t} 案例",
                                       100.0 * d / t if t else 0))
                    else:
                        lines.append(line)
                proc.wait()
                payload = json.loads("\n".join(lines))
            except Exception as e:
                self.after(0, lambda e=e: self._ocr_run_failed(e))
                return
            self.after(0, lambda: self._ocr_populate_results(payload))
        threading.Thread(target=worker, daemon=True).start()

    def _ocr_run_failed(self, e):
        self.ocr_run_btn.configure(state="normal")
        self._set_status_bar(f"运行失败: {e}", 0)

    def _ocr_run_photos(self):
        """Passport-photo mode: locate -> OCR -> verify for every photo."""
        selected = list(self.ocr_case_list.selection())
        if not selected:
            messagebox.showerror("错误", "请选择至少一张护照照片")
            return
        for r in self.ocr_detail.get_children():
            self.ocr_detail.delete(r)
        for r in self.ocr_summary.get_children():
            self.ocr_summary.delete(r)
        self._ocr_detail_map = {}
        for v in self.ocr_metrics.values():
            v.set("--")
        self.ocr_run_btn.configure(state="disabled")
        self._set_status_bar(f"准备识别 {len(selected)} 张照片 ...", None)

        def worker():
            try:
                from passport_pipeline import process_image, CNN, PIC_DIR
                results = []
                for i, name in enumerate(selected, 1):
                    rec = process_image(PIC_DIR / name, tool=CNN)
                    results.append(rec)
                    self.after(0, lambda i=i, n=len(selected), nm=name,
                               rec=rec: (
                        self._set_status_bar(
                            f"正在识别 [{i}/{n}] {nm}  定位:"
                            f" {'MRZ' if (rec.get('loc') or {}).get('mrz') else '无'}"
                            f" 判定: {rec['grade']}", 100.0 * i / n,
                            done=i, total=n)))
                # Final 100% bar when the photo pass completes.
                self.after(0, lambda: self._ocr_populate_photos(results))
                self.after(0, lambda n=len(results):
                           self._set_status_bar(
                               f"完成 {n} 张照片识别", 100,
                               done=n, total=n))
            except Exception as e:  # noqa: BLE001
                import traceback as _tb
                _tb.print_exc()
                self.after(0, lambda e=e: self._ocr_run_failed(e))

        threading.Thread(target=worker, daemon=True).start()

    def _ocr_populate_photos(self, results):
        """Render photo-recognition results into detail/summary/metrics,
        then auto-select the first row so the right pane shows the photo
        + region overlay + MRZ result immediately."""
        self.ocr_detail.configure(columns=("file", "grade", "verified", "ms",
                                           "reason"))
        for c, anc in [("file", "w"), ("grade", "center"),
                       ("verified", "center"), ("ms", "center"),
                       ("reason", "w")]:
            self.ocr_detail.heading(c, text={"file": "照片文件",
                                             "grade": "判定",
                                             "verified": "OCR校验",
                                             "ms": "耗时ms",
                                             "reason": "说明"}[c])
            self.ocr_detail.column(c, width=90, anchor=anc)
        self.ocr_detail.column("file", width=240)
        self.ocr_detail.column("reason", width=260)
        for tag, spec in (("PASS", self._pal["tag_ok"]),
                          ("SUSPECT", self._pal["tag_warn"]),
                          ("REJECT", self._pal["tag_fail"])):
            bg, fg = spec["bg"], spec["fg"]
            self.ocr_detail.tag_configure(tag, background=bg, foreground=fg)
        stats = {"PASS": 0, "SUSPECT": 0, "REJECT": 0, "verified": 0}
        ms_sum = 0.0
        self._ocr_last = {"photo": results}
        for rec in results:
            g = rec["grade"]
            stats[g] = stats.get(g, 0) + 1
            if rec.get("verified"):
                stats["verified"] += 1
            ms_sum += rec["ms"]
            iid = rec["file"]
            self.ocr_detail.insert("", "end", iid=iid, tags=(g,),
                values=(rec["file"], g,
                        "✓" if rec.get("verified") else "—",
                        f"{rec['ms']:.0f}", rec.get("reason") or ""))
            self._ocr_detail_map[iid] = ("photo", rec["file"])
        # Auto-select first photo row so the right pane refreshes with
        # thumbnail + region overlay + MRZ result.
        if results:
            try:
                first = self.ocr_detail.get_children()
                if first:
                    self.ocr_detail.selection_set(first[0])
                    self.ocr_detail.focus(first[0])
                    self._ocr_on_detail_select()
            except tk.TclError:
                pass
        self.ocr_summary.configure(columns=("grade", "n", "verified"))
        for c in ("grade", "n", "verified"):
            self.ocr_summary.heading(c, text={"grade": "判定",
                                              "n": "数量",
                                              "verified": "OCR校验通过"}[c])
            self.ocr_summary.column(c, width=110, anchor="center")
        for g in ("PASS", "SUSPECT", "REJECT"):
            self.ocr_summary.insert("", "end", iid=g, tags=("row_ok" if g == "PASS"
                                                            else "row_fail"),
                values=(g, stats[g],
                        stats["verified"] if g == "PASS" else ""))
        m = self.ocr_metrics
        m["cases"].set(len(results))
        m["mean"].set(f"{ms_sum / max(1, len(results)):.1f}")
        m["pass"].set(f"{100.0 * stats['PASS'] / max(1, len(results)):.1f}")
        for k in ("p95", "l1", "l2", "full"):
            m[k].set("--")
        self.ocr_run_btn.configure(state="normal")
        self._set_status_bar(
            f"识别完成：PASS={stats['PASS']} SUSPECT={stats['SUSPECT']}"
            f" REJECT={stats['REJECT']}，OCR校验通过 {stats['verified']} 张",
            100)

    def _ocr_populate_results(self, payload):
        methods = payload["methods"]
        agg = payload["agg"]
        # Cache per-case results for the single-sample diff view.
        self._ocr_last = {}
        for rec in payload["records"]:
            d = {}
            for m in methods:
                d[m] = rec["per_method"][m]["raw"]
            self._ocr_last[rec["id"]] = d
        # Metrics card (pooled across methods).
        recs = payload["records"]
        all_ms = sorted(r["per_method"][m]["raw"]["ms"]
                        for r in recs for m in methods)
        n_ms = len(all_ms)
        mean = sum(all_ms) / n_ms if n_ms else 0.0
        p95 = all_ms[int(n_ms * 0.95)] if n_ms else 0.0
        l1c = l2c = l1t = l2t = okc = full = 0
        for m in methods:
            a = agg[m]
            l1c += a["l1_correct"]; l1t += a["l1_total"]
            l2c += a["l2_correct"]; l2t += a["l2_total"]
            okc += a["ok"]; full += a["full_match"]
        total_n = sum(a["n"] for a in agg.values())
        self.ocr_metrics["cases"].set(str(payload["total_cases"]))
        self.ocr_metrics["mean"].set(f"{mean:.1f}")
        self.ocr_metrics["p95"].set(f"{p95:.1f}")
        self.ocr_metrics["pass"].set(
            f"{100.0 * okc / total_n:.1f}" if total_n else "0")
        self.ocr_metrics["l1"].set(
            f"{100.0 * l1c / l1t:.2f}" if l1t else "--")
        self.ocr_metrics["l2"].set(
            f"{100.0 * l2c / l2t:.2f}" if l2t else "--")
        self.ocr_metrics["full"].set(str(full))
        # Per-method summary.
        for m in methods:
            a = agg[m]
            ms_avg = a["ms_total"] / a["n"] if a["n"] else 0
            ok_pct = 100.0 * a["ok"] / a["n"] if a["n"] else 0
            l1 = 100.0 * a["l1_correct"] / a["l1_total"] if a["l1_total"] else 0
            l2 = 100.0 * a["l2_correct"] / a["l2_total"] if a["l2_total"] else 0
            label = METHOD_LABELS.get(m, m)
            tag = "row_ok" if a["ok"] == a["n"] else "row_fail"
            self.ocr_summary.insert("", "end", tags=(tag,), values=(
                label, a["n"], a["ok"], f"{ok_pct:.1f}%",
                f"{ms_avg:.2f}", f"{l1:.2f}%", f"{l2:.2f}%",
                a["full_match"]))
        self._fit_columns(self.ocr_summary, cap=140)
        # Per-case x method detail rows (compact; full text in right pane).
        for rec in payload["records"]:
            for m in methods:
                pm = rec["per_method"][m]
                r = pm["raw"]
                ok = r["ok"]
                l1p = 100.0 * pm["l1_correct"] / pm["l1_total"] if pm["l1_total"] else 0
                l2p = 100.0 * pm["l2_correct"] / pm["l2_total"] if pm["l2_total"] else 0
                iid = f"{rec['id']}::{m}"
                self._ocr_detail_map[iid] = (rec["id"], m)
                self.ocr_detail.insert("", "end", iid=iid,
                    tags=(ok and "OK" or "FAIL",),
                    values=(rec["id"],
                            METHOD_LABELS.get(m, m),
                            "PASS" if ok else "FAIL",
                            f"{r['ms']:.2f}", f"{l1p:.2f}%", f"{l2p:.2f}%"))
        self._fit_columns(self.ocr_detail, cap=160)
        self.ocr_run_btn.configure(state="normal")
        self._set_status_bar(
            f"完成 {payload['total_cases']} 个案例，方案={','.join(methods)}，"
            f"平均耗时 {mean:.1f} ms", 100)

    # ---------------- NFC tab ----------------
    def _build_nfc_tab(self):
        f = self.tab_nfc

        # ---- Top: compact control bar (backend + script + run) ----
        top = ttk.Frame(f); top.pack(fill="x", padx=4, pady=2)
        ttk.Label(top, text="后端:").pack(side="left")
        self.nfc_backend_var = tk.StringVar(value="Mock 卡片")
        ttk.Combobox(top, textvariable=self.nfc_backend_var, state="readonly",
                     values=("Mock 卡片", "真实读卡器"), width=10,
                     style="Field.TCombobox").pack(side="left")
        ttk.Label(top, text="脚本:").pack(side="left", padx=(8, 2))
        self.nfc_script_var = tk.StringVar(
            value=str(ROOT / "4_nfc_reader" / "data" / "script_happy.json"))
        ttk.Entry(top, textvariable=self.nfc_script_var, width=52,
                  style="Field.TEntry").pack(
            side="left", fill="x", expand=True)
        self._cta(top, "▤ 浏览…", self._nfc_pick_script,
                  kind="ghost").pack(side="left")
        self.nfc_run_btn = self._cta(top, "⚡ 开始执行", self._nfc_run)
        self.nfc_run_btn.pack(side="left", padx=(6, 0))

        # ---- MRZ compact row ----
        mrz = ttk.Frame(f); mrz.pack(fill="x", padx=4, pady=1)
        ttk.Label(mrz, text="line1:").pack(side="left")
        self.nfc_mrz1_var = tk.StringVar(value=SAMPLE_MRZ.splitlines()[0])
        ttk.Entry(mrz, textvariable=self.nfc_mrz1_var, width=46,
                  font=("Menlo", 9), style="Field.TEntry").pack(side="left", padx=(2, 6))
        ttk.Label(mrz, text="line2:").pack(side="left")
        self.nfc_mrz2_var = tk.StringVar(value=SAMPLE_MRZ.splitlines()[1])
        ttk.Entry(mrz, textvariable=self.nfc_mrz2_var, width=46,
                  font=("Menlo", 9), style="Field.TEntry").pack(side="left", padx=(2, 6))
        self._cta(mrz, "从 OCR 填充",
                  self._nfc_fill_mrz_from_ocr, kind="ghost").pack(side="left", padx=2)
        self._cta(mrz, "内置样例",
                  self._nfc_fill_mrz_sample, kind="ghost").pack(side="left")
        self.nfc_badge_var = tk.StringVar(value="")
        # Visual-pass 2.0: chip-style badge with explicit bg so it never
        # falls back to the platform default (the historical white square).
        pal = self._pal
        self.nfc_badge = tk.Label(
            mrz, textvariable=self.nfc_badge_var,
            bg=pal["bg"], fg=pal["chip"]["dim"]["fg"],
            font=pal["font_ui_bold"], padx=8, pady=2)
        self.nfc_badge.pack(side="left", padx=8)

        self.nfc_status_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.nfc_status_var,
                  font=("TkDefaultFont", 9)).pack(anchor="w", padx=4)

        # ---- Middle: splitter (state machine | APDU trace + inspect) ----
        body = ttk.PanedWindow(f, orient="horizontal")
        body.pack(fill="both", expand=True, padx=4, pady=2)

        left = ttk.Frame(body)
        body.add(left, weight=1)
        self._nfc_build_state_panel(left)
        self._nfc_build_keys_panel(left)

        right = ttk.Frame(body)
        body.add(right, weight=2)
        self._nfc_build_trace_panel(right)

        # ---- Bottom: results cards ----
        bottom = ttk.Frame(f); bottom.pack(fill="x", padx=4, pady=2)
        cards = ttk.Frame(bottom); cards.pack(fill="x")
        self._nfc_build_dg1_card(cards)
        self._nfc_build_sod_card(cards)
        self._nfc_build_progress_card(cards)
        self._nfc_build_adv_panel(bottom)

    def _nfc_build_state_panel(self, parent):
        lab = ttk.LabelFrame(parent, text="密码学与认证状态机")
        lab.pack(fill="both", expand=True, padx=2, pady=2)
        pal = self._pal
        chip = pal["chip"]
        self._nfc_states = []
        for title in ("① MRZ 因子校验", "② 基础密钥派生 Kseed/Kenc/Kmac",
                      "③ 相互认证 (BAC)", "④ Secure Messaging Ready"):
            row = ttk.Frame(lab); row.pack(fill="x", padx=6, pady=2)
            ttk.Label(row, text=title, anchor="w").pack(side="left")
            var = tk.StringVar(value=" PENDING ")
            lb = tk.Label(
                row, textvariable=var,
                bg=pal["bg"], fg=chip["idle"]["fg"],
                font=pal["font_ui_bold"], padx=6, pady=1)
            lb.pack(side="right")
            self._nfc_states.append((title, var, lb))

    def _nfc_build_keys_panel(self, parent):
        lab = ttk.LabelFrame(parent, text="会话密钥（点击复制）")
        lab.pack(fill="x", padx=2, pady=2)
        pal = self._pal
        self._nfc_key_vars = {}
        for name in ("Kseed", "Kenc", "Kmac", "KSenc", "KSmac"):
            row = ttk.Frame(lab); row.pack(fill="x", padx=4, pady=1)
            ttk.Label(row, text=name, width=6).pack(side="left")
            var = tk.StringVar(value="—")
            en = tk.Entry(row, textvariable=var, width=34, font=("Menlo", 9),
                          state="readonly",
                          readonlybackground=pal["field"], fg=pal["accent"])
            en.pack(side="left", padx=2)
            self._nfc_key_entries[name] = en
            self._cta(row, "⧉ 复制",
                      lambda n=name, v=var: self._nfc_copy_key(n, v),
                      kind="ghost").pack(side="left")
            self._nfc_key_vars[name] = var

    def _nfc_build_trace_panel(self, parent):
        insp = ttk.LabelFrame(parent, text="报文透视（Raw / Decrypted / MAC）")
        insp.pack(fill="x", side="bottom", padx=2, pady=2)
        pal = self._pal
        self.nfc_inspect = tk.Text(
            insp, height=7, font=("Menlo", 9), wrap="none",
            bg=pal["code"], fg=pal["fg"],
            insertbackground=pal.get("accent", pal["primary"]))
        yi = ttk.Scrollbar(insp, orient="vertical",
                           command=self.nfc_inspect.yview)
        xi = ttk.Scrollbar(insp, orient="horizontal",
                           command=self.nfc_inspect.xview)
        self.nfc_inspect.configure(yscrollcommand=yi.set,
                                   xscrollcommand=xi.set)
        xi.pack(side="bottom", fill="x")
        self.nfc_inspect.pack(side="left", fill="both", expand=True)
        yi.pack(side="right", fill="y")

        lab = ttk.LabelFrame(parent, text="APDU 实时跟踪（点行查看报文透视）")
        lab.pack(fill="both", expand=True, padx=2, pady=2)
        self.nfc_trace = ttk.Treeview(
            lab, columns=("i", "cmd", "hex", "sw", "ms"),
            show="headings", style="Data.Treeview")
        for c, w, t in [("i", 30, "#"), ("cmd", 150, "Cmd / Rsp"),
                        ("hex", 340, "Hex 报文 (C → R)"), ("sw", 56, "SW"),
                        ("ms", 48, "耗时")]:
            self.nfc_trace.heading(c, text=t)
            self.nfc_trace.column(c, width=w, stretch=(c == "hex"),
                                  anchor="center" if c in ("i", "sw", "ms") else "w")
        self.nfc_trace.tag_configure("sw_ok", foreground="#3FB950",
                                     font=("Menlo", 10, "bold"))
        self.nfc_trace.tag_configure("sw_bad", foreground="#F85149",
                                     font=("Menlo", 10, "bold"))
        ysb = ttk.Scrollbar(lab, orient="vertical", command=self.nfc_trace.yview)
        self.nfc_trace.configure(yscrollcommand=ysb.set)
        self.nfc_trace.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.nfc_trace.bind("<<TreeviewSelect>>", self._nfc_on_trace_select)

    def _nfc_build_dg1_card(self, parent):
        card = ttk.LabelFrame(parent, text="DG1 持证人信息")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_dg1_vars = {}
        for row, (key, label) in enumerate(
                [("name", "姓名"), ("doc", "证件号"), ("nat", "国籍"),
                 ("dob", "出生日期"), ("exp", "有效期")]):
            ttk.Label(card, text=label + ":").grid(
                row=row, column=0, sticky="e", padx=(6, 2))
            var = tk.StringVar(value="—")
            lb = tk.Label(card, textvariable=var, font=("Menlo", 9, "bold"),
                          bg=self._pal["bg"], fg=self._pal["fg"])
            lb.grid(
                row=row, column=1, sticky="w", padx=(2, 6))
            self._nfc_dg1_lbs[key] = lb
            self._nfc_dg1_vars[key] = var

    def _nfc_build_sod_card(self, parent):
        card = ttk.LabelFrame(parent, text="SOD 被动认证 (Passive Auth)")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_sod_sha = tk.StringVar(value="—")
        self._nfc_sod_rsa = tk.StringVar(value="—")
        ttk.Label(card, text="DG1 SHA-256:").grid(
            row=0, column=0, sticky="e", padx=(6, 2))
        tk.Label(card, textvariable=self._nfc_sod_sha,
                 font=("Menlo", 9),
                 bg=self._pal["bg"], fg=self._pal["fg"]).grid(row=0, column=1, sticky="w")
        ttk.Label(card, text="RSA 验签:").grid(
            row=1, column=0, sticky="e", padx=(6, 2))
        tk.Label(card, textvariable=self._nfc_sod_rsa,
                 font=("Menlo", 9),
                 bg=self._pal["bg"], fg=self._pal["fg"]).grid(row=1, column=1, sticky="w")

    def _nfc_build_progress_card(self, parent):
        card = ttk.LabelFrame(parent, text="读取进度")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_prog_vars = {}
        for i, ef in enumerate(("EF.COM", "EF.DG1", "EF.DG2", "EF.SOD")):
            r, c = divmod(i, 2)
            var = tk.StringVar(value="—")
            lb = tk.Label(card, textvariable=var,
                          font=("TkDefaultFont", 9, "bold"),
                          bg=self._pal["bg"], fg=self._pal["fg"])
            lb.grid(row=r, column=c * 2, sticky="e", padx=(6, 2))
            self._nfc_prog_lbs[ef] = lb
            ttk.Label(card, text=ef).grid(row=r, column=c * 2 + 1,
                                          sticky="w", padx=(0, 10))
            self._nfc_prog_vars[ef] = (var, lb)

    def _nfc_build_adv_panel(self, parent):
        self._adv_open = tk.BooleanVar(value=False)
        self._adv_frame = ttk.LabelFrame(parent, text="高级密码学参数")
        self._adv_frame.pack(fill="x", padx=2, pady=2)
        head = ttk.Frame(self._adv_frame); head.pack(fill="x")
        self._adv_toggle_btn = ttk.Button(
            head, text="▾ 展开", command=self._nfc_toggle_adv)
        self._adv_toggle_btn.pack(side="left", padx=4)
        # wrap="none" keeps hex lines intact, but the previous height=4
        # without scrollbars clipped long C-APDU/R-APDU strings at the
        # right edge with no way to reach them. Give the panel both
        # axes of scrolling and a sensible starting height.
        pal = self._pal
        body = ttk.Frame(self._adv_frame)
        self._adv_text_body = body
        # Create the Text widget FIRST, then bind the scrollbars to its
        # xview/yview. The previous order raised AttributeError because
        # lambdas captured ``self.nfc_adv_text`` before it existed.
        self.nfc_adv_text = tk.Text(
            body, height=8, font=("Menlo", 9), wrap="none",
            bg=pal["code"], fg=pal["fg"],
            insertbackground=pal.get("accent", pal["primary"]),
            undo=True, maxundo=-1)
        adv_yscroll = ttk.Scrollbar(body, orient="vertical",
                                    command=self.nfc_adv_text.yview)
        adv_xscroll = ttk.Scrollbar(body, orient="horizontal",
                                    command=self.nfc_adv_text.xview)
        self.nfc_adv_text.configure(yscrollcommand=adv_yscroll.set,
                                    xscrollcommand=adv_xscroll.set)
        adv_xscroll.pack(side="bottom", fill="x")
        self.nfc_adv_text.pack(side="left", fill="both", expand=True)
        adv_yscroll.pack(side="right", fill="y")
        body.pack(fill="x", padx=4, pady=2)
        body.pack_forget()

    def _nfc_toggle_adv(self):
        body = getattr(self, "_adv_text_body", None)
        if body is None:
            return
        btn = getattr(self, "_adv_toggle_btn", None)
        # Use the actual mapped state of the body rather than the cached
        # BooleanVar, so the toggle stays correct after external forgets
        # (e.g. parent re-layouts or theme refreshes).
        is_mapped = bool(body.winfo_ismapped())
        if is_mapped:
            body.pack_forget()
            self._adv_open.set(False)
            if btn is not None:
                try:
                    btn.configure(text="▾ 展开")
                except tk.TclError:
                    pass
        else:
            body.pack(fill="x", padx=4, pady=2)
            self._adv_open.set(True)
            if btn is not None:
                try:
                    btn.configure(text="▴ 折叠")
                except tk.TclError:
                    pass

    def _nfc_copy_key(self, name, var):
        val = var.get()
        if val and val != "—":
            self.clipboard_clear()
            self.clipboard_append(val)
            self.nfc_status_var.set(f"已复制 {name} → 剪贴板")

    def _nfc_pick_script(self):
        path = filedialog.askopenfilename(
            initialdir=str(ROOT / "4_nfc_reader" / "data"),
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.nfc_script_var.set(path)

    def _nfc_fill_mrz_sample(self):
        self.nfc_mrz1_var.set(SAMPLE_MRZ.splitlines()[0])
        self.nfc_mrz2_var.set(SAMPLE_MRZ.splitlines()[1])

    def _nfc_fill_mrz_from_ocr(self):
        """Grab line1/line2 from the last OCR run (first method's raw)."""
        last = getattr(self, "_ocr_last", {})
        for cid, d in last.items():
            for m, raw in d.items():
                if raw.get("line1") and raw.get("line2"):
                    self.nfc_mrz1_var.set(raw["line1"])
                    self.nfc_mrz2_var.set(raw["line2"])
                    self.nfc_status_var.set(
                        f"已从 OCR 结果填充 MRZ（案例 {cid}）")
                    return
        self.nfc_status_var.set("OCR 结果中没有可用 MRZ，改用内置样例")
        self._nfc_fill_mrz_sample()

    def _nfc_run(self):
        script = self.nfc_script_var.get().strip()
        if not script or not Path(script).exists():
            messagebox.showerror("错误", f"找不到 mock 脚本:\n{script}")
            return
        l1 = self.nfc_mrz1_var.get().strip()
        l2 = self.nfc_mrz2_var.get().strip()
        if not l1 or not l2:
            messagebox.showerror("错误", "请先填写 line1/line2 两行 MRZ")
            return
        for r in self.nfc_trace.get_children():
            self.nfc_trace.delete(r)
        self._nfc_trace_data = []
        for w in (self.nfc_inspect, self.nfc_adv_text):
            w.delete("1.0", "end")
        for var in self._nfc_key_vars.values():
            var.set("—")
        chip_dim = self._pal["chip"]["dim"]
        for _, var, lb in self._nfc_states:
            var.set("  ARMED  ")
            lb.configure(bg=chip_dim["bg"], fg=chip_dim["fg"])
        for var in self._nfc_dg1_vars.values():
            var.set("—")
        self._nfc_sod_sha.set("—")
        self._nfc_sod_rsa.set("—")
        for _, lb in self._nfc_prog_vars.values():
            lb.configure(bg=chip_dim["bg"], fg=chip_dim["fg"])
        self.nfc_badge_var.set("")
        self.nfc_status_var.set("运行中…")
        def worker():
            try:
                proc = subprocess.run(
                    [str(NFC_TOOL), script, l1, l2],
                    text=True, capture_output=True, timeout=30)
                rc = proc.returncode
                out = proc.stdout
                err = proc.stderr
            except Exception as e:
                self.after(0, lambda e=e: self.nfc_status_var.set(f"运行失败: {e}"))
                return
            self.after(0, lambda: self._nfc_show(rc, out, err))
        threading.Thread(target=worker, daemon=True).start()

    def _nfc_show(self, rc: int, stdout: str, stderr: str):
        """Render full BAC flow from nfc_tool output: state machine, keys,
        APDU trace (with click-to-inspect), DG1 card, SOD passive auth,
        read progress, and raw payloads."""
        ok = (rc == 0)
        self.nfc_status_var.set(
            f"\u2713 全部成功（{rc}）" if ok else f"\u2717 异常退出 (rc={rc})")
        chip = self._pal["chip"]
        if ok:
            self.nfc_badge_var.set(" ✓ BAC PASS ")
            self.nfc_badge.configure(bg=chip["ok"]["bg"],
                                     fg=chip["ok"]["fg"])
        else:
            self.nfc_badge_var.set(" ✗ BAC FAIL ")
            self.nfc_badge.configure(bg=chip["fail"]["bg"],
                                     fg=chip["fail"]["fg"])
        kv = {}
        trace = []
        for ln in stdout.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("trace.json"):
                _, _, v = ln.partition(":")
                try:
                    trace = json.loads(v).get("steps", [])
                except Exception:
                    trace = []
                continue
            if ":" not in ln:
                continue
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
        d = kv.get("result.detail", "")

        # ---- BAC 状态机（①-④）----
        states = [
            ("① MRZ 因子校验", bool(kv.get("result.mrz_info"))),
            ("② 基础密钥派生 Kseed/Kenc/Kmac", bool(kv.get("result.kseed"))),
            ("③ 相互认证 (BAC)", bool(kv.get("result.ksenc"))),
            ("④ Secure Messaging Ready", ok),
        ]
        for title, var, lb in self._nfc_states:
            good = dict(states).get(title, False)
            if good:
                var.set(" ✓ PASS ")
                lb.configure(bg=chip["ok"]["bg"], fg=chip["ok"]["fg"])
            else:
                var.set(" ✗ FAIL ")
                lb.configure(bg=chip["fail"]["bg"], fg=chip["fail"]["fg"])

        # ---- 密钥卡片 ----
        for name in ("Kseed", "Kenc", "Kmac", "KSenc", "KSmac"):
            self._nfc_key_vars[name].set(
                kv.get(f"result.{name.lower()}", "—") or "—")

        # ---- APDU 跟踪 ----
        for r in self.nfc_trace.get_children():
            self.nfc_trace.delete(r)
        self._nfc_trace_data = []
        if trace:
            for i, t in enumerate(trace, 1):
                sw = t.get("sw", "")
                tag = ("sw_ok" if sw == "9000"
                   else "sw_bad" if sw else "")
                cap = t.get("capdu", "")
                rsp = t.get("rapdu", "")
                hexmsg = f"{cap} → {rsp}"
                self.nfc_trace.insert("", "end", iid=str(i), values=(
                    i, t.get("name", ""), hexmsg, sw, t.get("ms", 0)),
                    tags=(tag,))
                self._nfc_trace_data.append(t)
        self._nfc_on_trace_select()

        # ---- 高级参数 / 原始载荷 ----
        adv = []
        adv.append(f"result.step    : {kv.get('result.step', '--')}")
        adv.append(f"result.detail  : {d}")
        if kv.get("result.rnd_icc"):
            adv.append(f"rnd_ICC        : {kv['result.rnd_icc']}")
        if kv.get("result.auth1"):
            adv.append(f"C-APDU E||M    : {kv['result.auth1']}")
        if kv.get("result.auth2"):
            adv.append(f"R-APDU         : {kv['result.auth2']}")
        adv.append(f"mock.unexpected: {kv.get('mock.unexpected', '0')}")
        if stderr:
            adv.append("[stderr]")
            adv.append(stderr)
        self.nfc_adv_text.delete("1.0", "end")
        self.nfc_adv_text.insert("1.0", "\n".join(adv))

        # ---- DG1 持证人信息 + SOD + 读取进度 ----
        dg1_lines = []
        for t in trace:
            if t.get("name", "").endswith("(DG1)") and t.get("plain"):
                raw = bytes.fromhex(t["plain"])
                # strip TLV: tag(5A) + len + 44x2 MRZ chars
                if raw and raw[0] == 0x5A and len(raw) >= 3:
                    raw = raw[2:]
                dg1_lines = raw.decode("latin-1").splitlines()
        if dg1_lines and len(dg1_lines) >= 2:
            l1, l2 = dg1_lines[0], dg1_lines[1]
            surname = l1[2:].split("<")[0]
            given = [p for p in l1[2:].split("<")[1:] if p]
            self._nfc_dg1_vars["name"].set(
                f"{surname}, {' '.join(given)}")
            self._nfc_dg1_vars["doc"].set(l2[0:9])
            self._nfc_dg1_vars["nat"].set(l2[10:13])
            self._nfc_dg1_vars["dob"].set(self._fmt_date(l2[13:19]))
            self._nfc_dg1_vars["exp"].set(self._fmt_date(l2[21:27]))
            self._nfc_sod_sha.set(
                hashlib.sha256(raw).hexdigest().upper())
            self._nfc_sod_rsa.set("—（Mock SOD 无证书签名，Passive Auth 跳过）")
        else:
            for k in self._nfc_dg1_vars:
                self._nfc_dg1_vars[k].set("—")
            self._nfc_sod_sha.set("—")
            self._nfc_sod_rsa.set("—")
        sod_ok = bool(kv.get("result.sod_head"))
        chip = self._pal["chip"]
        for ef, (var, lb) in self._nfc_prog_vars.items():
            got = {"EF.COM": False, "EF.DG1": bool(dg1_lines),
                   "EF.DG2": False, "EF.SOD": sod_ok}[ef]
            if got:
                var.set(" ✓ ")
                lb.configure(bg=chip["ok"]["bg"], fg=chip["ok"]["fg"])
            else:
                var.set(" — ")
                lb.configure(bg=chip["idle"]["bg"], fg=chip["idle"]["fg"])

    @staticmethod
    def _fmt_date(yymmdd):
        try:
            yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
            cyy = time.localtime().tm_year % 100
            year = 2000 + yy if yy <= cyy else 1900 + yy
            return f"{year:04d}-{mm:02d}-{dd:02d}"
        except Exception:
            return yymmdd or "—"

    def _nfc_on_trace_select(self, _evt=None):
        """Deep-dive the selected APDU: raw hex, decrypted plaintext, MAC check."""
        sel = self.nfc_trace.selection()
        data = getattr(self, "_nfc_trace_data", [])
        self.nfc_inspect.delete("1.0", "end")
        if not sel or not data:
            self.nfc_inspect.insert("1.0",
                "点击上方 APDU 行查看 Raw / Decrypted / MAC 校验细节")
            return
        t = data[int(sel[0]) - 1]
        sw = t.get("sw", "")
        mac = {1: "✓ CBC-MAC 校验通过", 0: "✗ CBC-MAC 不匹配",
               -1: "n/a（无 MAC）"}.get(t.get("mac_ok"), "—")
        lines = [
            f"命令      : {t.get('name', '')}",
            f"C-APDU    : {t.get('capdu', '')}",
            f"R-APDU    : {t.get('rapdu', '')}",
            f"SW        : {sw}  ({'OK' if sw == '9000' else 'FAIL'})",
            f"耗时      : {t.get('ms', 0)} ms",
            f"MAC       : {mac}",
        ]
        if t.get("plain"):
            lines.append(f"解密明文  : {t['plain']}")
        self.nfc_inspect.insert("1.0", "\n".join(lines))

    # ---------------- MRZ tab ----------------
    def _build_mrz_tab(self):
        f = self.tab_mrz
        left = ttk.Frame(f); left.pack(side="left", fill="both",
                                       expand=True, padx=4, pady=4)
        right = ttk.Frame(f); right.pack(side="right", fill="both",
                                         expand=True, padx=4, pady=4)

        # ---- Left: MRZ text -> decode ----
        ttk.Label(left, text="MRZ 文本（两行 44 字符）",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.mrz_text = tk.Text(left, width=52, height=6, font=("Menlo", 12),
                                bg="#ffffff", fg="#111827")
        self.mrz_text.insert("1.0", SAMPLE_MRZ)
        self.mrz_text.pack(fill="x")
        btn = ttk.Frame(left); btn.pack(fill="x", pady=4)
        ttk.Button(btn, text="解码并校验",
                   command=self._mrz_decode_run).pack(side="left", padx=2)
        ttk.Button(btn, text="编码结果 → 填入左侧",
                   command=self._mrz_encode_run).pack(side="left", padx=2)
        ttk.Button(btn, text="运行防伪验证",
                   command=lambda: (self.tab_ac.focus_set(),
                                    self._ac_run(self.mrz_text.get("1.0", "end")))).pack(side="left", padx=2)
        ttk.Label(left, text="输出", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(8, 0))
        self.mrz_out = tk.Text(left, width=80, height=18, font=("Menlo", 11),
                               bg="#ffffff", fg="#111827")
        self.mrz_out.pack(fill="both", expand=True)

        # ---- Right: 10 fields -> encode ----
        ttk.Label(right, text="TD3 字段 → 编码生成 MRZ",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        fields = [
            ("doc_type",      "证件类型 (P<)",      "P"),
            ("issuing_state", "签发国 (3字母)",     "UTO"),
            ("surname",       "姓",                "ERIKSSON"),
            ("given_names",   "名",                "ANNA MARIA"),
            ("passport_no",   "护照号 (9位)",       "L898902C3"),
            ("nationality",   "国籍 (3字母)",       "UTO"),
            ("birth",         "出生日期 (YYMMDD)",  "690806"),
            ("sex",           "性别 (M/F)",        "F"),
            ("expiry",        "有效期 (YYMMDD)",   "940623"),
            ("personal_no",   "个人号码 (14位)",    "ZE184226B"),
        ]
        self.mrz_fields = {}
        for key, label, default in fields:
            row = ttk.Frame(right); row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=var, width=22,
                  style="Field.TEntry").pack(side="left", fill="x", expand=True)
            self.mrz_fields[key] = var
        ttk.Button(right, text="编码生成 MRZ",
                   command=self._mrz_encode_run).pack(anchor="e", pady=(6, 2))

    def _mrz_decode_run(self):
        """Decode + verify the MRZ text in the left editor (decode subcommand)."""
        text = self.mrz_text.get("1.0", "end").strip()
        if not text:
            messagebox.showerror("错误", "请先输入 MRZ 文本")
            return
        try:
            r = subprocess.run([str(MRZ_TOOL), "decode", text],
                               text=True, capture_output=True, timeout=10)
            self.mrz_out.delete("1.0", "end")
            self.mrz_out.insert("1.0", r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or ""))
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _mrz_encode_run(self):
        """Encode the 10 right-hand fields into an MRZ string, then place
        it back into the left editor (encode subcommand)."""
        vals = [self.mrz_fields[k].get().strip() for k in
                ("doc_type", "issuing_state", "surname", "given_names",
                 "passport_no", "nationality", "birth", "sex", "expiry",
                 "personal_no")]
        try:
            r = subprocess.run([str(MRZ_TOOL), "encode", *vals],
                               text=True, capture_output=True, timeout=10)
            self.mrz_out.delete("1.0", "end")
            self.mrz_out.insert("1.0", r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or ""))
            if r.returncode == 0 and r.stdout.strip():
                self.mrz_text.delete("1.0", "end")
                self.mrz_text.insert("1.0", r.stdout)
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # ---------------- AC tab ----------------
    def _build_ac_tab(self):
        f = self.tab_ac
        ttk.Label(f, text="防伪验证", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=4)
        top = ttk.Frame(f); top.pack(fill="x", padx=4)
        ttk.Label(top, text="(使用 MRZ 解码标签页中的文本)").pack(side="left")
        ttk.Button(top, text="运行验证", command=lambda: self._ac_run(None)).pack(side="right")
        self.ac_out = tk.Text(f, font=("Menlo", 11),
                              bg=self._pal["field"], fg=self._pal["field_fg"])
        self.ac_out.pack(fill="both", expand=True, padx=4, pady=4)

    def _ac_run(self, _text=None):
        text = self.mrz_text.get("1.0", "end") if _text is None else _text
        try:
            # ac_tool reads the MRZ from stdin when invoked with "-".
            r = subprocess.run([str(AC_TOOL), "-"], input=text,
                               text=True, capture_output=True, timeout=10)
            self.ac_out.delete("1.0", "end")
            self.ac_out.insert("1.0", r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or ""))
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # ---------------- Face tab ----------------
    def _build_face_tab(self):
        f = self.tab_face
        pal = self._pal
        chip = pal["chip"]
        # ---- Reference picker row ----
        ref = ttk.Frame(f); ref.pack(fill="x", padx=4, pady=2)
        ttk.Label(ref, text="内置参考样本:").pack(side="left")
        self.face_ref = tk.StringVar()
        face_ref_cb = ttk.Combobox(ref, textvariable=self.face_ref,
                                   values=list(FACE_SAMPLES.keys()),
                                   state="readonly", width=26,
                                   style="Field.TCombobox")
        face_ref_cb.pack(side="left", padx=2)
        face_ref_cb.current(0)
        ttk.Button(ref, text="⇥ 填入 照片A",
                   command=self._face_load_ref).pack(side="left", padx=4)
        ttk.Label(ref, text="（face_a / face_b / face_c 已预置，可直接点“比对”）",
                  foreground="#888").pack(side="left", padx=6)

        # ---- File picker rows (A & B) ----
        def _path_row(label, var, picker):
            row = ttk.Frame(f); row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=var, width=46,
                      style="Field.TEntry").pack(
                          side="left", padx=2, fill="x", expand=True)
            ttk.Button(row, text="▤ 浏览...", command=picker).pack(side="left")
            return row

        self.face_a = tk.StringVar(value="")
        self.face_b = tk.StringVar(value="")
        _path_row("照片A:", self.face_a, lambda: self._face_pick("a"))
        _path_row("照片B:", self.face_b, lambda: self._face_pick("b"))

        # ---- Action + result strip ----
        action_row = ttk.Frame(f); action_row.pack(fill="x", padx=4, pady=2)
        self.face_run_btn = self._cta(action_row, "⚡ 比对", self._face_run)
        self.face_run_btn.pack(side="left", padx=(0, 8))
        # Score chip (tk.Label: 3-state color).
        self.face_score_chip = tk.Label(
            action_row, text="  SCORE --  ",
            bg=pal["bg"], fg=chip["dim"]["fg"],
            font=pal["font_ui_bold"], padx=8, pady=2)
        self.face_score_chip.pack(side="left", padx=4)
        self.face_method_var = tk.StringVar(value="")
        ttk.Label(action_row, textvariable=self.face_method_var,
                  foreground="#888",
                  font=("Menlo", 9)).pack(side="left", padx=8)

        # ---- Body: two preview tiles + detail text ----
        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=4, pady=2)
        body.grid_columnconfigure(0, weight=1, uniform="face")
        body.grid_columnconfigure(1, weight=1, uniform="face")
        body.grid_rowconfigure(1, weight=1)
        ttk.Label(body, text="  照片 A  ",
                  foreground=pal.get("accent", pal["primary"]),
                  font=("TkDefaultFont", 10, "bold")).grid(
                      row=0, column=0, sticky="w", padx=4, pady=(0, 4))
        ttk.Label(body, text="  照片 B  ",
                  foreground=pal.get("accent", pal["primary"]),
                  font=("TkDefaultFont", 10, "bold")).grid(
                      row=0, column=1, sticky="w", padx=4, pady=(0, 4))
        tile_a = tk.LabelFrame(body, text=" ", bg=pal["bg"],
                                fg=pal["fg"], bd=0, relief="flat",
                                highlightthickness=1,
                                highlightbackground=pal["border"],
                                highlightcolor=pal["accent"])
        tile_a.grid(row=1, column=0, sticky="nsew", padx=(4, 6), pady=4)
        tile_b = tk.LabelFrame(body, text=" ", bg=pal["bg"],
                                fg=pal["fg"], bd=0, relief="flat",
                                highlightthickness=1,
                                highlightbackground=pal["border"],
                                highlightcolor=pal["accent"])
        tile_b.grid(row=1, column=1, sticky="nsew", padx=(6, 4), pady=4)
        self.face_canvas_a = tk.Canvas(
            tile_a, width=256, height=256,
            bg=pal["code"], highlightthickness=0)
        self.face_canvas_a.pack(padx=10, pady=10)
        self.face_canvas_b = tk.Canvas(
            tile_b, width=256, height=256,
            bg=pal["code"], highlightthickness=0)
        self.face_canvas_b.pack(padx=10, pady=10)

        # ---- Detail text under tiles ----
        detail = ttk.LabelFrame(f, text=" 对比详情 (raw 输出 / 方法 / 耗时) ")
        detail.pack(fill="both", expand=False, padx=4, pady=(2, 4))
        self.face_out = tk.Text(detail, height=8, wrap="none",
                                font=("Menlo", 9),
                                bg=pal["code"], fg=pal["fg"],
                                insertbackground=pal.get("accent", pal["primary"]))
        self.face_out.pack(fill="both", expand=True, padx=4, pady=4)

        # Image refs kept alive.
        self._face_imgs = {}
        # Default face_a vs face_c -> 95/100 -> SUSPECT chip so the chip
        # visibly changes the moment the user presses 比对.
        self.face_a.set(str(FACE_SAMPLES["face_a (合成, 128×128)"]))
        self.face_b.set(str(FACE_SAMPLES["face_c (合成, 128×128)"]))
        self._face_render_preview("a", self.face_a.get())
        self._face_render_preview("b", self.face_b.get())
        self.face_out.insert(
            "1.0",
            "提示：内置 face_a / face_b / face_c 已预置，点击“比对”即可直接测试。\n"
            "也可从内置参考样本下拉选择并填入照片A，或浏览本地图片。\n")

    def _face_load_ref(self):
        path = FACE_SAMPLES.get(self.face_ref.get())
        if path:
            self.face_a.set(str(path))
            self._face_render_preview("a", str(path))

    def _face_pick(self, which):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.ppm *.bmp *.png *.jpg"), ("All", "*.*")])
        if path:
            (self.face_a if which == "a" else self.face_b).set(path)
            self._face_render_preview(which, path)

    def _face_render_preview(self, side: str, path: str):
        """Load an image into the preview canvas for side 'a' or 'b'.

        Tk is single-threaded for image loading. We try the native
        PhotoImage first (PPM / PGM / GIF / PNG) and fall back to PIL
        for JPG / BMP and for resizing below native dimensions.
        """
        canvas = self.face_canvas_a if side == "a" else self.face_canvas_b
        try:
            canvas.delete("all")
        except tk.TclError:
            return
        path = path or ""
        if not path or not Path(path).exists():
            canvas.create_text(
                128, 128, text="(no image)",
                fill=self._pal.get("fg", "#888"), anchor="center")
            return
        img = None
        try:
            img = tk.PhotoImage(file=path)
        except tk.TclError:
            img = None
        if img is None:
            try:
                from PIL import Image, ImageTk  # type: ignore
                pil = Image.open(path).convert("RGB")
                img = ImageTk.PhotoImage(pil)
            except Exception:
                canvas.create_text(
                    128, 128, text="(unsupported format)",
                    fill=self._pal.get("fg", "#888"), anchor="center")
                return
        cw, ch = 256, 256
        iw, ih = img.width(), img.height()
        scale = min(cw / iw, ch / ih)
        if scale < 1.0:
            try:
                from PIL import Image as _PILImage  # type: ignore
                pil = _PILImage.open(path).convert("RGB").resize(
                    (max(1, int(iw * scale)), max(1, int(ih * scale))),
                    _PILImage.LANCZOS)
                from PIL import ImageTk  # type: ignore
                img = ImageTk.PhotoImage(pil)
                iw, ih = img.width(), img.height()
            except Exception:
                pass
        self._face_imgs[side] = img
        canvas.create_image((cw - iw) // 2, (ch - ih) // 2,
                            image=img, anchor="nw")
        canvas.create_rectangle(0, 0, cw - 1, ch - 1, outline=self._pal["canvas_border"])

    @staticmethod
    def _face_grade(score: int):
        """Map a 0..100 score to a chip kind + human label."""
        if score >= 90:
            return "ok", "PASS"
        if score >= 70:
            return "run", "SUSPECT"
        return "fail", "FAIL"

    def _face_run(self):
        a = self.face_a.get().strip(); b = self.face_b.get().strip()
        if not (a and b):
            messagebox.showerror("错误", "请选择两张照片")
            return
        self._face_render_preview("a", a)
        self._face_render_preview("b", b)
        t0 = time.time()
        try:
            r = subprocess.run([str(FACE_TOOL), a, b, "2"],
                               text=True, capture_output=True, timeout=20)
            ms = int((time.time() - t0) * 1000)
            self.face_out.delete("1.0", "end")
            method = "(unknown)"
            score = 0
            for ln in r.stdout.splitlines():
                if ln.startswith("method:"):
                    method = ln.split(":", 1)[1].strip()
                elif ln.startswith("score"):
                    tail = ln.split(":", 1)[1].strip()
                    head = tail.split("/", 1)[0]
                    digits = "".join(ch for ch in head if ch.isdigit())
                    try:
                        score = int(digits)
                    except Exception:
                        score = 0
            kind, label = self._face_grade(score)
            self.face_method_var.set(f"method={method}")
            chip = self._pal["chip"][kind]
            try:
                self.face_score_chip.configure(
                    bg=chip["bg"], fg=chip["fg"],
                    text=f"  {label}  {score}/100  ")
            except tk.TclError:
                pass
            header = (f"method : {method}\n"
                      f"score  : {score}/100  ({label})\n"
                      f"耗时   : {ms} ms\n"
                      f"图片A  : {a}\n"
                      f"图片B  : {b}\n"
                      f"\n--- raw face_tool output ---\n")
            tail = r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or "")
            self.face_out.insert("1.0", header + tail)
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # ---------------- menu action ----------------
    def _menu_gen_sample(self):
        try:
            r = subprocess.run([str(GEN_SAMP)], capture_output=True,
                               text=True, timeout=20)
            messagebox.showinfo("生成样本", r.stdout or (r.stderr and ("[stderr]\n" + r.stderr) or ""))
        except Exception as e:
            messagebox.showerror("错误", str(e))


# Small lazy regex import kept at the bottom so the rest of the
# module stays compatible with non-regex-only environments.
def re_compile(pattern):
    import re
    return re.compile(pattern)


def main():
    app = PassportGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
