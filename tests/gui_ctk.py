#!/usr/bin/env python3
"""passport_test_gui (CustomTkinter preview) -- exercises all 4 modules.

Run from the project root with the bundled venv so that customtkinter
is on the path:

    ./.venv-ctk/bin/python tests/gui_ctk.py

The script falls back to ``sys.executable`` if the venv is missing,
but it will print a friendly hint instead of NameError when the
customtkinter import fails.

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

# ---- Visual-pass 3.0: CustomTkinter core -------------------------
# ctk is the modern, rounded, anti-aliased dark-friendly toolkit. We keep
# the ttkbackend for widgets ctk does not provide (Notebook, Treeview,
# Progressbar, PanedWindow) so the visual upgrade is additive, not a
# rewrite.
try:
    import customtkinter as ctk
    from customtkinter import CTkFont
    HAS_CTK = True
except ImportError:
    HAS_CTK = False
    CTkFont = None  # type: ignore[assignment]

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


class PassportGUI(ctk.CTk if HAS_CTK else (Window if HAS_TTKB else tk.Tk)):
    def __init__(self, themename="darkly"):
        # Fail fast: every UI builder below references ``ctk`` directly,
        # so a missing module would explode mid-build with NameError.
        if not HAS_CTK:
            raise RuntimeError(
                "tests/gui_ctk.py requires the customtkinter package. "
                "Run it with the bundled venv: "
                "./.venv-ctk/bin/python tests/gui_ctk.py "
                "(or `pip install customtkinter` into your environment).")
        # ctk takes precedence; it auto-configures DPI scaling and uses a
        # neutral dark palette that we override per .impeccable.md.
        try:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("dark-blue")
        except Exception:
            pass
        super().__init__()
        self._theme_name = themename
        self.title("护照机测试机 - Passport Test Bench")
        self.geometry("1280x780")
        self._set_window_icon()
        self._build_menu()
        self._apply_theme_proof_styles()
        # Visual-pass 3.0: brand root colour (deep space black) so the
        # window chrome blends with the canvas.
        if HAS_CTK:
            try:
                self.configure(fg_color="#0a0e14")
            except tk.TclError:
                pass
        else:
            try:
                self.configure(bg="#0a0e14")
            except tk.TclError:
                pass
        self._nfc_key_entries = {}
        self._nfc_dg1_lbs = {}
        self._nfc_prog_lbs = {}
        self._pal = self._theme_palette()
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
        self._refresh_tk_theme()
        self.status = tk.StringVar(value=self._status_text())
        sbar = ttk.Frame(self)
        sbar.pack(fill="x", side="bottom")
        self.ocr_status_var = tk.StringVar(value="就绪")
        ttk.Label(sbar, textvariable=self.ocr_status_var, anchor="w",
                  padding=4).pack(side="left")
        self.ocr_progress = ttk.Progressbar(sbar, length=180,
                                            mode="determinate", maximum=100)
        self.ocr_progress.pack(side="left", padx=8, pady=2)
        # Visual-pass 3.0: status bar becomes a ctk chip strip. Five chips
        # report per-tool presence with neon green/red dots.
        chip_frame = ctk.CTkFrame(sbar, fg_color="transparent")
        chip_frame.pack(side="right", fill="x", padx=4, pady=2)
        self._status_chip_labels = {}
        pal = self._pal
        chip_palette = pal["chip"]
        for label, path in (("MRZ", MRZ_TOOL), ("AC", AC_TOOL),
                            ("FACE", FACE_TOOL),
                            ("GEN", GEN_SAMP), ("NFC", NFC_TOOL)):
            ok = path.exists()
            chosen = chip_palette["ok"] if ok else chip_palette["fail"]
            lb = ctk.CTkLabel(
                chip_frame, text=f" ● {label} ",
                fg_color=chosen["bg"], text_color=chosen["fg"],
                font=ctk.CTkFont(family=pal["font_ui_bold"][0],
                                 size=pal["font_ui_bold"][1],
                                 weight="bold"),
                corner_radius=4, padx=2)
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
        to a light-field/dark-text scheme that survives macOS dark mode."""
        st = ttk.Style()
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
            FIELD, TEXT, HEAD, SEL = "#ffffff", "#111827", "#e9edf3", "#0b5cad"
        # Pin palette-derived border so Labelframe/Card uses a faint line.
        border = (self._pal.get("border") if hasattr(self, "_pal") else "#3A3B3C")
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
            # 1px flat border tinted to the palette border colour.
            st.configure("TLabelframe", bordercolor=border, borderwidth=1,
                         relief="solid")
            st.configure("TLabelframe.Label", foreground=TEXT, padding=(6, 2))
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
        except Exception:
            pass

    def _switch_theme(self, name):
        st = ttk.Style()
        st.theme_use(name)
        self._theme_name = name
        self._apply_theme_proof_styles()
        self._pal = self._theme_palette()
        self._refresh_tk_theme()

    def _theme_palette(self):
        """Semantic color palette for the active theme (tk widgets)."""
        st = ttk.Style()
        try:
            c = st.colors
            if isinstance(c, dict):
                bg = c.get("bg", "#14181e")
                fg = c.get("fg", "#d5dae2")
                field = c.get("inputbg", "#1f242c")
                field_fg = c.get("inputfg", "#e6ebf2")
                primary = c.get("primary", "#0b5cad")
            else:
                bg = getattr(c, "bg", "#14181e")
                fg = getattr(c, "fg", "#d5dae2")
                field = getattr(c, "inputbg", "#1f242c")
                field_fg = getattr(c, "inputfg", "#e6ebf2")
                primary = getattr(c, "primary", "#0b5cad")
        except AttributeError:
            bg, fg, field, field_fg, primary = ("#14181e", "#d5dae2",
                                                "#1f242c", "#e6ebf2", "#0b5cad")
        dark = (not HAS_TTKB) or self._theme_name in ("darkly", "cyborg")
        accent = "#00e5ff" if dark else primary
        # ---- Visual-pass 2.0: semantic token set (chip / border / fonts) ----
        # Surface elevation: bg < card < field, kept inside a tight grayscale
        # so the contrast ratio survives light/dark theme switches.
        # Brand-aligned per .impeccable.md: neon cyan accent + neon red/green.
        # idle = "armed but waiting" (cyan); dim = true empty placeholder.
        if dark:
            border = "#3A3B3C"
            card = "#0d1117"   # canvas-deep card per brand spec
            code = "#0a0e14"   # code surface, near-black per brand spec
            chip_idle_fg, chip_idle_bg = "#00e5ff", "#0f2a33"   # neon cyan armed
            chip_ok_fg,   chip_ok_bg   = "#2ee6a8", "#0f3325"   # neon green
            chip_run_fg,  chip_run_bg  = "#b388ff", "#251a3d"   # violet in-flight
            chip_fail_fg, chip_fail_bg = "#ff5c7a", "#3a1620"   # neon red
            chip_dim_fg,  chip_dim_bg  = "#6a737d", "#1a1d22"   # muted grey (true empty)
        else:
            border = "#D0D5DD"
            card = "#F2F4F7"
            code = "#FFFFFF"
            chip_idle_fg, chip_idle_bg = "#0b5cad", "#E6F0FA"   # brand blue armed
            chip_ok_fg,   chip_ok_bg   = "#027a48", "#D1FADF"
            chip_run_fg,  chip_run_bg  = "#6c2bd9", "#EDE4FF"
            chip_fail_fg, chip_fail_bg = "#B42318", "#FEE4E2"
            chip_dim_fg,  chip_dim_bg  = "#666666", "#EAECF0"
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
        return {
            "bg": bg, "fg": fg, "field": field, "field_fg": field_fg,
            "primary": primary, "accent": accent,
            "ok": "#2ee6a8", "err": "#ff5c7a",
            "card": card, "code": code, "border": border,
            "chip": {
                "idle":  {"fg": chip_idle_fg,  "bg": chip_idle_bg},
                "ok":    {"fg": chip_ok_fg,    "bg": chip_ok_bg},
                "run":   {"fg": chip_run_fg,   "bg": chip_run_bg},
                "fail":  {"fg": chip_fail_fg,  "bg": chip_fail_bg},
                "dim":   {"fg": chip_dim_fg,   "bg": chip_dim_bg},
            },
            "font_ui": (ui_family, 10),
            "font_ui_bold": (ui_family, 10, "bold"),
            "font_mono": (mono_family, 10),
            "font_mono_sm": (mono_family, 9),
            # CTK-aware fields; only used when HAS_CTK.
            "ctk": {
                "fg_color":   card,         # card surface
                "bg_color":   "#0a0e14",    # root window background
                "top_fg":     "#0d1117",    # elevated card
                "entry_bg":   code,         # code surface
                "border":     border,
                "primary":    "#2e86ff",    # brand electric blue (upgraded per spec)
                "primary_h":  "#3d97ff",
                "accent":     "#00e5ff",
                "success":    "#2ee6a8",
                "danger":     "#ff5c7a",
                "text":       "#e6edf3",
                "text_dim":   "#8b949e",
                "hover":      "#1f2933",
            } if HAS_CTK else None,
        }

    def _refresh_tk_theme(self):
        """Re-color tk (non-ttk) widgets to match the active theme."""
        pal = self._pal
        chip = pal["chip"]
        # Text/code widgets: bg = field, fg = field_fg (already contrast-safe).
        for n in ("ocr_diff_text", "loc_info", "nfc_inspect", "nfc_adv_text",
                  "mrz_text", "mrz_out", "ac_out", "face_out"):
            w = getattr(self, n, None)
            if w is not None:
                try:
                    w.configure(bg=pal["field"], fg=pal["field_fg"])
                except tk.TclError:
                    pass
        w = getattr(self, "loc_canvas", None)
        if w is not None:
            try:
                w.configure(bg=pal["bg"])
            except tk.TclError:
                pass
        # ---- Visual-pass 2.0: chip-style badges need explicit bg/fg ----
        # Right-top corner empty white square fix: nfc_badge starts blank but
        # was inheriting the platform default bg; pin to window bg + dim chip.
        w = getattr(self, "nfc_badge", None)
        if w is not None:
            chosen = chip["dim"]
            try:
                w.configure(fg_color=chosen["bg"], text_color=chosen["fg"])
            except (tk.TclError, ValueError):
                try:
                    w.configure(bg=chosen["bg"], fg=chosen["fg"])
                except tk.TclError:
                    pass
        # BAC state-machine chips: reset every known badge to the idle
        # chip palette; _nfc_show() will recolour them on the next render.
        for _title, _var, lb in getattr(self, "_nfc_states", []):
            chosen = chip["idle"]
            try:
                lb.configure(fg_color=chosen["bg"], text_color=chosen["fg"])
            except (tk.TclError, ValueError):
                try:
                    lb.configure(bg=chosen["bg"], fg=chosen["fg"])
                except tk.TclError:
                    pass
        # DG1 / SOD / read-progress labels: under CTK, dicts hold None
        # placeholders because the chips are stored inline. The plain tk
        # build still keeps tk.Label refs; the no-op check below stays.
        for d in ("_nfc_dg1_lbs", "_nfc_prog_lbs"):
            dd = getattr(self, d, None) or {}
            for w in dd.values():
                if w is None:
                    continue
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
        # contrast-safe chip backgrounds. Dual-mode: ctk uses fg_color /
        # text_color; tk fallback uses foreground/background.
        for _label, (lb, ok) in getattr(self, "_status_chip_labels", {}).items():
            chosen = chip["ok"] if ok else chip["fail"]
            try:
                lb.configure(fg_color=chosen["bg"], text_color=chosen["fg"])
            except tk.TclError:
                try:
                    lb.configure(foreground=chosen["fg"],
                                 background=chosen["bg"])
                except tk.TclError:
                    pass
        # Run-button accent. Under CTK, nfc_run_btn is a CTkButton (no bg);
        # under legacy tk, all three are tk.Button and accept bg/fg.
        for n in ("ocr_run_btn", "loc_run_btn", "nfc_run_btn"):
            w = getattr(self, n, None)
            if w is None:
                continue
            try:
                if HAS_CTK:
                    w.configure(fg_color=pal["primary"],
                                hover_color=pal["accent"],
                                text_color="#ffffff")
                else:
                    w.configure(bg=pal["primary"], fg="white",
                                activebackground=pal["primary"],
                                activeforeground="white")
            except (tk.TclError, ValueError):
                pass
        w = getattr(self, "ocr_preview_label", None)
        if w is not None:
            try:
                w.configure(background=pal["bg"])
            except tk.TclError:
                pass
        for d in ("_nfc_key_entries", "_nfc_dg1_lbs", "_nfc_prog_lbs"):
            dd = getattr(self, d, None) or {}
            for w in dd.values():
                if w is None:
                    continue
                try:
                    if HAS_CTK:
                        w.configure(fg_color=pal["code"],
                                    text_color=pal["field_fg"])
                    else:
                        w.configure(bg=pal["field"], fg=pal["field_fg"],
                                    readonlybackground=pal["field"])
                except (tk.TclError, ValueError):
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

    # ---- CTK card primitive ----------------------------------------
    # Replaces ttk.LabelFrame with a flat card: CTkFrame (rounded) holding
    # a CTkLabel header. Used by every tab builder that opts into ctk.
    def _ctk_card(self, parent, title: str, padding: int = 8):
        """Return (frame, body_frame) where frame is the rounded card and
        body_frame is the inner content area below the title row."""
        c = self._pal["ctk"]
        outer = ctk.CTkFrame(
            parent, fg_color=c["fg_color"],
            corner_radius=10, border_width=1, border_color=c["border"])
        outer.pack(fill="x", padx=6, pady=4, anchor="n")
        # Title: small caps style with letter-spacing via extra spaces.
        ctk.CTkLabel(
            outer, text=f"  {title.upper()}  ",
            font=ctk.CTkFont(family=self._pal["font_ui"][0], size=10,
                             weight="bold"),
            text_color=c["text_dim"], fg_color="transparent",
            anchor="w").pack(fill="x", padx=padding, pady=(padding, 2))
        body = ctk.CTkFrame(outer, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=padding, pady=(0, padding))
        return outer, body

    def _ctk_entry(self, parent, textvariable, **kw):
        c = self._pal["ctk"]
        return ctk.CTkEntry(
            parent, textvariable=textvariable,
            fg_color=c["entry_bg"], border_color=c["border"],
            text_color=c["accent"],
            font=ctk.CTkFont(family=self._pal["font_mono_sm"][0],
                             size=self._pal["font_mono_sm"][1]),
            height=28, corner_radius=6, **kw)

    def _ctk_btn_primary(self, parent, text, command=None, **kw):
        c = self._pal["ctk"]
        return ctk.CTkButton(
            parent, text=text, command=command,
            fg_color=c["primary"], hover_color=c["primary_h"],
            text_color="#ffffff",
            font=ctk.CTkFont(family=self._pal["font_ui_bold"][0],
                             size=self._pal["font_ui_bold"][1],
                             weight=self._pal["font_ui_bold"][2]
                             if len(self._pal["font_ui_bold"]) > 2 else "bold"),
            corner_radius=6, height=30, **kw)

    def _ctk_btn_ghost(self, parent, text, command=None, **kw):
        c = self._pal["ctk"]
        return ctk.CTkButton(
            parent, text=text, command=command,
            fg_color=c["hover"], hover_color=c["top_fg"],
            text_color=c["text"],
            font=ctk.CTkFont(family=self._pal["font_ui"][0],
                             size=self._pal["font_ui"][1]),
            corner_radius=6, height=28, **kw)

    def _ctk_chip(self, parent, kind: str, textvariable=None, text=None):
        """Render a brand-aligned chip. kind in chip/idle/ok/run/fail/dim."""
        chip = self._pal["chip"][kind]
        kwargs = dict(
            text_color=chip["fg"], fg_color=chip["bg"],
            font=ctk.CTkFont(family=self._pal["font_ui_bold"][0],
                             size=self._pal["font_ui_bold"][1],
                             weight="bold"),
            corner_radius=4, height=22, padx=2)
        if textvariable is not None:
            kwargs["textvariable"] = textvariable
        else:
            kwargs["text"] = text
        return ctk.CTkLabel(parent, **kwargs)

    def _build_menu(self):
        menubar = tk.Menu(self)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="生成样本图片", command=self._menu_gen_sample)
        filem.add_separator()
        filem.add_command(label="退出", command=self.destroy)
        menubar.add_cascade(label="文件", menu=filem)
        viewm = tk.Menu(menubar, tearoff=0)
        for name in ("darkly", "cyborg", "cosmo", "journal"):
            viewm.add_command(label=name,
                              command=lambda n=name: self._switch_theme(n))
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
        self.ocr_methods = tk.StringVar(value="traditional,cnn")
        for label, key in [("传统模板", "traditional"), ("CNN", "cnn")]:
            ttk.Radiobutton(row1, text=label, variable=self.ocr_methods,
                            value=key).pack(side="left", padx=2)
        self.ocr_method_both = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="同时跑两个方案",
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
        # Accent button: dark-blue filled, white text — explicit, theme-proof.
        # tk.Button (not ttk) so bg/fg render on macOS aqua.
        self.ocr_run_btn = tk.Button(
            row1, text="▶ 开始测试", command=self._ocr_run,
            bg="#0b5cad", fg="white", activebackground="#2b7ed6",
            activeforeground="white", disabledforeground="#9cc3ee",
            highlightbackground="#0b5cad", highlightthickness=1,
            relief="flat", cursor="hand2", bd=0,
            font=("TkDefaultFont", 10, "bold"), padx=14, pady=4)
        self.ocr_run_btn.pack(side="right", padx=4, pady=2)

        row2 = ttk.Frame(ctrl); row2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(row2, text="全选",
                   command=lambda: self._ocr_select_all(True)).pack(side="left", padx=2)
        ttk.Button(row2, text="反选",
                   command=lambda: self._ocr_select_all(False)).pack(side="left", padx=2)
        ttk.Button(row2, text="清空选择",
                   command=self._ocr_clear_sel).pack(side="left", padx=2)
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
            self.ocr_detail.column(c, width=90, anchor=anc)
        self.ocr_detail.tag_configure("OK", background="#c8e6c9",
                                      foreground="#1b5e20")
        self.ocr_detail.tag_configure("FAIL", background="#ffcdd2",
                                      foreground="#b71c1c")
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
        ttk.Label(preview, textvariable=self.ocr_preview_meta,
                  wraplength=380, justify="left",
                  font=("TkDefaultFont", 9)).pack(anchor="w")
        self.ocr_preview_label = ttk.Label(preview, background="#222",
                                           anchor="center")
        self.ocr_preview_label.pack(fill="both", expand=True)
        self.ocr_preview_label.bind("<Configure>", self._update_preview_image)
        ttk.Label(preview, text="识别结果 · 字符级比对 (GT vs Pred)").pack(anchor="w")
        self.ocr_diff_text = tk.Text(preview, height=11, width=48,
                                     font=("Menlo", 10), wrap="none",
                                     bg="#ffffff", fg="#111827")
        self.ocr_diff_text.tag_configure("hdr", font=("TkDefaultFont", 9, "bold"),
                                         foreground="#555")
        self.ocr_diff_text.tag_configure("lab", foreground="#888")
        self.ocr_diff_text.tag_configure("match", foreground="#2e7d32",
                                         font=("Menlo", 10, "bold"))
        self.ocr_diff_text.tag_configure("mm", background="#ffebee",
                                         foreground="#c62828",
                                         font=("Menlo", 10, "bold"))
        self.ocr_diff_text.tag_configure("okline", foreground="#2e7d32",
                                         font=("Menlo", 10))
        d_y = ttk.Scrollbar(preview, orient="vertical",
                            command=self.ocr_diff_text.yview)
        d_x = ttk.Scrollbar(preview, orient="horizontal",
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
            self.ocr_run_btn.configure(text="▶ 开始识别")
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
            self.ocr_run_btn.configure(text="▶ 开始测试")
            self._ocr_load_corpus()

    def _build_loc_tab(self):
        """Passport image OCR region locator: photo / data / MRZ (standalone tab).
        Built from tests/passport_locator.py (pure PIL+numpy).
        """
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
        ttk.Button(row, text="浏览文件…", command=self._loc_pick).pack(side="left", padx=4)
        self.loc_run_btn = tk.Button(row, text="▶ 定位区域", command=self._loc_run,
                  bg="#0b5cad", fg="white", activebackground="#2b7ed6",
                  activeforeground="white", disabledforeground="#9cc3ee",
                  highlightbackground="#0b5cad", highlightthickness=1,
                  relief="flat", cursor="hand2", bd=0,
                  font=("TkDefaultFont", 10, "bold"), padx=12, pady=3)
        self.loc_run_btn.pack(side="left", padx=4)
        self.loc_status = ttk.Label(row, text="", foreground="#666")
        self.loc_status.pack(side="left", padx=8)

        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=4, pady=(4, 6))
        self.loc_canvas = tk.Canvas(body, width=760, height=440, bg="#222",
                                    highlightthickness=1, highlightbackground="#555")
        self.loc_canvas.pack(side="left", fill="both", expand=True)
        self.loc_info = tk.Text(body, width=52, height=20, font=("Menlo", 9),
                                state="disabled", wrap="none",
                                bg="#fbfbfb", fg="#111827", relief="solid", bd=1)
        self.loc_info.pack(side="left", padx=(6, 0), fill="y")

    def _loc_pick(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.png *.ppm *.bmp *.jpg"), ("All", "*.*")])
        if path:
            self.loc_sample_path.set(path)
            self.loc_sample.set(f"自定义: {os.path.basename(path)}")
            try:
                self._loc_display_image(path, "已选择文件，点“定位区域”开始定位")
            except Exception as e:
                self.loc_status.config(text=f"加载失败: {e}", foreground="#b71c1c")

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
                                    anchor="sw", fill="#9aa", font=("Menlo", 8))
        if note:
            self.loc_status.config(text=note, foreground="#666")

    def _loc_on_sample_selected(self, _evt=None):
        try:
            path = self._loc_resolve_path()
            self._loc_display_image(path, "已显示样本图片，点“定位区域”开始定位")
        except Exception as e:
            self.loc_status.config(text=f"加载失败: {e}", foreground="#b71c1c")

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
                                        anchor="sw", fill="#9aa", font=("Menlo", 8))
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
            self.loc_status.config(text=f"完成：{ok}/3 区域", foreground="#1b5e20")
            self._loc_set_text("\n".join(txt))
        except Exception as e:
            self.loc_status.config(text=f"定位失败: {e}", foreground="#b71c1c")
            self._loc_set_text(f"错误: {e}")

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
        for key, label in [("cases", "案例数"), ("mean", "Mean耗时"),
                           ("p95", "P95耗时"), ("pass", "Pass率"),
                           ("l1", "平均L1"), ("l2", "平均L2"),
                           ("full", "全匹配")]:
            cell = ttk.Frame(card); cell.pack(side="left", padx=12, pady=4)
            ttk.Label(cell, text=label, foreground="#666",
                      font=("TkDefaultFont", 9)).pack()
            var = tk.StringVar(value="--")
            ttk.Label(cell, textvariable=var,
                      font=("TkDefaultFont", 11, "bold")).pack()
            self.ocr_metrics[key] = var

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

    def _set_status_bar(self, msg, pct=None):
        self.ocr_status_var.set(msg)
        if pct is None:
            self.ocr_progress.configure(mode="indeterminate")
            self.ocr_progress.start(12)
        else:
            self.ocr_progress.stop()
            self.ocr_progress.configure(mode="determinate", value=float(pct))

    def _ocr_method_toggle(self):
        if self.ocr_method_both.get():
            self.ocr_methods.set("traditional,cnn")
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
            f"skew={rec['skew']}\n{rec['image']}")
        try:
            from PIL import Image
            self._preview_img_pil = Image.open(img_path).convert("RGB")
            self._update_preview_image()
        except Exception:
            self._preview_img_pil = None
            self.ocr_preview_label.configure(image="",
                text=f"(无法预览 {img_path.name})")

    def _update_preview_image(self, _evt=None):
        im = getattr(self, "_preview_img_pil", None)
        if im is None:
            return
        w = self.ocr_preview_label.winfo_width()
        h = self.ocr_preview_label.winfo_height()
        if w <= 1 or h <= 1:
            return
        im2 = im.copy()
        im2.thumbnail((max(1, w - 6), max(1, h - 6)))
        try:
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(im2)
        except Exception:
            return
        self._preview_imgs["_current"] = photo
        self.ocr_preview_label.configure(image=photo, text="")

    def _render_diff_block(self, tw, label, gt, pred):
        """Append one aligned GT/Pred block with per-char highlighting."""
        n = max(len(gt), len(pred))
        gt = gt.ljust(n)
        pred = pred.ljust(n)
        tw.insert("end", label + "\n", "hdr")
        tw.insert("end", "GT  : ", "lab")
        for i in range(n):
            tw.insert("end", gt[i],
                      "match" if gt[i] == pred[i] else "mm")
        tw.insert("end", "\nPRED: ", "lab")
        mism = []
        for i in range(n):
            if gt[i] != pred[i]:
                mism.append(i)
            tw.insert("end", pred[i],
                      "match" if gt[i] == pred[i] else "mm")
        tw.insert("end", "\n     ")
        if mism:
            for i in range(n):
                tw.insert("end", "^" if i in mism else " ", "mm")
            tw.insert("end", "\n")
        else:
            tw.insert("end", "（全部匹配）\n", "okline")

    def _show_diff(self, blocks):
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        for label, gt, pred in blocks:
            self._render_diff_block(tw, label, gt, pred)
        tw.configure(state="disabled")

    def _show_gt_only(self, rec):
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        tw.insert("end", "line1 (GT)\n", "hdr")
        tw.insert("end", rec["line1"] + "\n", "match")
        tw.insert("end", "line2 (GT)\n", "hdr")
        tw.insert("end", rec["line2"] + "\n", "match")
        tw.insert("end", "（运行测试后在此显示 Pred 与字符级比对）", "lab")
        tw.configure(state="disabled")

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
                label = {"traditional": "传统模板", "cnn": "CNN"}[m]
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
        """Resolve the corpus record for the currently selected detail row
        and pop the full-size viewer. Detail rows are keyed by (case_id,
        method); the case_id is the first 8 chars of the row iid."""
        sel = self.ocr_detail.selection()
        if not sel:
            return
        iid = sel[0]
        # detail row iids are "<case_id>::<method>" per _ocr_run/_ocr_run_photos
        cid = iid.split("::", 1)[0] if "::" in iid else iid[:8]
        rec = next((r for r in self._ocr_corpus["records"]
                    if r["id"] == cid), None)
        if rec is None:
            return
        self._ocr_open_original(cid, rec)

    def _ocr_open_original(self, cid=None, rec=None):
        """Open a top-level window showing the original (unresized) image
        with zoom controls (1x / 2x / 4x / fit) and a chip bar reporting
        case metadata."""
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
        # Defer heavy imports to call time; PIL is already required by
        # _load_preview so this is cheap.
        try:
            from PIL import Image as _PILImage
        except Exception as e:
            messagebox.showerror("依赖缺失",
                                 "原图查看器需要 Pillow: pip install pillow")
            return
        try:
            full_pil = _PILImage.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法读取 {img_path.name}: {e}")
            return
        # Build the top-level window.
        if HAS_CTK:
            win = ctk.CTkToplevel(self)
        else:
            win = tk.Toplevel(self)
        win.title(f"原图 · {cid}  ·  {img_path.name}")
        win.configure(fg_color=self._pal["ctk"]["bg_color"])
        # Geometry: 80% of main window, clamped to image size.
        iw, ih = full_pil.size
        try:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        except Exception:
            sw, sh = 1440, 900
        ww = min(max(iw + 80, 720), int(sw * 0.9))
        wh = min(max(ih + 140, 540), int(sh * 0.9))
        win.geometry(f"{ww}x{wh}")
        # ---- Top metadata bar (chips) ----
        pal = self._pal
        top = ctk.CTkFrame(win, fg_color=pal["ctk"]["fg_color"],
                           corner_radius=10, border_width=1,
                           border_color=pal["ctk"]["border"])
        top.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(
            top, text=f"  原图 · {cid}  ",
            text_color=pal["ctk"]["accent"],
            font=ctk.CTkFont(family=pal["font_ui_bold"][0],
                             size=pal["font_ui_bold"][1],
                             weight="bold")).pack(side="left", padx=(10, 6), pady=6)
        for k, label, kind in (
                ("scale",   "scale",   "idle"),
                ("noise",   "noise",   "idle"),
                ("skew",    "skew",    "idle"),
                ("channel", "channel", "idle"),
        ):
            v = rec.get(k, "-")
            ctk.CTkLabel(
                top, text=f"  {label}={v}  ",
                text_color=pal["chip"][kind]["fg"],
                fg_color=pal["chip"][kind]["bg"],
                font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                                 size=pal["font_mono_sm"][1],
                                 weight="bold"),
                corner_radius=4, padx=2,
            ).pack(side="left", padx=4)
        ctk.CTkLabel(
            top, text=f"  {iw}x{ih}  ",
            text_color=pal["ctk"]["text_dim"],
            font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                             size=pal["font_mono_sm"][1]),
        ).pack(side="right", padx=8)
        # ---- Zoom controls ----
        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.pack(fill="x", padx=8, pady=4)
        zoom_state = {"factor": 1.0}
        status_var = tk.StringVar(value="zoom 1.0x")
        ctk.CTkLabel(
            bar, textvariable=status_var,
            text_color=pal["ctk"]["text_dim"],
            font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                             size=pal["font_mono_sm"][1]),
        ).pack(side="right", padx=8)
        # ---- Canvas + scrollbars ----
        canvas_frame = ctk.CTkFrame(win, fg_color=pal["ctk"]["top_fg"],
                                    corner_radius=10, border_width=1,
                                    border_color=pal["ctk"]["border"])
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        canvas = tk.Canvas(canvas_frame, bg=pal["code"],
                           highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        xscroll = ttk.Scrollbar(canvas_frame, orient="horizontal",
                                command=canvas.xview)
        xscroll.pack(side="bottom", fill="x")
        yscroll = ttk.Scrollbar(canvas_frame, orient="vertical",
                                command=canvas.yview)
        yscroll.pack(side="right", fill="y")
        canvas.configure(xscrollcommand=xscroll.set,
                         yscrollcommand=yscroll.set)
        photo_ref = {"img": None, "iid": None}

        def render(factor):
            """Render the full image at the given scale factor (1.0 == original pixels)."""
            factor = max(0.1, min(factor, 8.0))
            zoom_state["factor"] = factor
            status_var.set(f"zoom {factor:.2f}x")
            zw = max(1, int(iw * factor))
            zh = max(1, int(ih * factor))
            if factor == 1.0:
                # 1x is a lossless pass-through to keep memory low.
                pil_small = full_pil
            else:
                # Use LANCZOS for downscale, BICUBIC for upscale.
                from PIL import Image as _PI
                method = _PI.LANCZOS if factor < 1.0 else _PI.BICUBIC
                pil_small = full_pil.resize((zw, zh), method)
            from PIL import ImageTk as _IT
            photo = _IT.PhotoImage(pil_small)
            photo_ref["img"] = photo
            canvas.delete("all")
            iid = canvas.create_image(0, 0, image=photo, anchor="nw")
            photo_ref["iid"] = iid
            canvas.configure(scrollregion=(0, 0, zw, zh))
            canvas._ocr_zoom = factor  # used by wheel handler

        def fit_to_window():
            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 1)
            ch = max(canvas.winfo_height(), 1)
            sx = cw / iw
            sy = ch / ih
            return render(min(sx, sy))

        def set_zoom(f):
            render(f)

        for label, value in (("Fit", None), ("1x", 1.0),
                             ("2x", 2.0), ("4x", 4.0)):
            cmd = (lambda v=value: v is None and fit_to_window() or set_zoom(v))                 if HAS_CTK else (lambda v=value: fit_to_window() if v is None
                                 else set_zoom(v))
            self._ctk_btn_ghost(
                bar, label, command=cmd, width=56).pack(side="left", padx=2)
        # Mouse wheel zoom (Ctrl+wheel) and pan (wheel).
        def on_wheel(evt):
            # macOS delta is event.delta (small multiples); win/linux num=4/5.
            if evt.state & 0x4 or evt.state & 0x0008:  # Ctrl / Cmd
                delta = evt.delta if hasattr(evt, "delta") else 0
                step = 1.1 if delta > 0 else 1 / 1.1
                render(zoom_state["factor"] * step)
                return "break"
            # plain wheel: vertical pan
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
        # Initial render: fit window.
        canvas.update_idletasks()
        fit_to_window()

    def _ocr_on_detail_select(self, _evt=None):
        sel = self.ocr_detail.selection()
        if not sel:
            return
        kind, name = self._ocr_detail_map[sel[0]]
        if kind == "photo":
            from passport_pipeline import PIC_DIR
            path = PIC_DIR / name
            self.ocr_preview_meta.set(f"{name}\n{path}")
            self._preview_img_pil = None
            self._update_preview_image()
            self.ocr_diff_text.configure(state="normal")
            self.ocr_diff_text.delete("1.0", "end")
            self.ocr_diff_text.insert("end", f"{name}\n", ("hdr",))
            rec = next((r for r in getattr(self, "_ocr_last", {}).get("photo", [])
                        if r.get("file") == name), None)
            if rec:
                for k, v in (("判定", rec.get("grade")),
                             ("定位", f"MRZ x={rec['loc']['mrz']['x']} "
                                      f"y={rec['loc']['mrz']['y']} "
                                      f"w={rec['loc']['mrz']['w']} "
                                      f"h={rec['loc']['mrz']['h']}"
                                      if rec.get("loc") and rec["loc"].get("mrz")
                                      else "未定位"),
                             ("OCR 校验", "通过" if rec.get("verified") else "未通过"),
                             ("耗时", f"{rec.get('ms', 0):.0f} ms"),
                             ("说明", rec.get("reason") or "")):
                    self.ocr_diff_text.insert("end", f"  {k}: {v}\n", ("lab",))
            self.ocr_diff_text.configure(state="disabled")
            return
        cid, m = kind, name
        rec = next((r for r in self._ocr_corpus["records"]
                    if r["id"] == cid), None)
        if rec is None:
            return
        self._load_preview(cid, rec)
        r = getattr(self, "_ocr_last", {}).get(cid, {}).get(m, {})
        label = {"traditional": "传统模板", "cnn": "CNN"}.get(m, m)
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
            methods = ["traditional", "cnn"]
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
            argv = ["python3", str(runner), "--progress",
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
                            f" 判定: {rec['grade']}", 100.0 * i / n)))
                self.after(0, lambda: self._ocr_populate_photos(results))
            except Exception as e:  # noqa: BLE001
                import traceback as _tb
                _tb.print_exc()
                self.after(0, lambda e=e: self._ocr_run_failed(e))

        threading.Thread(target=worker, daemon=True).start()

    def _ocr_populate_photos(self, results):
        """Render photo-recognition results into detail/summary/metrics."""
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
        for tag, bg, fg in (("PASS", "#c8e6c9", "#1b5e20"),
                            ("SUSPECT", "#fff3cd", "#8a6d00"),
                            ("REJECT", "#ffcdd2", "#b71c1c")):
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
        m["mean"].set(f"{ms_sum / max(1, len(results)):.1f} ms")
        m["pass"].set(f"{100.0 * stats['PASS'] / max(1, len(results)):.1f}%")
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
        self.ocr_metrics["mean"].set(f"{mean:.1f} ms")
        self.ocr_metrics["p95"].set(f"{p95:.1f} ms")
        self.ocr_metrics["pass"].set(
            f"{100.0 * okc / total_n:.1f}%" if total_n else "0%")
        self.ocr_metrics["l1"].set(
            f"{100.0 * l1c / l1t:.2f}%" if l1t else "--")
        self.ocr_metrics["l2"].set(
            f"{100.0 * l2c / l2t:.2f}%" if l2t else "--")
        self.ocr_metrics["full"].set(str(full))
        # Per-method summary.
        for m in methods:
            a = agg[m]
            ms_avg = a["ms_total"] / a["n"] if a["n"] else 0
            ok_pct = 100.0 * a["ok"] / a["n"] if a["n"] else 0
            l1 = 100.0 * a["l1_correct"] / a["l1_total"] if a["l1_total"] else 0
            l2 = 100.0 * a["l2_correct"] / a["l2_total"] if a["l2_total"] else 0
            label = {"traditional": "传统模板", "cnn": "CNN"}.get(m, m)
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
                            {"traditional": "传统", "cnn": "CNN"}.get(m, m),
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
        # ---- Top: ctk-rounded control bar -----------------------------
        top = ctk.CTkFrame(f, fg_color=self._pal["ctk"]["fg_color"],
                           corner_radius=10, border_width=1,
                           border_color=self._pal["ctk"]["border"])
        top.pack(fill="x", padx=6, pady=(8, 4))
        ctk.CTkLabel(top, text="后端:",
                     text_color=self._pal["ctk"]["text_dim"],
                     font=ctk.CTkFont(family=self._pal["font_ui"][0],
                                      size=self._pal["font_ui"][1])
                     ).pack(side="left", padx=(10, 4), pady=6)
        self.nfc_backend_var = tk.StringVar(value="Mock 卡片")
        ctk.CTkOptionMenu(
            top, variable=self.nfc_backend_var,
            values=("Mock 卡片", "真实读卡器"),
            fg_color=self._pal["ctk"]["entry_bg"],
            button_color=self._pal["ctk"]["primary"],
            button_hover_color=self._pal["ctk"]["primary_h"],
            text_color=self._pal["ctk"]["text"],
            dropdown_fg_color=self._pal["ctk"]["top_fg"],
            dropdown_text_color=self._pal["ctk"]["text"],
            dropdown_hover_color=self._pal["ctk"]["hover"],
            width=140, height=28, corner_radius=6).pack(side="left")
        ctk.CTkLabel(top, text="脚本:",
                     text_color=self._pal["ctk"]["text_dim"],
                     font=ctk.CTkFont(family=self._pal["font_ui"][0],
                                      size=self._pal["font_ui"][1])
                     ).pack(side="left", padx=(14, 4))
        self.nfc_script_var = tk.StringVar(
            value=str(ROOT / "4_nfc_reader" / "data" / "script_happy.json"))
        self._ctk_entry(top, self.nfc_script_var).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        self._ctk_btn_ghost(top, "浏览…",
                            command=self._nfc_pick_script).pack(side="left")
        self.nfc_run_btn = self._ctk_btn_primary(
            top, "▶ 开始执行", command=self._nfc_run, width=140)
        self.nfc_run_btn.pack(side="left", padx=(8, 8))

        # ---- MRZ compact row (rounded) ---------------------------------
        mrz = ctk.CTkFrame(f, fg_color=self._pal["ctk"]["fg_color"],
                           corner_radius=10, border_width=1,
                           border_color=self._pal["ctk"]["border"])
        mrz.pack(fill="x", padx=6, pady=4)
        pal = self._pal
        ctk.CTkLabel(mrz, text="MRZ L1:",
                     text_color=pal["ctk"]["text_dim"]).pack(side="left",
                                                              padx=(10, 4), pady=6)
        self.nfc_mrz1_var = tk.StringVar(value=SAMPLE_MRZ.splitlines()[0])
        self._ctk_entry(mrz, self.nfc_mrz1_var, width=520).pack(
            side="left", padx=(0, 10))
        ctk.CTkLabel(mrz, text="L2:",
                     text_color=pal["ctk"]["text_dim"]).pack(side="left",
                                                              padx=(0, 4))
        self.nfc_mrz2_var = tk.StringVar(value=SAMPLE_MRZ.splitlines()[1])
        self._ctk_entry(mrz, self.nfc_mrz2_var, width=520).pack(
            side="left", padx=(0, 10))
        self._ctk_btn_ghost(mrz, "从 OCR 填充",
                            command=self._nfc_fill_mrz_from_ocr).pack(
                                side="left", padx=4)
        self._ctk_btn_ghost(mrz, "内置样例",
                            command=self._nfc_fill_mrz_sample).pack(
                                side="left", padx=4)
        self.nfc_badge_var = tk.StringVar(value="")
        self.nfc_badge = self._ctk_chip(mrz, "dim", textvariable=self.nfc_badge_var)
        self.nfc_badge.pack(side="left", padx=10)

        self.nfc_status_var = tk.StringVar(value="")
        ctk.CTkLabel(f, textvariable=self.nfc_status_var,
                     text_color=pal["ctk"]["text_dim"],
                     font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                                      size=pal["font_mono_sm"][1])
                     ).pack(anchor="w", padx=12, pady=(0, 4))

        # ---- Middle: splitter (state machine | APDU trace + inspect) ----
        body = ttk.PanedWindow(f, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(body)
        body.add(left, weight=1)
        right = ttk.Frame(body)
        body.add(right, weight=2)
        self._nfc_build_state_panel(left)
        self._nfc_build_keys_panel(left)
        self._nfc_build_trace_panel(right)

        # ---- Bottom: results cards ------------------------------------
        bottom = ctk.CTkFrame(f, fg_color="transparent")
        bottom.pack(fill="x", padx=6, pady=(0, 6))
        cards = ctk.CTkFrame(bottom, fg_color="transparent")
        cards.pack(fill="x")
        self._nfc_build_dg1_card(cards)
        self._nfc_build_sod_card(cards)
        self._nfc_build_progress_card(cards)
        self._nfc_build_adv_panel(bottom)

    def _nfc_build_state_panel(self, parent):
        outer, body = self._ctk_card(parent, "密码学与认证状态机")
        pal = self._pal
        chip = pal["chip"]
        self._nfc_states = []
        for title in ("① MRZ 因子校验", "② 基础密钥派生 Kseed/Kenc/Kmac",
                      "③ 相互认证 (BAC)", "④ Secure Messaging Ready"):
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(
                row, text=title, anchor="w",
                text_color=pal["ctk"]["text"],
                font=ctk.CTkFont(family=pal["font_ui"][0],
                                 size=pal["font_ui"][1])
                ).pack(side="left")
            var = tk.StringVar(value="  ARMED  ")
            chip_lb = self._ctk_chip(row, "idle", textvariable=var)
            chip_lb.pack(side="right")
            self._nfc_states.append((title, var, chip_lb))

    def _nfc_build_keys_panel(self, parent):
        outer, body = self._ctk_card(parent, "会话密钥（点击复制）")
        self._nfc_key_vars = {}
        pal = self._pal
        for name in ("Kseed", "Kenc", "Kmac", "KSenc", "KSmac"):
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(
                row, text=name, width=60, anchor="w",
                text_color=pal["ctk"]["accent"],
                font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                                 size=pal["font_mono_sm"][1],
                                 weight="bold")
                ).pack(side="left")
            var = tk.StringVar(value="—")
            en = ctk.CTkEntry(
                row, textvariable=var,
                fg_color=pal["ctk"]["entry_bg"],
                border_color=pal["ctk"]["border"],
                text_color=pal["ctk"]["accent"],
                font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                                 size=pal["font_mono_sm"][1]),
                height=26, corner_radius=6, state="readonly")
            en.pack(side="left", fill="x", expand=True, padx=(4, 4))
            self._ctk_btn_ghost(
                row, "复制",
                command=lambda n=name, v=var: self._nfc_copy_key(n, v),
                width=56).pack(side="right")
            self._nfc_key_vars[name] = var
            self._nfc_key_entries[name] = en
            ttk.Button(row, text="复制", width=4,
                       command=lambda n=name, v=var: self._nfc_copy_key(n, v)
                       ).pack(side="left")
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
                        ("hex", 260, "Hex 报文 (C → R)"), ("sw", 56, "SW"),
                        ("ms", 48, "耗时")]:
            self.nfc_trace.heading(c, text=t)
            self.nfc_trace.column(c, width=w,
                                  anchor="center" if c in ("i", "sw", "ms") else "w")
        self.nfc_trace.tag_configure("sw_ok", background="#e8f5e9",
                                     foreground="#1b5e20")
        self.nfc_trace.tag_configure("sw_bad", background="#ffebee",
                                     foreground="#b71c1c")
        ysb = ttk.Scrollbar(lab, orient="vertical", command=self.nfc_trace.yview)
        self.nfc_trace.configure(yscrollcommand=ysb.set)
        self.nfc_trace.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.nfc_trace.bind("<<TreeviewSelect>>", self._nfc_on_trace_select)

    def _nfc_build_dg1_card(self, parent):
        outer, body = self._ctk_card(parent, "DG1 持证人信息")
        outer.pack(side="left", fill="both", expand=True, padx=4, pady=4,
                   anchor="n")
        pal = self._pal
        self._nfc_dg1_vars = {}
        body.grid_columnconfigure(1, weight=1)
        for r, (key, label) in enumerate(
                [("name", "姓名"), ("doc", "证件号"), ("nat", "国籍"),
                 ("dob", "出生日期"), ("exp", "有效期")]):
            ctk.CTkLabel(
                body, text=label + ":", anchor="e",
                text_color=pal["ctk"]["text_dim"],
                font=ctk.CTkFont(family=pal["font_ui"][0],
                                 size=pal["font_ui"][1])
                ).grid(row=r, column=0, sticky="e", padx=(4, 4), pady=2)
            var = tk.StringVar(value="—")
            ctk.CTkLabel(
                body, textvariable=var, anchor="w",
                text_color=pal["ctk"]["text"],
                font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                                 size=pal["font_mono_sm"][1],
                                 weight="bold")
                ).grid(row=r, column=1, sticky="w", padx=(0, 8), pady=2)
            self._nfc_dg1_lbs[key] = None
            self._nfc_dg1_vars[key] = var

    def _nfc_build_sod_card(self, parent):
        outer, body = self._ctk_card(parent, "SOD 被动认证 (Passive Auth)")
        outer.pack(side="left", fill="both", expand=True, padx=4, pady=4,
                   anchor="n")
        pal = self._pal
        self._nfc_sod_sha = tk.StringVar(value="—")
        self._nfc_sod_rsa = tk.StringVar(value="—")
        body.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            body, text="DG1 SHA-256:", anchor="e",
            text_color=pal["ctk"]["text_dim"]).grid(
                row=0, column=0, sticky="e", padx=(4, 4), pady=2)
        ctk.CTkLabel(
            body, textvariable=self._nfc_sod_sha, anchor="w",
            text_color=pal["ctk"]["accent"],
            font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                             size=pal["font_mono_sm"][1])
            ).grid(row=0, column=1, sticky="w", padx=(0, 8), pady=2)
        ctk.CTkLabel(
            body, text="RSA 验签:", anchor="e",
            text_color=pal["ctk"]["text_dim"]).grid(
                row=1, column=0, sticky="e", padx=(4, 4), pady=2)
        ctk.CTkLabel(
            body, textvariable=self._nfc_sod_rsa, anchor="w",
            text_color=pal["ctk"]["accent"],
            font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                             size=pal["font_mono_sm"][1])
            ).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=2)

    def _nfc_build_progress_card(self, parent):
        outer, body = self._ctk_card(parent, "读取进度")
        outer.pack(side="left", fill="both", expand=True, padx=4, pady=4,
                   anchor="n")
        pal = self._pal
        self._nfc_prog_vars = {}
        for i, ef in enumerate(("EF.COM", "EF.DG1", "EF.DG2", "EF.SOD")):
            r, c = divmod(i, 2)
            var = tk.StringVar(value=" — ")
            chip_lb = self._ctk_chip(body, "idle", textvariable=var)
            chip_lb.grid(row=r, column=c * 2, sticky="e", padx=(4, 4), pady=4)
            ctk.CTkLabel(
                body, text=ef,
                text_color=pal["ctk"]["text_dim"]).grid(
                    row=r, column=c * 2 + 1, sticky="w", padx=(0, 10), pady=4)
            self._nfc_prog_lbs[ef] = None
            self._nfc_prog_vars[ef] = (var, chip_lb)

    def _nfc_build_adv_panel(self, parent):
        self._adv_open = tk.BooleanVar(value=False)
        self._adv_frame = ttk.LabelFrame(parent, text="高级密码学参数")
        self._adv_frame.pack(fill="x", padx=2, pady=2)
        head = ttk.Frame(self._adv_frame); head.pack(fill="x")
        ttk.Button(head, text="展开 / 折叠",
                   command=self._nfc_toggle_adv).pack(side="left", padx=4)
        # wrap="none" disables line wrapping, which means long hex lines
        # were silently clipped at the right edge with no way to scroll.
        # Keep nowrap for hex fidelity, but attach a horizontal Scrollbar
        # and a reasonable vertical Scrollbar so the full payload is
        # reachable regardless of panel width or content length.
        pal = self._pal
        body = ttk.Frame(self._adv_frame)
        self._adv_text_body = body
        # Create the Text widget FIRST so scrollbars can bind to its
        # xview/yview directly without relying on late-bound attribute
        # access.
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
        if self._adv_open.get():
            body.pack_forget()
            self._adv_open.set(False)
        else:
            body.pack(fill="x", padx=4, pady=2)
            self._adv_open.set(True)

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
        # Visual-pass 3.0: dual-mode reset. The chips live in CTkLabel
        # under CustomTkinter (fg_color/text_color) and tk.Label under
        # ttkbootstrap (bg/fg). Try ctk first, then fall back.
        chip_dim = self._pal["chip"]["dim"]
        def _reset_chip(lb):
            try:
                lb.configure(fg_color=chip_dim["bg"],
                             text_color=chip_dim["fg"])
            except (tk.TclError, ValueError):
                try:
                    lb.configure(bg=chip_dim["bg"], fg=chip_dim["fg"])
                except tk.TclError:
                    pass
        for _, var, lb in self._nfc_states:
            var.set("  ARMED  ")
            _reset_chip(lb)
        for var in self._nfc_dg1_vars.values():
            var.set("—")
        self._nfc_sod_sha.set("—")
        self._nfc_sod_rsa.set("—")
        for _, lb in self._nfc_prog_vars.values():
            _reset_chip(lb)
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
        chosen = chip["ok"] if ok else chip["fail"]
        self.nfc_badge_var.set(" ✓ BAC PASS " if ok else " ✗ BAC FAIL ")
        try:
            self.nfc_badge.configure(fg_color=chosen["bg"],
                                     text_color=chosen["fg"])
        except tk.TclError:
            self.nfc_badge.configure(bg=chosen["bg"], fg=chosen["fg"])
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
            chosen = chip["ok"] if good else chip["fail"]
            var.set(" ✓ PASS " if good else " ✗ FAIL ")
            try:
                lb.configure(fg_color=chosen["bg"], text_color=chosen["fg"])
            except tk.TclError:
                lb.configure(bg=chosen["bg"], fg=chosen["fg"])

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
                tag = "sw_ok" if sw == "9000" else "sw_bad"
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
            chosen = chip["ok"] if got else chip["idle"]
            var.set(" ✓ " if got else " — ")
            try:
                lb.configure(fg_color=chosen["bg"], text_color=chosen["fg"])
            except tk.TclError:
                lb.configure(bg=chosen["bg"], fg=chosen["fg"])

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
        self.ac_out = tk.Text(f, font=("Menlo", 11), bg="#ffffff", fg="#111827")
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
        # ---- Title strip ----
        title_row = ctk.CTkFrame(f, fg_color=pal["ctk"]["fg_color"],
                                  corner_radius=10, border_width=1,
                                  border_color=pal["ctk"]["border"])
        title_row.pack(fill="x", padx=6, pady=(8, 4))
        ctk.CTkLabel(
            title_row, text="  照片比对  ",
            text_color=pal["ctk"]["text"],
            font=ctk.CTkFont(family=pal["font_ui_bold"][0],
                             size=pal["font_ui_bold"][1],
                             weight="bold"),
        ).pack(side="left", padx=(10, 4), pady=6)
        ctk.CTkLabel(
            title_row, text="（face_a / face_b / face_c 已预置，可直接点“比对”）",
            text_color=pal["ctk"]["text_dim"],
            font=ctk.CTkFont(family=pal["font_ui"][0],
                             size=pal["font_ui"][1]),
        ).pack(side="left", padx=4)

        # ---- Reference picker row ----
        ref = ctk.CTkFrame(f, fg_color=pal["ctk"]["fg_color"],
                           corner_radius=10, border_width=1,
                           border_color=pal["ctk"]["border"])
        ref.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(ref, text="内置参考样本:",
                     text_color=pal["ctk"]["text_dim"]).pack(
                         side="left", padx=(10, 4), pady=6)
        self.face_ref = tk.StringVar()
        self.face_ref_cb = ctk.CTkOptionMenu(
            ref, variable=self.face_ref,
            values=list(FACE_SAMPLES.keys()),
            fg_color=pal["ctk"]["entry_bg"],
            button_color=pal["ctk"]["primary"],
            button_hover_color=pal["ctk"]["primary_h"],
            text_color=pal["ctk"]["text"],
            dropdown_fg_color=pal["ctk"]["top_fg"],
            dropdown_text_color=pal["ctk"]["text"],
            dropdown_hover_color=pal["ctk"]["hover"],
            width=240, height=28, corner_radius=6,
        )
        self.face_ref_cb.pack(side="left", padx=4)
        try:
            self.face_ref_cb.set(list(FACE_SAMPLES.keys())[0])
        except Exception:
            pass
        self._ctk_btn_ghost(
            ref, "填入 -> 照片A",
            command=self._face_load_ref, width=140).pack(
                side="left", padx=6)

        # ---- File picker rows (A & B) ----
        def _path_row(parent, label, var, picker):
            row = ctk.CTkFrame(parent, fg_color=pal["ctk"]["fg_color"],
                               corner_radius=10, border_width=1,
                               border_color=pal["ctk"]["border"])
            row.pack(fill="x", padx=6, pady=4)
            ctk.CTkLabel(row, text=label,
                         text_color=pal["ctk"]["text_dim"],
                         font=ctk.CTkFont(family=pal["font_ui"][0],
                                          size=pal["font_ui"][1])
                         ).pack(side="left", padx=(10, 4), pady=6)
            self._ctk_entry(row, var).pack(
                side="left", fill="x", expand=True, padx=(0, 6))
            self._ctk_btn_ghost(row, "浏览...",
                                command=picker, width=72).pack(
                                    side="right", padx=8)
            return row

        self.face_a = tk.StringVar(value="")
        self.face_b = tk.StringVar(value="")
        _path_row(f, "照片A:", self.face_a, lambda: self._face_pick("a"))
        _path_row(f, "照片B:", self.face_b, lambda: self._face_pick("b"))

        # ---- Action + result strip ----
        action_row = ctk.CTkFrame(f, fg_color="transparent")
        action_row.pack(fill="x", padx=6, pady=4)
        self._ctk_btn_primary(action_row, "▶ 比对",
                              command=self._face_run, width=140).pack(
                                  side="left", padx=(0, 8))
        self.face_score_chip = self._ctk_chip(
            action_row, "dim", text="  SCORE --  ")
        self.face_score_chip.pack(side="left", padx=4)
        self.face_method_var = tk.StringVar(value="")
        ctk.CTkLabel(
            action_row, textvariable=self.face_method_var,
            text_color=pal["ctk"]["text_dim"],
            font=ctk.CTkFont(family=pal["font_mono_sm"][0],
                             size=pal["font_mono_sm"][1])
        ).pack(side="left", padx=8)

        # ---- Body: two preview tiles + detail text ----
        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        body.grid_columnconfigure(0, weight=1, uniform="face")
        body.grid_columnconfigure(1, weight=1, uniform="face")
        body.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(body, text="  照片 A  ",
                     text_color=pal["ctk"]["accent"],
                     font=ctk.CTkFont(family=pal["font_ui_bold"][0],
                                      size=pal["font_ui_bold"][1],
                                      weight="bold")
                     ).grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))
        ctk.CTkLabel(body, text="  照片 B  ",
                     text_color=pal["ctk"]["accent"],
                     font=ctk.CTkFont(family=pal["font_ui_bold"][0],
                                      size=pal["font_ui_bold"][1],
                                      weight="bold")
                     ).grid(row=0, column=1, sticky="w", padx=4, pady=(0, 4))
        self.face_tile_a = ctk.CTkFrame(body, fg_color=pal["ctk"]["top_fg"],
                                        corner_radius=10, border_width=1,
                                        border_color=pal["ctk"]["border"])
        self.face_tile_a.grid(row=1, column=0, sticky="nsew",
                              padx=(4, 6), pady=4)
        self.face_tile_b = ctk.CTkFrame(body, fg_color=pal["ctk"]["top_fg"],
                                        corner_radius=10, border_width=1,
                                        border_color=pal["ctk"]["border"])
        self.face_tile_b.grid(row=1, column=1, sticky="nsew",
                              padx=(6, 4), pady=4)
        self.face_canvas_a = tk.Canvas(
            self.face_tile_a, width=256, height=256,
            bg=pal["code"], highlightthickness=0)
        self.face_canvas_a.pack(padx=10, pady=10)
        self.face_canvas_b = tk.Canvas(
            self.face_tile_b, width=256, height=256,
            bg=pal["code"], highlightthickness=0)
        self.face_canvas_b.pack(padx=10, pady=10)

        # Detail text under both tiles.
        detail = ctk.CTkFrame(f, fg_color=pal["ctk"]["fg_color"],
                              corner_radius=10, border_width=1,
                              border_color=pal["ctk"]["border"])
        detail.pack(fill="both", expand=False, padx=6, pady=(0, 8))
        ctk.CTkLabel(
            detail, text="  对比详情 (raw 输出 / 方法 / 耗时)",
            text_color=pal["ctk"]["text_dim"],
            font=ctk.CTkFont(family=pal["font_ui"][0],
                             size=pal["font_ui"][1])
        ).pack(anchor="w", padx=10, pady=(6, 2))
        self.face_out = tk.Text(
            detail, height=8, wrap="none",
            font=("Menlo", 10), bg=pal["code"], fg=pal["ctk"]["text"],
            insertbackground=pal["ctk"]["accent"], relief="flat",
            bd=0, padx=10, pady=6)
        self.face_out.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Image refs (kept alive so Tk does not garbage-collect the bitmaps).
        self._face_imgs = {}
        # Default face_a vs face_c -> 95/100 -> SUSPECT chip so the user
        # sees the score-chip update the first time they press 比对.
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

        Tk is single-threaded for image loading under tk.PhotoImage; for
        formats like JPG/PNG we fall back to PIL.ImageTk via the project's
        already-installed pillow. PPM/BMP are first-class via PhotoImage.
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
                fill=self._pal["ctk"]["text_dim"], anchor="center")
            return
        img = None
        # Native PhotoImage for PPM / PGM / GIF / PNG.
        try:
            img = tk.PhotoImage(file=path)
        except tk.TclError:
            img = None
        # Fallback: PIL for jpg / bmp / non-native PNG.
        if img is None:
            try:
                from PIL import Image, ImageTk  # type: ignore
                pil = Image.open(path).convert("RGB")
                img = ImageTk.PhotoImage(pil)
            except Exception:
                canvas.create_text(
                    128, 128, text="(unsupported format)",
                    fill=self._pal["ctk"]["text_dim"], anchor="center")
                return
        # Centre / fit the image into a 256x256 canvas keeping aspect ratio.
        cw, ch = 256, 256
        iw, ih = img.width(), img.height()
        scale = min(cw / iw, ch / ih)
        if scale < 1.0:
            # Render via PIL for proper downsampling; keep Tk PhotoImage
            # only when scale >= 1 to avoid hidden resize quality loss.
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
        # Keep a reference so Tk does not GC the bitmap.
        self._face_imgs[side] = img
        canvas.create_image((cw - iw) // 2, (ch - ih) // 2,
                            image=img, anchor="nw")
        # Subtle neon border to mirror the rest of the UI chrome.
        canvas.create_rectangle(0, 0, cw - 1, ch - 1, outline="#3A3B3C")

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
        # Always refresh previews when the user clicks 比对, in case the
        # path string was set programmatically (e.g. test harness).
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
                    # matches 'score : 95/100' / 'score:95/100'
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
                    fg_color=chip["bg"], text_color=chip["fg"],
                    text=f"  {label}  {score}/100  ")
            except (tk.TclError, ValueError):
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
    if not HAS_CTK:
        sys.stderr.write(
            "[tests/gui_ctk.py] customtkinter is not importable.\n"
            "  Activate the bundled venv and rerun:\n"
            "      ./.venv-ctk/bin/python tests/gui_ctk.py\n"
            "  Or install it into the active interpreter:\n"
            "      python3 -m pip install --user customtkinter\n")
        sys.stderr.flush()
        sys.exit(2)
    app = PassportGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
