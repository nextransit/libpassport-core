#!/usr/bin/env python3
"""passport_test_gui_ctk -- CustomTkinter Modern UI for the passport test bench.

A modern rewrite of gui.py using CustomTkinter for rounded corners,
dark/light appearance mode, and a cleaner card-based layout.  Reuses
the same CLI tools and ocr_bench_runner module so results are identical
to the original ttkbootstrap GUI.

Tabs (same as gui.py):
  1. MRZ OCR 测试       -- corpus-driven per-case benchmark
  2. 护照定位           -- locate MRZ band on a photo
  3. 照片→MRZ           -- photo direct recognition
  4. NFC 读卡           -- mock R-APDU eMRTD flow
  5. MRZ 解码           -- text TD3 -> fields / check digits
  6. 防伪验证           -- MRZ + sample -> AC report
  7. 照片比对           -- two image files -> SSIM similarity
"""
from __future__ import annotations
import os, subprocess, sys, threading, time, json, re, hashlib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ROOT = Path(__file__).resolve().parent.parent

from gui import (
    METHOD_LABELS, METHOD_ORDER, CRNN_BACKENDS,
    MRZ_TOOL, AC_TOOL, FACE_TOOL, GEN_SAMP, NFC_TOOL,
    tool_path, _load_ocrb_font, _ocrb_font,
    THEME_PREF_FILE, _load_theme_pref, _save_theme_pref,
    FAST_TIER_SINGLE, FAST_TIER_FUSION, FAST_TIER_DEFAULT,
    iter_passport_photos,
)

OCRB_FONT_NAME = None

METRIC_SPECS = [
    ("cases", "案例数", "#3B82F6"),
    ("mean",  "Mean (ms)", "#06B6D4"),
    ("p95",   "P95 (ms)", "#06B6D4"),
    ("pass",  "Pass (%)", "#10B981"),
    ("l1",    "L1 (%)",   "#10B981"),
    ("l2",    "L2 (%)",   "#10B981"),
    ("full",  "匹配",     "#F59E0B"),
]

TAB_SPECS = [
    ("ocr",     "⚡ MRZ OCR",      "MRZ OCR 综合测试"),
    ("loc",     "🎯 护照定位",     "护照 MRZ 区域定位"),
    ("restore", "🖼 照片→MRZ",     "照片直接识别 MRZ"),
    ("nfc",     "📡 NFC 读卡",     "NFC eMRTD 读卡"),
    ("mrz",     "📋 MRZ 解码",     "MRZ 文本解码 / 编码"),
    ("ac",      "🛡 防伪验证",     "MRZ + 样本 → 防伪报告"),
    ("face",    "👤 照片比对",     "两张照片 SSIM 比对"),
]


class MetricCard(ctk.CTkFrame):
    """A single KPI card: small grey title + large bold value + accent strip."""

    def __init__(self, master, title, accent_color, **kw):
        super().__init__(master, corner_radius=8, **kw)
        self._accent = accent_color
        self._title_lbl = ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60"), anchor="w")
        self._title_lbl.pack(anchor="w", padx=12, pady=(10, 0))
        self._value_var = tk.StringVar(value="--")
        self._value_lbl = ctk.CTkLabel(
            self, textvariable=self._value_var,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=accent_color, anchor="w")
        self._value_lbl.pack(anchor="w", padx=12, pady=(0, 10))

    def set_value(self, text):
        self._value_var.set(text)


class StatusPill(ctk.CTkLabel):
    """Capsule-style PASS / FAIL / SUSPECT badge."""

    _STYLES = {
        "PASS":    {"fg": "#10B981", "bg": "#D1FADF"},
        "FAIL":    {"fg": "#EF4444", "bg": "#FEE4E2"},
        "SUSPECT": {"fg": "#F59E0B", "bg": "#FEF3C7"},
        "OK":      {"fg": "#10B981", "bg": "#D1FADF"},
        "RUN":     {"fg": "#8B5CF6", "bg": "#EDE4FF"},
        "IDLE":    {"fg": "#6B7280", "bg": "#F3F4F6"},
    }

    def __init__(self, master, text="IDLE", **kw):
        s = self._STYLES.get(text, self._STYLES["IDLE"])
        super().__init__(master, text=f" {text} ",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=s["fg"], fg_color=s["bg"],
                         corner_radius=10, padx=6, pady=0, **kw)

    def set_status(self, text):
        s = self._STYLES.get(text, self._STYLES["IDLE"])
        self.configure(text=f" {text} ", text_color=s["fg"], fg_color=s["bg"])


class CtkPassportGUI(ctk.CTk):
    """Modern CustomTkinter rewrite of PassportGUI."""

    def __init__(self):
        super().__init__()
        self.title("PASSPORT TEST BENCH 护照机综合测试台 v3.0 (Modern UI)")
        self.geometry("1360x820")
        self.minsize(1100, 680)

        self._init_state()
        self._init_layout()
        self._build_sidebar()
        self._build_topbar()
        self._build_statusbar()
        self._build_all_tabs()
        self._apply_appearance()

    # ------------------------------------------------------------------
    # state + layout skeleton
    # ------------------------------------------------------------------
    def _init_state(self):
        self._current_tab = "ocr"
        self._appearance = _load_theme_pref() or "dark"
        if self._appearance not in ("dark", "light"):
            self._appearance = "dark"
        ctk.set_appearance_mode(self._appearance)
        self.ocr_status_var = tk.StringVar(value="就绪")
        self.ocr_progress_var = ctk.DoubleVar(value=0)
        self._ocr_running = False
        self._ocr_stop_flag = threading.Event()
        self._ocr_result_cache = {}
        self.ocr_metrics = {}
        self._preview_imgs = {}
        self._ocr_case_paths = {}
        self._ocr_results = []
        self.ocr_photo_mode = tk.BooleanVar(value=False)
        self.ocr_method_both = tk.BooleanVar(value=True)
        self.ocr_method_pick_var = tk.StringVar(value=METHOD_LABELS["traditional"])
        self.ocr_methods = tk.StringVar(value=",".join(METHOD_ORDER))
        self.ocr_weights_pic = tk.BooleanVar(value=False)
        self.ocr_set_var = tk.StringVar(value="默认（当前训练集）")
        self.ocr_filter_var = tk.StringVar(value="all")
        self.ocr_filter_pick_var = tk.StringVar(value="全部")
        self.ocr_search_var = tk.StringVar()
        self.ocr_photo_mode = tk.BooleanVar(value=False)

    def _init_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)   # topbar
        self.grid_rowconfigure(1, weight=1)   # content
        self.grid_rowconfigure(2, weight=0)   # statusbar

    # ------------------------------------------------------------------
    # sidebar (left navigation)
    # ------------------------------------------------------------------
    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=160, corner_radius=0,
                               fg_color=("gray90", "gray18"))
        sidebar.grid(row=0, column=0, rowspan=3, sticky="nsw")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(len(TAB_SPECS) + 2, weight=1)
        self._sidebar = sidebar

        ctk.CTkLabel(
            sidebar, text="🛂  PASSPORT",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E40AF", "#3B82F6")).grid(
            row=0, column=0, padx=12, pady=(18, 4))
        ctk.CTkLabel(
            sidebar, text="Test Bench v3.0",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray55")).grid(
            row=1, column=0, padx=12, pady=(0, 16))

        self._nav_buttons = {}
        for idx, (key, label, _desc) in enumerate(TAB_SPECS):
            btn = ctk.CTkButton(
                sidebar, text=label, anchor="w",
                font=ctk.CTkFont(size=13),
                fg_color="transparent",
                hover_color=("gray80", "gray25"),
                corner_radius=8, height=38,
                command=lambda k=key: self._switch_tab(k))
            btn.grid(row=idx + 2, column=0, sticky="ew", padx=8, pady=2)
            self._nav_buttons[key] = btn

        sep = ctk.CTkFrame(sidebar, height=1, fg_color=("gray75", "gray28"))
        sep.grid(row=len(TAB_SPECS) + 3, column=0, sticky="ew",
                 padx=12, pady=8)

        self._theme_switch = ctk.CTkSwitch(
            sidebar, text="深色模式",
            command=self._toggle_appearance,
            onvalue="dark", offvalue="light")
        self._theme_switch.grid(row=len(TAB_SPECS) + 4, column=0,
                                sticky="sw", padx=12, pady=(0, 16))
        if self._appearance == "dark":
            self._theme_switch.select()

    def _switch_tab(self, key):
        self._current_tab = key
        for k, btn in self._nav_buttons.items():
            if k == key:
                btn.configure(fg_color=("#DBEAFE", "#1E3A5F"),
                              text_color=("#1E40AF", "#60A5FA"))
            else:
                btn.configure(fg_color="transparent",
                              text_color=("gray20", "gray80"))
        for k, frame in self._tab_frames.items():
            if k == key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

    # ------------------------------------------------------------------
    # top bar (breadcrumb + quick actions)
    # ------------------------------------------------------------------
    def _build_topbar(self):
        topbar = ctk.CTkFrame(self, height=44, corner_radius=0,
                              fg_color=("gray88", "gray20"))
        topbar.grid(row=0, column=1, sticky="ew", columnspan=1)
        topbar.grid_propagate(False)
        topbar.grid_columnconfigure(1, weight=1)

        self._breadcrumb = ctk.CTkLabel(
            topbar, text="⚡ MRZ OCR  ›  MRZ OCR 综合测试",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("gray25", "gray85"))
        self._breadcrumb.grid(row=0, column=0, sticky="w", padx=16, pady=10)

        self._tool_chips_frame = ctk.CTkFrame(topbar, fg_color="transparent")
        self._tool_chips_frame.grid(row=0, column=2, sticky="e", padx=12)
        self._tool_chip_labels = {}
        for label, path in (("MRZ", MRZ_TOOL), ("AC", AC_TOOL),
                            ("FACE", FACE_TOOL), ("GEN", GEN_SAMP),
                            ("NFC", NFC_TOOL)):
            ok = path.exists()
            chip = ctk.CTkLabel(
                self._tool_chips_frame, text=f"● {label}",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=("#10B981" if ok else "#EF4444",
                            "#10B981" if ok else "#EF4444"))
            chip.pack(side="right", padx=4)
            self._tool_chip_labels[label] = (chip, ok)

        self._content_area = ctk.CTkFrame(self, fg_color="transparent",
                                           corner_radius=0)
        self._content_area.grid(row=1, column=1, sticky="nsew",
                                in_=self)
        self._content_area.grid_propagate(False)
        self._content_area.grid_rowconfigure(0, weight=1)
        self._content_area.grid_columnconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # status bar (bottom)
    # ------------------------------------------------------------------
    def _build_statusbar(self):
        sbar = ctk.CTkFrame(self, height=34, corner_radius=0,
                            fg_color=("gray85", "gray17"))
        sbar.grid(row=2, column=0, columnspan=2, sticky="ew")
        sbar.grid_propagate(False)
        sbar.grid_columnconfigure(1, weight=1)

        self._status_lbl = ctk.CTkLabel(
            sbar, textvariable=self.ocr_status_var,
            font=ctk.CTkFont(size=11), anchor="w",
            text_color=("gray35", "gray65"))
        self._status_lbl.grid(row=0, column=0, sticky="w", padx=12, pady=6)

        self.ocr_progress = ctk.CTkProgressBar(
            sbar, width=200, height=6)
        self.ocr_progress.set(0)

        self._progress_chip = StatusPill(sbar, text="IDLE")
        self._status_bar = sbar

    def _set_status(self, msg, pct=None):
        self.ocr_status_var.set(msg)
        if pct is not None:
            pct = max(0.0, min(100.0, float(pct)))
            self.ocr_progress.set(pct / 100.0)
            if not self.ocr_progress.winfo_ismapped():
                self.ocr_progress.grid(row=0, column=1, sticky="w",
                                       padx=(8, 4), pady=10)
            kind = "OK" if pct >= 100 else ("IDLE" if pct >= 40 else "RUN")
            self._progress_chip.set_status(kind)
            if not self._progress_chip.winfo_ismapped():
                self._progress_chip.grid(row=0, column=2, sticky="e",
                                         padx=4, pady=4)
            if pct >= 100:
                self.after(1200, self._hide_progress)
        else:
            self.ocr_progress.configure(mode="indeterminate")
            self.ocr_progress.start()
            if not self.ocr_progress.winfo_ismapped():
                self.ocr_progress.grid(row=0, column=1, sticky="w",
                                       padx=(8, 4), pady=10)
            self._progress_chip.set_status("RUN")
            if not self._progress_chip.winfo_ismapped():
                self._progress_chip.grid(row=0, column=2, sticky="e",
                                         padx=4, pady=4)

    def _hide_progress(self):
        try:
            self.ocr_progress.stop()
            self.ocr_progress.grid_forget()
            self._progress_chip.grid_forget()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # tab container
    # ------------------------------------------------------------------
    def _build_all_tabs(self):
        self._tab_frames = {}
        for key, label, desc in TAB_SPECS:
            frame = ctk.CTkFrame(self._content_area, fg_color="transparent",
                                 corner_radius=0)
            self._tab_frames[key] = frame
        self._build_ocr_tab()
        self._build_loc_tab()
        self._build_restore_tab()
        self._build_nfc_tab()
        self._build_mrz_tab()
        self._build_ac_tab()
        self._build_face_tab()
        self._switch_tab("ocr")
        self._ocr_load_corpus()

    def _get_tab(self, key):
        return self._tab_frames[key]

    # ------------------------------------------------------------------
    # OCR tab (main benchmark)
    # ------------------------------------------------------------------
    def _build_ocr_tab(self):
        f = self._get_tab("ocr")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)

        self._build_ocr_control(f)
        self._build_ocr_metrics(f)
        self._build_ocr_body(f)

    def _build_ocr_control(self, parent):
        ctrl = ctk.CTkFrame(parent, corner_radius=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        ctrl.grid_columnconfigure(12, weight=1)

        ctk.CTkLabel(ctrl, text="模式:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=0, padx=(12, 4), pady=8)
        ctk.CTkCheckBox(ctrl, text="图片直接识别",
                        variable=self.ocr_photo_mode,
                        command=self._ocr_mode_switch) \
            .grid(row=0, column=1, padx=4, pady=8)

        ctk.CTkLabel(ctrl, text="方案:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=2, padx=(12, 4), pady=8)
        self.ocr_method_pick = ctk.CTkOptionMenu(
            ctrl, variable=self.ocr_method_pick_var,
            values=[METHOD_LABELS[k] for k in METHOD_ORDER],
            width=140, command=self._ocr_method_on_pick)
        self.ocr_method_pick.grid(row=0, column=3, padx=4, pady=8)
        ctk.CTkCheckBox(ctrl, text="全部方案",
                        variable=self.ocr_method_both,
                        command=self._ocr_method_toggle) \
            .grid(row=0, column=4, padx=4, pady=8)

        ctk.CTkLabel(ctrl, text="训练集:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=5, padx=(12, 4), pady=8)
        ctk.CTkCheckBox(ctrl, text="assets/pic",
                        variable=self.ocr_weights_pic,
                        command=self._ocr_weights_toggle) \
            .grid(row=0, column=6, padx=4, pady=8)

        ctk.CTkLabel(ctrl, text="照片集:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=7, padx=(12, 4), pady=8)
        self.ocr_set_pick = ctk.CTkOptionMenu(
            ctrl, variable=self.ocr_set_var,
            values=self._ocr_set_choices(), width=150,
            command=self._ocr_set_on_pick)
        self.ocr_set_pick.grid(row=0, column=8, padx=4, pady=8)

        ctk.CTkLabel(ctrl, text="过滤:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=9, padx=(12, 4), pady=8)
        self.ocr_filter_pick = ctk.CTkOptionMenu(
            ctrl, variable=self.ocr_filter_pick_var,
            values=["全部", "干净", "带噪"], width=80,
            command=self._ocr_filter_on_pick)
        self.ocr_filter_pick.grid(row=0, column=10, padx=4, pady=8)

        self.ocr_pick_btn = ctk.CTkButton(
            ctrl, text="🖼 直接识别图片",
            fg_color="transparent", border_width=1,
            border_color=("gray60", "gray40"),
            command=self._ocr_pick_and_recognize)
        self.ocr_pick_btn.grid(row=0, column=11, padx=4, pady=8)

        self.ocr_run_btn = ctk.CTkButton(
            ctrl, text="⚡ 开始测试",
            fg_color=("#3B82F6", "#1E40AF"),
            hover_color=("#2563EB", "#1E3A5F"),
            command=self._ocr_run)
        self.ocr_run_btn.grid(row=0, column=12, padx=(4, 12), pady=8,
                              sticky="e")

    def _build_ocr_metrics(self, parent):
        metrics_frame = ctk.CTkFrame(parent, fg_color="transparent",
                                     corner_radius=0)
        metrics_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        for i in range(len(METRIC_SPECS)):
            metrics_frame.grid_columnconfigure(i, weight=1, uniform="kpi")
        self._metric_cards = []
        for i, (key, label, color) in enumerate(METRIC_SPECS):
            card = MetricCard(metrics_frame, label, color)
            card.grid(row=0, column=i, sticky="nsew", padx=3, pady=2)
            self.ocr_metrics[key] = card._value_var
            self._metric_cards.append(card)

    def _build_ocr_body(self, parent):
        body = ttk.PanedWindow(parent, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew", padx=4, pady=2)
        self._ocr_body = body

        self._build_ocr_case_list(body)
        self._build_ocr_middle(body)
        self._build_ocr_preview(body)

    def _build_ocr_case_list(self, body):
        left = ctk.CTkFrame(body, corner_radius=8)
        body.add(left, weight=22)
        left.grid_rowconfigure(2, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="案例集列表",
                     font=ctk.CTkFont(size=12, weight="bold")) \
            .grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        btn_bar = ctk.CTkFrame(left, fg_color="transparent")
        btn_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
        ctk.CTkButton(btn_bar, text="全选", width=50,
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=lambda: self._ocr_select_all(True)) \
            .pack(side="left", padx=2)
        ctk.CTkButton(btn_bar, text="反选", width=50,
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=lambda: self._ocr_select_all(False)) \
            .pack(side="left", padx=2)
        ctk.CTkButton(btn_bar, text="清空", width=50,
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=self._ocr_clear_sel) \
            .pack(side="left", padx=2)
        search = ctk.CTkEntry(btn_bar, textvariable=self.ocr_search_var,
                              width=100, placeholder_text="搜索...")
        search.pack(side="right", padx=2)
        search.bind("<KeyRelease>", self._ocr_apply_search)

        tree_frame = ctk.CTkFrame(left, fg_color="transparent")
        tree_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.ocr_case_list = ttk.Treeview(
            tree_frame, columns=("id", "scale", "noise", "skew"),
            show="headings", selectmode="extended", height=22,
            style="Data.Treeview")
        for c, w in [("id", 200), ("scale", 50), ("noise", 50), ("skew", 46)]:
            self.ocr_case_list.heading(c, text=c)
            self.ocr_case_list.column(c, width=w, minwidth=40, anchor="w",
                                      stretch=(c == "id"))
        ysb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.ocr_case_list.yview)
        self.ocr_case_list.configure(yscrollcommand=ysb.set)
        self.ocr_case_list.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        self.ocr_case_list.bind("<<TreeviewSelect>>", self._ocr_on_case_select)

    def _build_ocr_middle(self, body):
        middle = ctk.CTkFrame(body, corner_radius=8)
        body.add(middle, weight=44)
        middle.grid_rowconfigure(2, weight=1)
        middle.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(middle, text="方案指标对比",
                     font=ctk.CTkFont(size=12, weight="bold")) \
            .grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))

        sum_frame = ctk.CTkFrame(middle, fg_color="transparent")
        sum_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
        sum_frame.grid_rowconfigure(0, weight=1)
        sum_frame.grid_columnconfigure(0, weight=1)
        self.ocr_summary = ttk.Treeview(
            sum_frame,
            columns=("method", "n", "ok", "okp", "ms_avg",
                     "l1_acc", "l2_acc", "full_match"),
            show="headings", height=3, style="Data.Treeview")
        col_widths = {"method": 90, "n": 48, "ok": 48, "okp": 60,
                      "ms_avg": 70, "l1_acc": 74, "l2_acc": 74,
                      "full_match": 80}
        for c, anc in [("method", "w"), ("n", "center"), ("ok", "center"),
                       ("okp", "center"), ("ms_avg", "center"),
                       ("l1_acc", "center"), ("l2_acc", "center"),
                       ("full_match", "center")]:
            self.ocr_summary.heading(c, text=c)
            self.ocr_summary.column(c, width=col_widths[c], minwidth=42,
                                    anchor=anc)
        self.ocr_summary.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(middle, text="详细结果",
                     font=ctk.CTkFont(size=12, weight="bold")) \
            .grid(row=2, column=0, sticky="nw", padx=12, pady=(8, 0))

        det_frame = ctk.CTkFrame(middle, fg_color="transparent")
        det_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(30, 8))
        det_frame.grid_rowconfigure(0, weight=1)
        det_frame.grid_columnconfigure(0, weight=1)
        self.ocr_detail = ttk.Treeview(
            det_frame, columns=("id", "method", "ok", "ms", "l1", "l2"),
            show="headings", height=14, style="Data.Treeview")
        col_widths = {"id": 140, "method": 86, "ok": 56,
                      "ms": 64, "l1": 120, "l2": 120}
        for c, anc in [("id", "w"), ("method", "w"), ("ok", "center"),
                       ("ms", "center"), ("l1", "center"), ("l2", "center")]:
            self.ocr_detail.heading(c, text=c)
            self.ocr_detail.column(c, width=col_widths[c], minwidth=48,
                                   anchor=anc, stretch=(c == "id"))
        ysb2 = ttk.Scrollbar(det_frame, orient="vertical",
                             command=self.ocr_detail.yview)
        xsb2 = ttk.Scrollbar(det_frame, orient="horizontal",
                             command=self.ocr_detail.xview)
        self.ocr_detail.configure(yscrollcommand=ysb2.set,
                                  xscrollcommand=xsb2.set)
        self.ocr_detail.grid(row=0, column=0, sticky="nsew")
        ysb2.grid(row=0, column=1, sticky="ns")
        xsb2.grid(row=1, column=0, sticky="ew")
        self.ocr_detail.bind("<<TreeviewSelect>>", self._ocr_on_detail_select)

    def _build_ocr_preview(self, body):
        right = ctk.CTkFrame(body, corner_radius=8)
        body.add(right, weight=26)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="图像预览 + 字符级视图",
                     font=ctk.CTkFont(size=12, weight="bold")) \
            .grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        self._preview_pil = None
        self._preview_ox = 0
        self._preview_oy = 0
        self._preview_zoom = 1.0
        self._preview_drag = None
        dark = ctk.get_appearance_mode() == "Dark"
        self.ocr_preview_canvas = tk.Canvas(
            right, relief="flat", bd=0,
            bg="#0B0F19" if dark else "#FFFFFF",
            highlightthickness=0)
        self.ocr_preview_canvas.grid(row=1, column=0, sticky="nsew",
                                     padx=4, pady=4)
        self.ocr_preview_canvas.bind("<ButtonPress-1>", self._on_preview_press)
        self.ocr_preview_canvas.bind("<B1-Motion>", self._on_preview_motion)
        self.ocr_preview_canvas.bind("<ButtonRelease-1>",
                                     self._on_preview_release)
        self.ocr_preview_canvas.bind("<MouseWheel>", self._on_preview_wheel)
        self.ocr_preview_canvas.bind("<Configure>",
                                     lambda _e: self._render_preview())

        self.ocr_diff_text = tk.Text(
            right, height=8, wrap="none", relief="flat",
            bg="#0B0F19" if dark else "#FFFFFF",
            fg="#E2E8F0" if dark else "#1a2433",
            font=("Menlo", 11), padx=8, pady=6)
        self.ocr_diff_text.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        right.grid_rowconfigure(2, weight=1)

    def _on_preview_press(self, event):
        self._preview_drag = (event.x, event.y)

    def _on_preview_motion(self, event):
        if self._preview_drag is None:
            return
        dx = event.x - self._preview_drag[0]
        dy = event.y - self._preview_drag[1]
        self._preview_ox += dx
        self._preview_oy += dy
        self._preview_drag = (event.x, event.y)
        self._render_preview()

    def _on_preview_release(self, _event):
        self._preview_drag = None

    def _on_preview_wheel(self, event):
        if self._preview_pil is None:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._preview_zoom = max(0.1, min(8.0,
                                          self._preview_zoom * factor))
        self._render_preview()

    def _render_preview(self):
        cv = getattr(self, "ocr_preview_canvas", None)
        if cv is None:
            return
        cv.delete("all")
        im = self._preview_pil
        if im is None:
            cv.create_text(cv.winfo_width() // 2 or 160,
                           cv.winfo_height() // 2 or 120,
                           text="（选择案例查看预览）",
                           fill="#6B7280", font=("Segoe UI", 11))
            return
        cw = cv.winfo_width()
        ch = cv.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 320, 200
        iw, ih = im.size
        base_scale = min(cw / iw, ch / ih)
        scale = base_scale * self._preview_zoom
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        im2 = im.resize((dw, dh), Image.LANCZOS)
        self._preview_tkimg = ImageTk.PhotoImage(im2)
        ox = self._preview_ox + (cw - dw) // 2
        oy = self._preview_oy + (ch - dh) // 2
        cv.create_image(ox, oy, anchor="nw", image=self._preview_tkimg)

    # ------------------------------------------------------------------
    # OCR tab: business logic (reuses ocr_bench_runner)
    # ------------------------------------------------------------------
    def _ocr_set_defs(self):
        """The named photo sets, or an empty dict when the file is absent."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import photo_sets
            return photo_sets.load_sets().get("sets", {})
        except Exception:
            return {}

    def _ocr_set_choices(self):
        defs = self._ocr_set_defs()
        return ["默认（当前训练集）"] + sorted(defs)

    def _ocr_current_set(self):
        pick = str(self.ocr_set_var.get() if hasattr(self, "ocr_set_var")
                   else "").strip()
        if pick in ("", "默认（当前训练集）"):
            return ""
        return pick if pick in self._ocr_set_defs() else ""

    def _ocr_set_entries(self):
        name = self._ocr_current_set()
        if not name:
            return []
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import photo_sets
            return photo_sets.resolve(name)
        except Exception:
            return []

    def _ocr_set_on_pick(self, _val=None):
        name = self._ocr_current_set()
        defs = self._ocr_set_defs()
        if name and defs.get(name, {}).get("weights") in ("pic", "passport"):
            want_pic = defs[name]["weights"] == "pic"
            if bool(self.ocr_weights_pic.get()) != want_pic:
                self.ocr_weights_pic.set(want_pic)
                if getattr(self, "_ocr_result_cache", None):
                    self._ocr_result_cache.clear()
        self._ocr_reset_results()
        if name:
            if not self.ocr_photo_mode.get():
                self.ocr_photo_mode.set(True)
            self._ocr_mode_switch()
        elif self.ocr_photo_mode.get():
            self._ocr_mode_switch()

    def _ocr_reset_results(self):
        for tree in (getattr(self, "ocr_detail", None),
                     getattr(self, "ocr_summary", None)):
            if tree is not None:
                for item in tree.get_children():
                    tree.delete(item)
        for key in self.ocr_metrics:
            self.ocr_metrics[key].set("--")

    def _ocr_use_pic_weights(self):
        try:
            return bool(self.ocr_weights_pic.get())
        except Exception:
            return False

    def _weights_set(self):
        return "pic" if self._ocr_use_pic_weights() else "passport"

    def _ocr_mode_switch(self):
        if self.ocr_photo_mode.get():
            self.ocr_run_btn.configure(text="⚡ 开始测试")
            for iid in self.ocr_case_list.get_children():
                self.ocr_case_list.delete(iid)
            self._ocr_case_order = []
            self._ocr_photo_entries = (self._ocr_set_entries()
                                       or list(iter_passport_photos(
                                           self._weights_set())))
            self._ocr_photo_paths = {}
            self._ocr_case_base_tags = {}
            for label, _f in self._ocr_photo_entries:
                self._ocr_photo_paths[label] = str(_f)
                self.ocr_case_list.insert("", "end", iid=label,
                                          values=(Path(label).name, "", "", ""),
                                          tags=("clean",))
                self._ocr_case_order.append(label)
                self._ocr_case_base_tags[label] = ("clean",)
            self._ocr_case_paths = dict(self._ocr_photo_paths)
        else:
            self.ocr_run_btn.configure(text="⚡ 开始测试")
            self._ocr_load_corpus()

    def _ocr_load_corpus(self):
        for iid in self.ocr_case_list.get_children():
            self.ocr_case_list.delete(iid)
        self._ocr_case_order = []
        self._ocr_case_paths = {}
        self._ocr_corpus = {"records": []}
        self._ocr_case_base_tags = {}
        try:
            from ocr_bench_runner import CORPUS_JSON
            corpus_path = Path(CORPUS_JSON)
            if corpus_path.exists():
                data = json.loads(corpus_path.read_text())
                for rec in data["records"]:
                    tag = "noisy" if rec["noise"] > 0 else "clean"
                    iid = rec["id"]
                    self._ocr_case_base_tags[iid] = (tag,)
                    self.ocr_case_list.insert("", "end", iid=iid,
                        values=(Path(rec["id"]).name,
                                rec["scale"], rec["noise"], rec["skew"]),
                        tags=(tag,))
                    self._ocr_case_order.append(iid)
                    self._ocr_case_paths[iid] = str(
                        corpus_path.parent / rec["image"])
                self._ocr_corpus = data
        except Exception:
            pass

    def _ocr_method_toggle(self):
        state = "disabled" if self.ocr_method_both.get() else "normal"
        self.ocr_method_pick.configure(state=state)

    def _ocr_method_on_pick(self, _val=None):
        pass

    def _ocr_weights_toggle(self):
        self._ocr_result_cache.clear()

    def _ocr_filter_on_pick(self, _val=None):
        val = self.ocr_filter_pick_var.get()
        self.ocr_filter_var.set({"全部": "all", "干净": "clean",
                                 "带噪": "noisy"}.get(val, "all"))
        self._ocr_apply_filter()

    def _ocr_select_all(self, on):
        items = self.ocr_case_list.get_children()
        if on:
            self.ocr_case_list.selection_set(items)
        else:
            sel = self.ocr_case_list.selection()
            self.ocr_case_list.selection_set(
                [i for i in items if i not in sel])

    def _ocr_clear_sel(self):
        self.ocr_case_list.selection_remove(
            self.ocr_case_list.get_children())

    def _ocr_apply_filter(self):
        pass

    def _ocr_apply_search(self, _evt=None):
        q = self.ocr_search_var.get().lower().strip()
        for iid in self.ocr_case_list.get_children():
            text = self.ocr_case_list.item(iid, "text")
            vals = self.ocr_case_list.item(iid, "values")
            hay = (str(text) + " " + " ".join(str(v) for v in vals)).lower()
            self.ocr_case_list.item(iid, tags=())
            if q and q not in hay:
                self.ocr_case_list.item(iid, tags=("hidden",))
        try:
            self.ocr_case_list.tag_configure("hidden", foreground="")
        except tk.TclError:
            pass

    def _ocr_on_case_select(self, _evt=None):
        sel = self.ocr_case_list.selection()
        if not sel:
            return
        iid = sel[0]
        path = self._ocr_case_paths.get(iid)
        if path and Path(path).exists():
            self._load_preview(iid, path)

    def _ocr_on_detail_select(self, _evt=None):
        pass

    def _load_preview(self, cid, path):
        try:
            img = Image.open(path).convert("RGB")
            self._preview_pil = img
            self._preview_ox = 0
            self._preview_oy = 0
            self._preview_zoom = 1.0
            self._render_preview()
        except Exception:
            self._preview_pil = None
            self._render_preview()

    def _ocr_pick_and_recognize(self):
        paths = filedialog.askopenfilenames(
            title="选择护照照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not paths:
            return
        for p in paths:
            self._ocr_result_cache.pop(p, None)
        self._ocr_run_photos(list(paths))

    def _ocr_run(self):
        sel = self.ocr_case_list.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在左侧列表选择案例")
            return
        if self._ocr_running:
            return
        self._ocr_running = True
        self._ocr_stop_flag.clear()
        self.ocr_run_btn.configure(state="disabled", text="⏳ 识别中…")
        threading.Thread(target=self._ocr_run_worker,
                          args=(sel,), daemon=True).start()

    def _ocr_run_worker(self, sel):
        try:
            from ocr_bench_runner import run_one, compare, CORPUS_JSON
            methods = self._ocr_selected_methods()
            tool = self._ocr_weights_tool()
            results = []
            total = len(sel) * len(methods)
            done = 0
            for iid in sel:
                if self._ocr_stop_flag.is_set():
                    break
                cid = self.ocr_case_list.item(iid, "text")
                img_path = self._ocr_case_paths.get(iid)
                if not img_path or not Path(img_path).exists():
                    done += len(methods)
                    continue
                for method in methods:
                    if self._ocr_stop_flag.is_set():
                        break
                    from ocr_bench_runner import METHODS, METHOD_ENV
                    m_tool = METHODS.get(method, (None, tool))[1]
                    env = METHOD_ENV.get(method)
                    rec = run_one(m_tool, Path(img_path), env)
                    rec["method"] = method
                    rec["cid"] = cid
                    results.append(rec)
                    done += 1
                    pct = done / total * 100
                    self.after(0, lambda d=done, t=total, p=pct:
                               self._set_status(
                                   f"识别中 {d}/{t}", p))
            self.after(0, lambda: self._ocr_populate_results(results))
            self.after(0, lambda: self._set_status(
                f"完成: {len(results)} 条结果", 100))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.after(0, lambda: self.ocr_run_btn.configure(
                state="normal", text="⚡ 开始测试"))
            self._ocr_running = False

    def _ocr_selected_methods(self):
        if self.ocr_method_both.get():
            return list(METHOD_ORDER)
        pick = self.ocr_method_pick_var.get()
        for k, v in METHOD_LABELS.items():
            if v == pick:
                return [k]
        return list(METHOD_ORDER)

    def _ocr_weights_tool(self):
        if self.ocr_weights_pic.get():
            pic_tool = tool_path("mrz_ocr_cnn_tool_pic",
                                 sub_dirs=("6_mrz_ocr",))
            return pic_tool if pic_tool.exists() else MRZ_TOOL
        return MRZ_TOOL

    def _ocr_run_photos(self, paths):
        if self._ocr_running:
            return
        self._ocr_running = True
        self._ocr_stop_flag.clear()
        self.ocr_run_btn.configure(state="disabled", text="⏳ 识别中…")
        threading.Thread(target=self._ocr_run_photos_worker,
                          args=(paths,), daemon=True).start()

    def _ocr_run_photos_worker(self, paths):
        try:
            from ocr_bench_runner import run_one, METHODS, METHOD_ENV
            methods = self._ocr_selected_methods()
            tool = self._ocr_weights_tool()
            results = []
            total = len(paths) * len(methods)
            done = 0
            for path in paths:
                if self._ocr_stop_flag.is_set():
                    break
                for method in methods:
                    if self._ocr_stop_flag.is_set():
                        break
                    m_tool = METHODS.get(method, (None, tool))[1]
                    env = METHOD_ENV.get(method)
                    rec = run_one(m_tool, Path(path), env)
                    rec["method"] = method
                    rec["cid"] = Path(path).name
                    results.append(rec)
                    done += 1
                    pct = done / total * 100
                    self.after(0, lambda d=done, t=total, p=pct:
                               self._set_status(f"识别中 {d}/{t}", p))
            self.after(0, lambda: self._ocr_populate_results(results))
            self.after(0, lambda: self._set_status(
                f"完成: {len(results)} 条结果", 100))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.after(0, lambda: self.ocr_run_btn.configure(
                state="normal", text="⚡ 开始测试"))
            self._ocr_running = False

    def _ocr_populate_results(self, results):
        self._ocr_results = results
        for item in self.ocr_detail.get_children():
            self.ocr_detail.delete(item)
        for rec in results:
            l1 = rec.get("line1", "")
            l2 = rec.get("line2", "")
            ok = "PASS" if rec.get("ok") and l1 and l2 else "FAIL"
            ms = f"{rec.get('ms', 0):.0f}"
            method = METHOD_LABELS.get(rec.get("method", ""),
                                       rec.get("method", ""))
            self.ocr_detail.insert("", "end",
                values=(rec.get("cid", ""), method, ok, ms, l1, l2),
                tags=(ok,))
        self._ocr_update_metrics(results)
        self._apply_treeview_tags()

    def _ocr_update_metrics(self, results):
        if not results:
            for key in self.ocr_metrics:
                self.ocr_metrics[key].set("--")
            return
        n = len(results)
        ok_count = sum(1 for r in results
                       if r.get("ok") and r.get("line1") and r.get("line2"))
        times = [r.get("ms", 0) for r in results]
        mean_ms = sum(times) / n
        sorted_t = sorted(times)
        p95_idx = min(len(sorted_t) - 1, int(len(sorted_t) * 0.95))
        p95_ms = sorted_t[p95_idx]
        pass_pct = ok_count / n * 100
        self.ocr_metrics["cases"].set(str(n))
        self.ocr_metrics["mean"].set(f"{mean_ms:.1f}")
        self.ocr_metrics["p95"].set(f"{p95_ms:.1f}")
        self.ocr_metrics["pass"].set(f"{pass_pct:.1f}")
        self.ocr_metrics["l1"].set("--")
        self.ocr_metrics["l2"].set("--")
        self.ocr_metrics["full"].set(f"{ok_count}/{n}")

    def _apply_treeview_tags(self):
        dark = ctk.get_appearance_mode() == "Dark"
        if dark:
            ok_bg, ok_fg = "#0d3a2a", "#7eeac2"
            fail_bg, fail_fg = "#43131f", "#ffaabb"
        else:
            ok_bg, ok_fg = "#F0FAF4", "#0F5132"
            fail_bg, fail_fg = "#FEF2F2", "#A00D20"
        for tree in (self.ocr_detail, self.ocr_summary,
                     getattr(self, "ocr_case_list", None)):
            if tree is None:
                continue
            try:
                tree.tag_configure("PASS", background=ok_bg, foreground=ok_fg)
                tree.tag_configure("OK", background=ok_bg, foreground=ok_fg)
                tree.tag_configure("FAIL", background=fail_bg,
                                   foreground=fail_fg)
                tree.tag_configure("REJECT", background=fail_bg,
                                   foreground=fail_fg)
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # 护照定位 tab
    # ------------------------------------------------------------------
    def _build_loc_tab(self):
        f = self._get_tab("loc")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(f, corner_radius=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ctrl.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(ctrl, text="图片:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=0, padx=(12, 4), pady=8)
        self.loc_path_var = tk.StringVar()
        ctk.CTkEntry(ctrl, textvariable=self.loc_path_var, width=300) \
            .grid(row=0, column=1, padx=4, pady=8)
        ctk.CTkButton(ctrl, text="浏览...",
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=self._loc_pick) \
            .grid(row=0, column=2, padx=4, pady=8)
        ctk.CTkButton(ctrl, text="🎯 定位 MRZ",
                      fg_color=("#3B82F6", "#1E40AF"),
                      hover_color=("#2563EB", "#1E3A5F"),
                      command=self._loc_run) \
            .grid(row=0, column=3, padx=(4, 12), pady=8, sticky="e")

        body = ctk.CTkFrame(f, corner_radius=8)
        body.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        self.loc_preview_lbl = ctk.CTkLabel(body, text="（选择图片开始定位）",
                                            text_color=("gray50", "gray55"))
        self.loc_preview_lbl.grid(row=0, column=0, padx=12, pady=12)

        self.loc_output = tk.Text(body, wrap="word", relief="flat",
                                  font=("Menlo", 11), padx=8, pady=6)
        self.loc_output.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

    def _loc_pick(self):
        p = filedialog.askopenfilename(
            title="选择护照图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if p:
            self.loc_path_var.set(p)
            try:
                img = Image.open(p)
                img.thumbnail((500, 350))
                tk_img = ImageTk.PhotoImage(img)
                self._preview_imgs["loc"] = tk_img
                self.loc_preview_lbl.configure(image=tk_img, text="")
            except Exception:
                self.loc_preview_lbl.configure(image=None,
                                               text="（无法加载预览）")

    def _loc_run(self):
        path = self.loc_path_var.get()
        if not path or not Path(path).exists():
            messagebox.showinfo("提示", "请先选择图片")
            return
        self.loc_output.delete("1.0", "end")
        self.loc_output.insert("end", "定位中…\n")
        threading.Thread(target=self._loc_run_worker, args=(path,),
                          daemon=True).start()

    def _loc_run_worker(self, path):
        try:
            from passport_locator import locate_regions, annotate
            img = Image.open(path)
            regions = locate_regions(img)
            self.after(0, lambda: self._loc_show_result(path, regions))
        except Exception as e:
            self.after(0, lambda: self.loc_output.delete("1.0", "end"))
            self.after(0, lambda: self.loc_output.insert("end", f"错误: {e}\n"))

    def _loc_show_result(self, path, regions):
        self.loc_output.delete("1.0", "end")
        self.loc_output.insert("end", f"图片: {path}\n\n")
        if not regions:
            self.loc_output.insert("end", "未检测到 MRZ 区域\n")
            return
        for i, r in enumerate(regions):
            self.loc_output.insert("end",
                f"区域 {i+1}: {r}\n")

    # ------------------------------------------------------------------
    # 照片→MRZ tab
    # ------------------------------------------------------------------
    def _build_restore_tab(self):
        f = self._get_tab("restore")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(f, corner_radius=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ctrl.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(ctrl, text="图片:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=0, padx=(12, 4), pady=8)
        self.rst_path_var = tk.StringVar()
        ctk.CTkEntry(ctrl, textvariable=self.rst_path_var, width=300) \
            .grid(row=0, column=1, padx=4, pady=8)
        ctk.CTkButton(ctrl, text="浏览...",
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=self._rst_pick) \
            .grid(row=0, column=2, padx=4, pady=8, sticky="w")
        ctk.CTkButton(ctrl, text="🔍 识别 MRZ",
                      fg_color=("#3B82F6", "#1E40AF"),
                      hover_color=("#2563EB", "#1E3A5F"),
                      command=self._rst_run) \
            .grid(row=0, column=3, padx=(4, 12), pady=8, sticky="e")

        body = ctk.CTkFrame(f, corner_radius=8)
        body.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        self.rst_preview_lbl = ctk.CTkLabel(body, text="（选择图片开始识别）",
                                            text_color=("gray50", "gray55"))
        self.rst_preview_lbl.grid(row=0, column=0, padx=12, pady=12)

        self.rst_info = tk.Text(body, wrap="word", relief="flat",
                                font=("Menlo", 11), padx=8, pady=6)
        self.rst_info.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

    def _rst_pick(self):
        p = filedialog.askopenfilename(
            title="选择护照照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if p:
            self.rst_path_var.set(p)
            try:
                img = Image.open(p)
                img.thumbnail((500, 350))
                tk_img = ImageTk.PhotoImage(img)
                self._preview_imgs["rst"] = tk_img
                self.rst_preview_lbl.configure(image=tk_img, text="")
            except Exception:
                self.rst_preview_lbl.configure(image=None,
                                               text="（无法加载预览）")

    def _rst_run(self):
        path = self.rst_path_var.get()
        if not path or not Path(path).exists():
            messagebox.showinfo("提示", "请先选择图片")
            return
        self.rst_info.delete("1.0", "end")
        self.rst_info.insert("end", "识别中…\n")
        threading.Thread(target=self._rst_run_worker, args=(path,),
                          daemon=True).start()

    def _rst_run_worker(self, path):
        try:
            from ocr_bench_runner import run_one, METHODS, METHOD_ENV
            tool = self._ocr_weights_tool()
            rec = run_one(tool, Path(path))
            self.after(0, lambda: self._rst_show_result(path, rec))
        except Exception as e:
            self.after(0, lambda: self.rst_info.delete("1.0", "end"))
            self.after(0, lambda: self.rst_info.insert("end", f"错误: {e}\n"))

    def _rst_show_result(self, path, rec):
        self.rst_info.delete("1.0", "end")
        self.rst_info.insert("end", f"图片: {path}\n")
        self.rst_info.insert("end", f"耗时: {rec.get('ms', 0):.0f} ms\n\n")
        self.rst_info.insert("end", f"Line1: {rec.get('line1', '')}\n")
        self.rst_info.insert("end", f"Line2: {rec.get('line2', '')}\n\n")
        self.rst_info.insert("end",
            f"Conf1: {rec.get('conf1', 0)}\n"
            f"Conf2: {rec.get('conf2', 0)}\n")
        if rec.get("err"):
            self.rst_info.insert("end", f"\n错误: {rec['err']}\n")

    # ------------------------------------------------------------------
    # NFC tab
    # ------------------------------------------------------------------
    def _build_nfc_tab(self):
        f = self._get_tab("nfc")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(f, corner_radius=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ctrl.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(ctrl, text="脚本:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=0, padx=(12, 4), pady=8)
        self.nfc_script_var = tk.StringVar()
        ctk.CTkEntry(ctrl, textvariable=self.nfc_script_var, width=300) \
            .grid(row=0, column=1, padx=4, pady=8)
        ctk.CTkButton(ctrl, text="浏览...",
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=self._nfc_pick_script) \
            .grid(row=0, column=2, padx=4, pady=8, sticky="w")
        ctk.CTkButton(ctrl, text="📡 读卡",
                      fg_color=("#3B82F6", "#1E40AF"),
                      hover_color=("#2563EB", "#1E3A5F"),
                      command=self._nfc_run) \
            .grid(row=0, column=3, padx=(4, 12), pady=8, sticky="e")

        body = ctk.CTkFrame(f, corner_radius=8)
        body.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.nfc_output = tk.Text(body, wrap="word", relief="flat",
                                  font=("Menlo", 11), padx=8, pady=6)
        self.nfc_output.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _nfc_pick_script(self):
        p = filedialog.askopenfilename(
            title="选择 NFC 脚本",
            filetypes=[("JSON", "*.json"), ("所有", "*.*")])
        if p:
            self.nfc_script_var.set(p)

    def _nfc_run(self):
        script = self.nfc_script_var.get()
        if not script:
            messagebox.showinfo("提示", "请先选择 NFC 脚本")
            return
        self.nfc_output.delete("1.0", "end")
        self.nfc_output.insert("end", "读卡中…\n")
        threading.Thread(target=self._nfc_run_worker, args=(script,),
                          daemon=True).start()

    def _nfc_run_worker(self, script):
        try:
            r = subprocess.run([str(NFC_TOOL), script],
                               capture_output=True, text=True, timeout=30)
            self.after(0, lambda: self._nfc_show(r.returncode,
                                                  r.stdout, r.stderr))
        except Exception as e:
            self.after(0, lambda: self.nfc_output.delete("1.0", "end"))
            self.after(0, lambda: self.nfc_output.insert("end", f"错误: {e}\n"))

    def _nfc_show(self, rc, stdout, stderr):
        self.nfc_output.delete("1.0", "end")
        self.nfc_output.insert("end", f"返回码: {rc}\n\n")
        self.nfc_output.insert("end", stdout)
        if stderr:
            self.nfc_output.insert("end", f"\n[stderr]\n{stderr}\n")

    # ------------------------------------------------------------------
    # MRZ 解码 tab
    # ------------------------------------------------------------------
    def _build_mrz_tab(self):
        f = self._get_tab("mrz")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)

        ctrl = ctk.CTkFrame(f, corner_radius=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl, text="MRZ (两行):",
                     font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=0, padx=(12, 4), pady=8)
        ctk.CTkButton(ctrl, text="📋 解码",
                      fg_color=("#3B82F6", "#1E40AF"),
                      hover_color=("#2563EB", "#1E3A5F"),
                      command=self._mrz_decode_run) \
            .grid(row=0, column=2, padx=(4, 12), pady=8, sticky="e")

        input_frame = ctk.CTkFrame(f, corner_radius=8)
        input_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        input_frame.grid_columnconfigure(0, weight=1)
        self.mrz_input = ctk.CTkTextbox(input_frame, height=70,
                                        font=("Menlo", 12))
        self.mrz_input.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self.mrz_input.insert("1.0",
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10")

        output_frame = ctk.CTkFrame(f, corner_radius=8)
        output_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)
        self.mrz_output = tk.Text(output_frame, wrap="word", relief="flat",
                                  font=("Menlo", 11), padx=8, pady=6)
        self.mrz_output.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _mrz_decode_run(self):
        text = self.mrz_input.get("1.0", "end").strip()
        if not text:
            return
        try:
            r = subprocess.run([str(MRZ_TOOL), "decode", text],
                               capture_output=True, text=True, timeout=10)
            self.mrz_output.delete("1.0", "end")
            self.mrz_output.insert("end", r.stdout)
            if r.stderr:
                self.mrz_output.insert("end", f"\n[stderr]\n{r.stderr}\n")
        except Exception as e:
            self.mrz_output.delete("1.0", "end")
            self.mrz_output.insert("end", f"错误: {e}\n")

    # ------------------------------------------------------------------
    # 防伪验证 tab
    # ------------------------------------------------------------------
    def _build_ac_tab(self):
        f = self._get_tab("ac")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)

        ctrl = ctk.CTkFrame(f, corner_radius=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl, text="MRZ:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=0, padx=(12, 4), pady=8)
        self.ac_mrz_var = tk.StringVar()
        ctk.CTkEntry(ctrl, textvariable=self.ac_mrz_var, width=300) \
            .grid(row=0, column=1, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctrl, text="样本:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=2, padx=(12, 4), pady=8)
        self.ac_sample_var = tk.StringVar()
        ctk.CTkEntry(ctrl, textvariable=self.ac_sample_var, width=200) \
            .grid(row=0, column=3, padx=4, pady=8)
        ctk.CTkButton(ctrl, text="浏览...",
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=self._ac_pick_sample) \
            .grid(row=0, column=4, padx=4, pady=8)
        ctk.CTkButton(ctrl, text="🛡 验证",
                      fg_color=("#3B82F6", "#1E40AF"),
                      hover_color=("#2563EB", "#1E3A5F"),
                      command=self._ac_run) \
            .grid(row=0, column=5, padx=(4, 12), pady=8, sticky="e")

        output_frame = ctk.CTkFrame(f, corner_radius=8)
        output_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)
        self.ac_output = tk.Text(output_frame, wrap="word", relief="flat",
                                 font=("Menlo", 11), padx=8, pady=6)
        self.ac_output.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _ac_pick_sample(self):
        p = filedialog.askopenfilename(
            title="选择样本图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if p:
            self.ac_sample_var.set(p)

    def _ac_run(self, _text=None):
        mrz = self.ac_mrz_var.get()
        sample = self.ac_sample_var.get()
        if not mrz:
            messagebox.showinfo("提示", "请输入 MRZ")
            return
        try:
            args = [str(AC_TOOL), mrz]
            if sample and Path(sample).exists():
                args.append(sample)
            r = subprocess.run(args, capture_output=True, text=True,
                               timeout=15)
            self.ac_output.delete("1.0", "end")
            self.ac_output.insert("end", r.stdout)
            if r.stderr:
                self.ac_output.insert("end", f"\n[stderr]\n{r.stderr}\n")
        except Exception as e:
            self.ac_output.delete("1.0", "end")
            self.ac_output.insert("end", f"错误: {e}\n")

    # ------------------------------------------------------------------
    # 照片比对 tab
    # ------------------------------------------------------------------
    def _build_face_tab(self):
        f = self._get_tab("face")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)

        ctrl = ctk.CTkFrame(f, corner_radius=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl, text="图片 A:", font=ctk.CTkFont(size=12)) \
            .grid(row=0, column=0, padx=(12, 4), pady=8)
        self.face_a_var = tk.StringVar()
        ctk.CTkEntry(ctrl, textvariable=self.face_a_var, width=250) \
            .grid(row=0, column=1, padx=4, pady=8, sticky="w")
        ctk.CTkButton(ctrl, text="浏览...",
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=lambda: self._face_pick("a")) \
            .grid(row=0, column=2, padx=4, pady=8)

        ctk.CTkLabel(ctrl, text="图片 B:", font=ctk.CTkFont(size=12)) \
            .grid(row=1, column=0, padx=(12, 4), pady=8)
        self.face_b_var = tk.StringVar()
        ctk.CTkEntry(ctrl, textvariable=self.face_b_var, width=250) \
            .grid(row=1, column=1, padx=4, pady=8, sticky="w")
        ctk.CTkButton(ctrl, text="浏览...",
                      fg_color="transparent", border_width=1,
                      border_color=("gray60", "gray40"),
                      command=lambda: self._face_pick("b")) \
            .grid(row=1, column=2, padx=4, pady=8)
        ctk.CTkButton(ctrl, text="👤 比对",
                      fg_color=("#3B82F6", "#1E40AF"),
                      hover_color=("#2563EB", "#1E3A5F"),
                      command=self._face_run) \
            .grid(row=1, column=3, padx=(4, 12), pady=8, sticky="e")

        body = ctk.CTkFrame(f, corner_radius=8)
        body.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        self.face_score_chip = StatusPill(body, text="IDLE")
        self.face_score_chip.grid(row=0, column=0, padx=12, pady=8, sticky="w")

        self.face_output = tk.Text(body, wrap="word", relief="flat",
                                   font=("Menlo", 11), padx=8, pady=6)
        self.face_output.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

    def _face_pick(self, which):
        p = filedialog.askopenfilename(
            title="选择照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if p:
            if which == "a":
                self.face_a_var.set(p)
            else:
                self.face_b_var.set(p)

    def _face_run(self):
        a = self.face_a_var.get()
        b = self.face_b_var.get()
        if not a or not b:
            messagebox.showinfo("提示", "请选择两张照片")
            return
        try:
            r = subprocess.run([str(FACE_TOOL), a, b],
                               capture_output=True, text=True, timeout=20)
            self.face_output.delete("1.0", "end")
            self.face_output.insert("end", r.stdout)
            if r.stderr:
                self.face_output.insert("end", f"\n[stderr]\n{r.stderr}\n")
            for ln in r.stdout.splitlines():
                if ln.startswith("score"):
                    tail = ln.split(":", 1)[1].strip()
                    digits = "".join(c for c in tail.split("/")[0]
                                     if c.isdigit())
                    score = int(digits) if digits else 0
                    kind, label = self._face_grade(score)
                    self.face_score_chip.set_status(kind)
                    break
        except Exception as e:
            self.face_output.delete("1.0", "end")
            self.face_output.insert("end", f"错误: {e}\n")

    @staticmethod
    def _face_grade(score):
        if score >= 80:
            return "PASS", "通过"
        if score >= 60:
            return "SUSPECT", "可疑"
        return "FAIL", "不通过"

    # ------------------------------------------------------------------
    # appearance / theme
    # ------------------------------------------------------------------
    def _toggle_appearance(self):
        self._appearance = "dark" if self._appearance == "light" else "light"
        ctk.set_appearance_mode(self._appearance)
        _save_theme_pref(self._appearance)
        self._apply_appearance()

    def _apply_appearance(self):
        dark = ctk.get_appearance_mode() == "Dark"
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        if dark:
            bg = "#0a0e14"
            field = "#111827"
            fg = "#dce3ec"
            head = "#16202c"
            border = "#21262D"
            sel_bg = "#1F6FEB"
            sel_fg = "#ffffff"
        else:
            bg = "#f4f6fa"
            field = "#ffffff"
            fg = "#1a2433"
            head = "#e9edf3"
            border = "#d0d5dd"
            sel_bg = "#DBEAFE"
            sel_fg = "#1E40AF"
        try:
            style.configure("Data.Treeview", background=field,
                            fieldbackground=field, foreground=fg,
                            bordercolor=border, rowheight=26)
            style.map("Data.Treeview",
                      background=[("selected", sel_bg)],
                      foreground=[("selected", sel_fg)])
            style.configure("Data.Treeview.Heading", background=head,
                            foreground=fg, padding=(6, 4), relief="flat")
            style.configure(".", background=bg, foreground=fg)
        except tk.TclError:
            pass
        self._apply_treeview_tags()
        for key, label, desc in TAB_SPECS:
            if key == self._current_tab:
                desc_text = desc
        idx = [k for k, _, _ in TAB_SPECS].index(self._current_tab)
        _, lbl, desc = TAB_SPECS[idx]
        self._breadcrumb.configure(text=f"{lbl}  ›  {desc}")


def main():
    app = CtkPassportGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
