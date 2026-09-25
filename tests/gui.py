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
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

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


class PassportGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("护照机测试机 - Passport Test Bench")
        self.geometry("1280x780")
        self._set_window_icon()
        self._build_menu()
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        # Tab order: OCR (1) -> NFC (2) -> MRZ (3) -> AC (4) -> face (5).
        self.tab_ocr  = ttk.Frame(nb)
        self.tab_nfc  = ttk.Frame(nb)
        self.tab_mrz  = ttk.Frame(nb)
        self.tab_ac   = ttk.Frame(nb)
        self.tab_face = ttk.Frame(nb)
        nb.add(self.tab_ocr,  text="MRZ OCR 测试")
        nb.add(self.tab_nfc,  text="NFC 读卡")
        nb.add(self.tab_mrz,  text="MRZ 解码")
        nb.add(self.tab_ac,   text="防伪验证")
        nb.add(self.tab_face, text="照片比对")
        self._build_ocr_tab()
        self._build_nfc_tab()
        self._build_mrz_tab()
        self._build_ac_tab()
        self._build_face_tab()
        self.status = tk.StringVar(value=self._status_text())
        sbar = ttk.Frame(self)
        sbar.pack(fill="x", side="bottom")
        self.ocr_status_var = tk.StringVar(value="就绪")
        ttk.Label(sbar, textvariable=self.ocr_status_var, anchor="w",
                  padding=4).pack(side="left")
        self.ocr_progress = ttk.Progressbar(sbar, length=180,
                                            mode="determinate", maximum=100)
        self.ocr_progress.pack(side="left", padx=8, pady=2)
        ttk.Label(sbar, textvariable=self.status, anchor="e",
                  relief="sunken", padding=4).pack(side="right", fill="x")
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

        # ---- Passport image region locator (photo / data / MRZ) ----
        self._build_loc_panel(f)

        # ---- Control card: methods | filter | search | primary CTA ----
        ctrl = ttk.LabelFrame(f, text="控制面板")
        ctrl.pack(fill="x", padx=4, pady=(4, 0))
        row1 = ttk.Frame(ctrl); row1.pack(fill="x", padx=8, pady=(6, 2))
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
        try:
            style.configure("Accent.TButton", background="#0b5cad",
                            foreground="white", padding=(14, 4),
                            font=("TkDefaultFont", 10, "bold"))
            style.map("Accent.TButton",
                      background=[("pressed", "#0a4a94"), ("active", "#2b7ed6")],
                      foreground=[("disabled", "#9cc3ee")])
        except Exception:
            pass
        self.ocr_run_btn = ttk.Button(
            row1, text="▶ 开始测试", command=self._ocr_run,
            style="Accent.TButton", cursor="hand2")
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
        search = ttk.Entry(row2, textvariable=self.ocr_search_var, width=30)
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
            show="headings", selectmode="extended", height=22)
        for c, w in [("id", 240), ("scale", 70), ("noise", 70), ("skew", 70)]:
            self.ocr_case_list.heading(c, text=c)
            self.ocr_case_list.column(c, width=w, anchor="w")
        ysb = ttk.Scrollbar(left, orient="vertical",
                            command=self.ocr_case_list.yview)
        self.ocr_case_list.configure(yscrollcommand=ysb.set)
        self.ocr_case_list.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.ocr_case_list.bind("<<TreeviewSelect>>", self._ocr_on_case_select)
        self.ocr_case_list.bind("<Double-Button-1>", self._ocr_on_case_select)

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
        # Theme-proof Treeview: light field with dark text on both system
        # modes, so the metric rows never render as white-on-white.
        try:
            st = ttk.Style()
            st.configure("OCR.Treeview", background="#f7f9fc",
                         fieldbackground="#f7f9fc", foreground="#111827",
                         bordercolor="#cbd5e1", lightcolor="#cbd5e1",
                         darkcolor="#cbd5e1")
            st.configure("OCR.Treeview.Heading", background="#e9edf3",
                         foreground="#111827", padding=(4, 2))
            self.ocr_summary.configure(style="OCR.Treeview")
        except Exception:
            pass
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
            show="headings", height=14)
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
                                     font=("Menlo", 10), wrap="none")
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
        corpus_path = _Path(CORPUS_JSON)
        self._ocr_case_order = []
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
        else:
            self._ocr_corpus = {"records": []}
        self.after(100, self._ocr_set_sashes)

    def _build_loc_panel(self, parent):
        """Passport image OCR region locator: photo / data / MRZ.
        Built from tests/passport_locator.py (pure PIL+numpy).
        """
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        from passport_locator import locate_regions, annotate  # type: ignore
        self._loc_locator = locate_regions
        self._loc_annotate = annotate

        panel = ttk.LabelFrame(parent, text="护照图片 OCR 区域定位（照片 / 数据区 / MRZ 区）")
        panel.pack(fill="x", padx=4, pady=(4, 0))

        row = ttk.Frame(panel); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="样本:").pack(side="left")
        self.loc_sample = tk.StringVar()
        self.loc_samples = [
            "合成护照样本 (TD3 整页)",
            "corpus MRZ 条带 (img_0001_p0_v0)",
        ]
        cb = ttk.Combobox(row, textvariable=self.loc_sample,
                          values=self.loc_samples, state="readonly", width=30)
        cb.pack(side="left", padx=2)
        cb.current(0)
        self.loc_sample_path = tk.StringVar()
        ttk.Button(row, text="浏览文件…", command=self._loc_pick).pack(side="left", padx=4)
        tk.Button(row, text="▶ 定位区域", command=self._loc_run,
                  bg="#0066cc", fg="white", activebackground="#0055aa",
                  activeforeground="white", disabledforeground="#9cc3ee",
                  relief="flat", cursor="hand2",
                  font=("TkDefaultFont", 10, "bold"), padx=12, pady=3).pack(side="left", padx=4)
        self.loc_status = ttk.Label(row, text="", foreground="#666")
        self.loc_status.pack(side="left", padx=8)

        body = ttk.Frame(panel); body.pack(fill="x", padx=8, pady=(0, 6))
        self.loc_canvas = tk.Canvas(body, width=560, height=300, bg="#222",
                                    highlightthickness=1, highlightbackground="#555")
        self.loc_canvas.pack(side="left")
        self.loc_info = tk.Text(body, width=48, height=15, font=("Menlo", 9),
                                state="disabled", wrap="none",
                                bg="#fbfbfb", relief="solid", bd=1)
        self.loc_info.pack(side="left", padx=(6, 0), fill="both", expand=True)

    def _loc_pick(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.png *.ppm *.bmp *.jpg"), ("All", "*.*")])
        if path:
            self.loc_sample_path.set(path)
            self.loc_sample.set(f"自定义: {os.path.basename(path)}")
            self.loc_status.config(text="已选择文件，点“定位区域”", foreground="#666")

    def _loc_resolve_path(self):
        path = self.loc_sample_path.get()
        if path and os.path.exists(path):
            return path
        key = self.loc_sample.get()
        if key.startswith("corpus"):
            from ocr_bench_runner import CORPUS_JSON  # type: ignore
            return str(Path(CORPUS_JSON).parent / "img_0001_p0_v0.ppm")
        return str(SAMPLE_PASSPORT_PNG)

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
            disp.thumbnail((self.loc_canvas.winfo_width() - 4,
                            self.loc_canvas.winfo_height() - 4))
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

    def _ocr_on_detail_select(self, _evt=None):
        sel = self.ocr_detail.selection()
        if not sel:
            return
        cid, m = self._ocr_detail_map[sel[0]]
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

        # ---- Top: compact control bar (backend + script + run) ----
        top = ttk.Frame(f); top.pack(fill="x", padx=4, pady=2)
        ttk.Label(top, text="后端:").pack(side="left")
        self.nfc_backend_var = tk.StringVar(value="Mock 卡片")
        ttk.Combobox(top, textvariable=self.nfc_backend_var, state="readonly",
                     values=("Mock 卡片", "真实读卡器"), width=10).pack(side="left")
        ttk.Label(top, text="脚本:").pack(side="left", padx=(8, 2))
        self.nfc_script_var = tk.StringVar(
            value=str(ROOT / "4_nfc_reader" / "data" / "script_happy.json"))
        ttk.Entry(top, textvariable=self.nfc_script_var, width=52).pack(
            side="left", fill="x", expand=True)
        ttk.Button(top, text="浏览…",
                   command=self._nfc_pick_script).pack(side="left")
        self.nfc_run_btn = tk.Button(top, text="▶ 开始执行", bg="#0b5cad",
                                     fg="white", relief="flat", cursor="hand2",
                                     font=("TkDefaultFont", 10, "bold"),
                                     command=self._nfc_run)
        self.nfc_run_btn.pack(side="left", padx=(6, 0))

        # ---- MRZ compact row ----
        mrz = ttk.Frame(f); mrz.pack(fill="x", padx=4, pady=1)
        ttk.Label(mrz, text="line1:").pack(side="left")
        self.nfc_mrz1_var = tk.StringVar(value=SAMPLE_MRZ.splitlines()[0])
        ttk.Entry(mrz, textvariable=self.nfc_mrz1_var, width=46,
                  font=("Menlo", 9)).pack(side="left", padx=(2, 6))
        ttk.Label(mrz, text="line2:").pack(side="left")
        self.nfc_mrz2_var = tk.StringVar(value=SAMPLE_MRZ.splitlines()[1])
        ttk.Entry(mrz, textvariable=self.nfc_mrz2_var, width=46,
                  font=("Menlo", 9)).pack(side="left", padx=(2, 6))
        ttk.Button(mrz, text="从 OCR 填充",
                   command=self._nfc_fill_mrz_from_ocr).pack(side="left", padx=2)
        ttk.Button(mrz, text="内置样例",
                   command=self._nfc_fill_mrz_sample).pack(side="left")
        self.nfc_badge_var = tk.StringVar(value="")
        self.nfc_badge = tk.Label(mrz, textvariable=self.nfc_badge_var,
                                  font=("TkDefaultFont", 9, "bold"))
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
        self._nfc_states = []
        for title in ("① MRZ 因子校验", "② 基础密钥派生 Kseed/Kenc/Kmac",
                      "③ 相互认证 (BAC)", "④ Secure Messaging Ready"):
            row = ttk.Frame(lab); row.pack(fill="x", padx=6, pady=1)
            ttk.Label(row, text=title, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            lb = tk.Label(row, textvariable=var,
                          font=("TkDefaultFont", 9, "bold"), fg="#888")
            lb.pack(side="left", padx=6)
            self._nfc_states.append((title, var, lb))

    def _nfc_build_keys_panel(self, parent):
        lab = ttk.LabelFrame(parent, text="会话密钥（点击复制）")
        lab.pack(fill="x", padx=2, pady=2)
        self._nfc_key_vars = {}
        for name in ("Kseed", "Kenc", "Kmac", "KSenc", "KSmac"):
            row = ttk.Frame(lab); row.pack(fill="x", padx=4, pady=1)
            ttk.Label(row, text=name, width=6).pack(side="left")
            var = tk.StringVar(value="—")
            en = tk.Entry(row, textvariable=var, width=34, font=("Menlo", 9),
                          state="readonly", readonlybackground="#f7f7f7")
            en.pack(side="left", padx=2)
            ttk.Button(row, text="复制", width=4,
                       command=lambda n=name, v=var: self._nfc_copy_key(n, v)
                       ).pack(side="left")
            self._nfc_key_vars[name] = var

    def _nfc_build_trace_panel(self, parent):
        insp = ttk.LabelFrame(parent, text="报文透视（Raw / Decrypted / MAC）")
        insp.pack(fill="x", side="bottom", padx=2, pady=2)
        self.nfc_inspect = tk.Text(insp, height=7, font=("Menlo", 9),
                                   wrap="none", bg="#fafafa")
        yi = ttk.Scrollbar(insp, orient="vertical",
                           command=self.nfc_inspect.yview)
        self.nfc_inspect.configure(yscrollcommand=yi.set)
        self.nfc_inspect.pack(side="left", fill="both", expand=True)
        yi.pack(side="right", fill="y")

        lab = ttk.LabelFrame(parent, text="APDU 实时跟踪（点行查看报文透视）")
        lab.pack(fill="both", expand=True, padx=2, pady=2)
        self.nfc_trace = ttk.Treeview(
            lab, columns=("i", "cmd", "hex", "sw", "ms"),
            show="headings")
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
        card = ttk.LabelFrame(parent, text="DG1 持证人信息")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_dg1_vars = {}
        for row, (key, label) in enumerate(
                [("name", "姓名"), ("doc", "证件号"), ("nat", "国籍"),
                 ("dob", "出生日期"), ("exp", "有效期")]):
            ttk.Label(card, text=label + ":").grid(
                row=row, column=0, sticky="e", padx=(6, 2))
            var = tk.StringVar(value="—")
            tk.Label(card, textvariable=var, font=("Menlo", 9, "bold")).grid(
                row=row, column=1, sticky="w", padx=(2, 6))
            self._nfc_dg1_vars[key] = var

    def _nfc_build_sod_card(self, parent):
        card = ttk.LabelFrame(parent, text="SOD 被动认证 (Passive Auth)")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_sod_sha = tk.StringVar(value="—")
        self._nfc_sod_rsa = tk.StringVar(value="—")
        ttk.Label(card, text="DG1 SHA-256:").grid(
            row=0, column=0, sticky="e", padx=(6, 2))
        tk.Label(card, textvariable=self._nfc_sod_sha,
                 font=("Menlo", 9)).grid(row=0, column=1, sticky="w")
        ttk.Label(card, text="RSA 验签:").grid(
            row=1, column=0, sticky="e", padx=(6, 2))
        tk.Label(card, textvariable=self._nfc_sod_rsa,
                 font=("Menlo", 9)).grid(row=1, column=1, sticky="w")

    def _nfc_build_progress_card(self, parent):
        card = ttk.LabelFrame(parent, text="读取进度")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_prog_vars = {}
        for i, ef in enumerate(("EF.COM", "EF.DG1", "EF.DG2", "EF.SOD")):
            r, c = divmod(i, 2)
            var = tk.StringVar(value="—")
            lb = tk.Label(card, textvariable=var,
                          font=("TkDefaultFont", 9, "bold"), fg="#888")
            lb.grid(row=r, column=c * 2, sticky="e", padx=(6, 2))
            ttk.Label(card, text=ef).grid(row=r, column=c * 2 + 1,
                                          sticky="w", padx=(0, 10))
            self._nfc_prog_vars[ef] = (var, lb)

    def _nfc_build_adv_panel(self, parent):
        self._adv_open = tk.BooleanVar(value=False)
        self._adv_frame = ttk.LabelFrame(parent, text="高级密码学参数")
        self._adv_frame.pack(fill="x", padx=2, pady=2)
        head = ttk.Frame(self._adv_frame); head.pack(fill="x")
        ttk.Button(head, text="展开 / 折叠",
                   command=self._nfc_toggle_adv).pack(side="left", padx=4)
        self.nfc_adv_text = tk.Text(self._adv_frame, height=4,
                                    font=("Menlo", 9), wrap="none", bg="#f7f7f7")
        self.nfc_adv_text.pack(fill="x", padx=4, pady=2)
        self.nfc_adv_text.pack_forget()

    def _nfc_toggle_adv(self):
        if self._adv_open.get():
            self.nfc_adv_text.pack_forget()
            self._adv_open.set(False)
        else:
            self.nfc_adv_text.pack(fill="x", padx=4, pady=2)
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
        for _, var, lb in self._nfc_states:
            var.set("…")
            lb.configure(fg="#888")
        for var in self._nfc_dg1_vars.values():
            var.set("—")
        self._nfc_sod_sha.set("—")
        self._nfc_sod_rsa.set("—")
        for _, lb in self._nfc_prog_vars.values():
            lb.configure(fg="#888")
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
        self.nfc_badge_var.set("✓ BAC 认证通过" if ok else "✗ 认证失败")
        self.nfc_badge.configure(fg="#1b5e20" if ok else "#b71c1c")
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
            var.set("✓ Pass" if good else "✗ Fail")
            lb.configure(fg="#1b5e20" if good else "#b71c1c")

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
        for ef, (var, lb) in self._nfc_prog_vars.items():
            got = {"EF.COM": False, "EF.DG1": bool(dg1_lines),
                   "EF.DG2": False, "EF.SOD": sod_ok}[ef]
            var.set("✓" if got else "—")
            lb.configure(fg="#1b5e20" if got else "#888")

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
        self.mrz_text = tk.Text(left, width=52, height=6, font=("Menlo", 12))
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
        self.mrz_out = tk.Text(left, width=80, height=18, font=("Menlo", 11))
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
            ttk.Entry(row, textvariable=var, width=22).pack(side="left", fill="x", expand=True)
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
        self.ac_out = tk.Text(f, font=("Menlo", 11))
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
        ttk.Label(f, text="照片比对", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=4)
        # Built-in reference samples row.
        ref = ttk.Frame(f); ref.pack(fill="x", padx=4, pady=(2, 0))
        ttk.Label(ref, text="内置参考样本:").pack(side="left")
        self.face_ref = tk.StringVar()
        face_ref_cb = ttk.Combobox(ref, textvariable=self.face_ref,
                                   values=list(FACE_SAMPLES.keys()),
                                   state="readonly", width=26)
        face_ref_cb.pack(side="left", padx=2)
        face_ref_cb.current(0)
        ttk.Button(ref, text="填入 → 照片A",
                   command=self._face_load_ref).pack(side="left", padx=4)
        ttk.Label(ref, text="（face_a 与 face_b 已预置，可直接点“比对”）",
                  foreground="#666").pack(side="left", padx=6)
        top = ttk.Frame(f); top.pack(fill="x", padx=4, pady=2)
        ttk.Label(top, text="照片A:").pack(side="left")
        self.face_a = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.face_a, width=46).pack(side="left", padx=2, fill="x", expand=True)
        ttk.Button(top, text="浏览…", command=lambda: self._face_pick("a")).pack(side="left")
        row2 = ttk.Frame(f); row2.pack(fill="x", padx=4, pady=2)
        ttk.Label(row2, text="照片B:").pack(side="left")
        self.face_b = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.face_b, width=46).pack(side="left", padx=2, fill="x", expand=True)
        ttk.Button(row2, text="浏览…", command=lambda: self._face_pick("b")).pack(side="left")
        ttk.Button(row2, text="比对", command=self._face_run).pack(side="left", padx=4)
        self.face_out = tk.Text(f, height=12, font=("Menlo", 11))
        self.face_out.pack(fill="both", expand=True, padx=4, pady=4)
        # Default: face_a vs face_b so the tab is directly testable.
        self.face_a.set(str(FACE_SAMPLES["face_a (合成, 128×128)"]))
        self.face_b.set(str(FACE_SAMPLES["face_b (合成, 128×128)"]))
        self.face_out.insert("1.0", "提示：内置 face_a / face_b 已预置，点击“比对”即可直接测试。\n"
                                    "也可从内置参考样本下拉选择并填入照片A，或浏览本地图片。\n")

    def _face_load_ref(self):
        path = FACE_SAMPLES.get(self.face_ref.get())
        if path:
            self.face_a.set(str(path))

    def _face_pick(self, which):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.ppm *.bmp *.png *.jpg"), ("All", "*.*")])
        if path:
            (self.face_a if which == "a" else self.face_b).set(path)

    def _face_run(self):
        a = self.face_a.get().strip(); b = self.face_b.get().strip()
        if not (a and b):
            messagebox.showerror("错误", "请选择两张照片")
            return
        try:
            r = subprocess.run([str(FACE_TOOL), a, b],
                               text=True, capture_output=True, timeout=20)
            self.face_out.delete("1.0", "end")
            self.face_out.insert("1.0", r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or ""))
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
