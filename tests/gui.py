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
import os, subprocess, sys, threading, json, tkinter as tk
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

SAMPLE_MRZ = (
    "P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO6908061F9406236ZE184226B<<<<<18\n"
)


def _load_ppm_as_tkimage(path: Path, master=None,
                          max_w: int = 480, max_h: int = 160):
    """Read a PPM (P6) image and produce a Tk PhotoImage.

    Tk PhotoImage natively understands P6 PPM via `file=`, so we just
    hand it the path. If the path is too large for the preview pane,
    we shrink it with the `subsample` method.

    Returns (img, (w, h)) or (None, (0, 0)) on failure.
    """
    if master is None:
        master = tk._default_root
    try:
        img = tk.PhotoImage(master=master, file=str(path))
    except tk.TclError:
        return None, (0, 0)
    # Subsample if larger than the preview pane.
    sx = max(1, img.width() // max_w)
    sy = max(1, img.height() // max_h)
    factor = max(sx, sy)
    if factor > 1:
        img = img.subsample(factor, factor)
    return img, (img.width(), img.height())


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
        bar = ttk.Label(self, textvariable=self.status, anchor="w",
                        relief="sunken", padding=4)
        bar.pack(fill="x", side="bottom")
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
        top = ttk.Frame(f); top.pack(fill="x", padx=4, pady=4)
        ttk.Label(top, text="MRZ OCR 识别精度 / 速度对比",
                  font=("TkDefaultFont", 11, "bold")).pack(side="left")
        # Method selection: individual radio for each method (caller
        # can run one or both).
        self.ocr_methods = tk.StringVar(value="traditional,cnn")
        m_frame = ttk.Frame(top); m_frame.pack(side="left", padx=8)
        ttk.Label(m_frame, text="方案:").pack(side="left")
        for label, key in [("传统模板", "traditional"),
                            ("CNN",       "cnn")]:
            ttk.Radiobutton(m_frame, text=label, variable=self.ocr_methods,
                            value=key).pack(side="left", padx=2)
        self.ocr_method_both = tk.BooleanVar(value=True)
        ttk.Checkbutton(m_frame, text="同时跑两个方案",
                        variable=self.ocr_method_both,
                        command=self._ocr_method_toggle).pack(side="left", padx=4)
        ttk.Button(top, text="运行", command=self._ocr_run).pack(side="right", padx=4)

        sel = ttk.Frame(f); sel.pack(fill="x", padx=4)
        ttk.Button(sel, text="全选", command=lambda: self._ocr_select_all(True)).pack(side="left", padx=2)
        ttk.Button(sel, text="反选", command=lambda: self._ocr_select_all(False)).pack(side="left", padx=2)
        ttk.Button(sel, text="清空选择", command=self._ocr_clear_sel).pack(side="left", padx=2)
        self.ocr_filter_var = tk.StringVar(value="all")
        ttk.Label(sel, text="过滤:").pack(side="left", padx=4)
        for label, key in [("全部", "all"), ("干净", "clean"), ("带噪", "noisy")]:
            ttk.Radiobutton(sel, text=label, variable=self.ocr_filter_var,
                            value=key, command=self._ocr_apply_filter).pack(side="left", padx=2)
        self.ocr_progress_var = tk.StringVar(value="")
        ttk.Label(sel, textvariable=self.ocr_progress_var).pack(side="right", padx=4)

        # ---- Body: 3-panel paned (case list | summary+detail | preview) ----
        body = ttk.PanedWindow(f, orient="horizontal")
        body.pack(fill="both", expand=True, padx=4, pady=4)

        # Left = case list. Width matches the summary table (~600 px).
        left = ttk.Frame(body); body.add(left, weight=1)
        ttk.Label(left, text="测试案例列表（点击查看原图）").pack(anchor="w")
        self.ocr_case_list = ttk.Treeview(left,
            columns=("id", "scale", "noise", "skew"),
            show="headings", selectmode="extended", height=22)
        # Wider columns so total width ≈ summary table width (~540 px)
        for c, w in [("id", 250), ("scale", 80), ("noise", 80), ("skew", 80)]:
            self.ocr_case_list.heading(c, text=c)
            self.ocr_case_list.column(c, width=w, anchor="w")
        ysb = ttk.Scrollbar(left, orient="vertical",
                            command=self.ocr_case_list.yview)
        self.ocr_case_list.configure(yscrollcommand=ysb.set)
        self.ocr_case_list.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.ocr_case_list.bind("<<TreeviewSelect>>", self._ocr_on_select)
        # Also bind double-click on the iid (row body, not heading)
        self.ocr_case_list.bind("<Double-Button-1>", self._ocr_on_select)

        # Right = summary + detail (split vertically).
        right = ttk.Frame(body); body.add(right, weight=2)
        ttk.Label(right, text="汇总").pack(anchor="w")
        self.ocr_summary = ttk.Treeview(right,
            columns=("method", "n", "ok", "okp", "ms_avg",
                      "l1_acc", "l2_acc", "full_match"),
            show="headings", height=3)
        for c, w in [("method", 100), ("n", 50), ("ok", 50), ("okp", 60),
                     ("ms_avg", 80), ("l1_acc", 80), ("l2_acc", 80),
                     ("full_match", 100)]:
            self.ocr_summary.heading(c, text=c)
            self.ocr_summary.column(c, width=w, anchor="w")
        self.ocr_summary.pack(fill="x")
        ttk.Label(right, text="详细结果（按案例 × 方案）").pack(anchor="w", pady=(8, 0))
        self.ocr_detail = ttk.Treeview(right,
            columns=("id", "method", "ok", "ms",
                      "l1_ok", "l1_total",
                      "l2_ok", "l2_total",
                      "gt1", "pred1", "s1",
                      "gt2", "pred2", "s2"),
            show="headings", height=14)
        widths = {"id": 100, "method": 90, "ok": 50, "ms": 70,
                  "l1_ok": 50, "l1_total": 50,
                  "l2_ok": 50, "l2_total": 50,
                  "gt1": 220, "pred1": 220, "s1": 220,
                  "gt2": 220, "pred2": 220, "s2": 220}
        for c, w in widths.items():
            self.ocr_detail.heading(c, text=c)
            self.ocr_detail.column(c, width=w, anchor="w")
        ysb2 = ttk.Scrollbar(right, orient="vertical",
                             command=self.ocr_detail.yview)
        xsb2 = ttk.Scrollbar(right, orient="horizontal",
                             command=self.ocr_detail.xview)
        self.ocr_detail.configure(yscrollcommand=ysb2.set,
                                  xscrollcommand=xsb2.set)
        self.ocr_detail.pack(side="top", fill="both", expand=True)
        ysb2.pack(side="right", fill="y")
        xsb2.pack(side="bottom", fill="x")

        # Far right = image preview (click a row to populate).
        preview = ttk.Frame(body); body.add(preview, weight=1)
        ttk.Label(preview, text="案例预览").pack(anchor="w")
        self.ocr_preview_meta = tk.StringVar(value="(选中的案例图片)")
        ttk.Label(preview, textvariable=self.ocr_preview_meta,
                  font=("TkDefaultFont", 9)).pack(anchor="w")
        self.ocr_preview_label = ttk.Label(preview, background="#222")
        self.ocr_preview_label.pack(fill="both", expand=True)
        self.ocr_preview_text = tk.Text(preview, height=10, width=40,
                                         font=("Menlo", 9))
        self.ocr_preview_text.pack(fill="x")

        # Load corpus.
        corpus_path = _Path(CORPUS_JSON)
        if corpus_path.exists():
            data = json.loads(corpus_path.read_text())
            for rec in data["records"]:
                tag = "noisy" if rec["noise"] > 0 else "clean"
                self.ocr_case_list.insert("", "end", iid=rec["id"],
                    values=(rec["id"], rec["scale"], rec["noise"], rec["skew"]),
                    tags=(tag,))
            self._ocr_corpus = data
        else:
            self._ocr_corpus = {"records": []}

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

    def _ocr_on_select(self, _evt=None):
        from pathlib import Path as _P
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        sel = self.ocr_case_list.selection()
        if not sel:
            return
        cid = sel[0]
        rec = next((r for r in self._ocr_corpus["records"]
                     if r["id"] == cid), None)
        if not rec:
            return
        corpus_dir = _P(CORPUS_JSON).parent
        img_path = corpus_dir / rec["image"]
        self.ocr_preview_meta.set(
            f"id={cid}  scale={rec['scale']}  noise={rec['noise']}  "
            f"skew={rec['skew']}\n路径={rec['image']}")
        # Render image.
        if img_path.exists():
            img, size = _load_ppm_as_tkimage(img_path, master=self)
            if img is not None:
                # Keep a strong ref keyed by case id, then assign so
                # Tk stores a reference of its own before configure().
                self._preview_imgs[cid] = img
                self.ocr_preview_label.configure(image=img, text="")
            else:
                self.ocr_preview_label.configure(image="",
                    text=f"(无法预览 {img_path.name})")
        # Show GT text and any cached predictions.
        text = (f"GT line1: {rec['line1']}\n"
                f"GT line2: {rec['line2']}\n")
        # Look up last-run results if any.
        last = getattr(self, "_ocr_last", {}).get(cid)
        if last:
            for m, r in last.items():
                text += (f"\n[{m}] line1={r.get('line1','')[:44]}\n"
                         f"     line2={r.get('line2','')[:44]}\n")
        self.ocr_preview_text.delete("1.0", "end")
        self.ocr_preview_text.insert("1.0", text)

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
        self.ocr_progress_var.set(f"准备运行 {len(selected)} 个案例 ...")
        def worker():
            from pathlib import Path as _P
            runner = _P(__file__).resolve().parent / "ocr_bench_runner.py"
            argv = ["python3", str(runner), ",".join(methods), *selected]
            try:
                proc = subprocess.run(argv, text=True, capture_output=True,
                                      timeout=900)
                payload = json.loads(proc.stdout)
            except Exception as e:
                self.after(0, lambda e=e: self.ocr_progress_var.set(f"运行失败: {e}"))
                return
            self.after(0, lambda: self._ocr_populate_results(payload))
        threading.Thread(target=worker, daemon=True).start()

    def _ocr_populate_results(self, payload):
        methods = payload["methods"]
        agg = payload["agg"]
        # Cache per-case results for the preview.
        self._ocr_last = {}
        for rec in payload["records"]:
            d = {}
            for m in methods:
                pm = rec["per_method"][m]
                d[m] = pm["raw"]
            self._ocr_last[rec["id"]] = d
        for m in methods:
            a = agg[m]
            ms_avg = a["ms_total"] / a["n"] if a["n"] else 0
            ok_pct = 100.0 * a["ok"] / a["n"] if a["n"] else 0
            l1 = 100.0 * a["l1_correct"] / a["l1_total"] if a["l1_total"] else 0
            l2 = 100.0 * a["l2_correct"] / a["l2_total"] if a["l2_total"] else 0
            label = {"traditional": "传统模板", "cnn": "CNN"}.get(m, m)
            self.ocr_summary.insert("", "end", values=(
                label, a["n"], a["ok"], f"{ok_pct:.1f}%",
                f"{ms_avg:.2f}",
                f"{l1:.2f}%", f"{l2:.2f}%", a["full_match"]))
        for rec in payload["records"]:
            for m in methods:
                pm = rec["per_method"][m]
                r = pm["raw"]
                self.ocr_detail.insert("", "end", values=(
                    rec["id"],
                    {"traditional": "传统", "cnn": "CNN"}.get(m, m),
                    "OK" if r["ok"] else "FAIL",
                    f"{r['ms']:.2f}",
                    pm["l1_correct"], pm["l1_total"],
                    pm["l2_correct"], pm["l2_total"],
                    rec["gt1"], r["line1"], pm["l1_status"],
                    rec["gt2"], r["line2"], pm["l2_status"]))
        self.ocr_progress_var.set(
            f"完成 {payload['total_cases']} 个案例, 方案={','.join(methods)}")

    # ---------------- NFC tab ----------------
    def _build_nfc_tab(self):
        f = self.tab_nfc
        top = ttk.Frame(f); top.pack(fill="x", padx=4, pady=4)
        ttk.Label(top, text="NFC 读卡流程（Mock R-APDU）",
                  font=("TkDefaultFont", 11, "bold")).pack(side="left")

        # Script picker row.
        row = ttk.Frame(f); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Mock 脚本:").pack(side="left")
        # Default scripts in 4_nfc_reader/data
        self.nfc_script_var = tk.StringVar(
            value=str(ROOT / "4_nfc_reader" / "data" / "script_happy.json"))
        ttk.Entry(row, textvariable=self.nfc_script_var, width=80).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row, text="浏览…",
                   command=self._nfc_pick_script).pack(side="left")
        ttk.Button(row, text="运行", command=self._nfc_run).pack(side="left", padx=2)
        self.nfc_status_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.nfc_status_var,
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=4)

        # APDU trace pane.
        trace_frame = ttk.LabelFrame(f, text="APDU 跟踪 (Command → Response)")
        trace_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.nfc_trace = ttk.Treeview(trace_frame,
            columns=("i", "c_apdu", "r_apdu", "sw", "ok"),
            show="headings", height=20)
        for c, w in [("i", 40), ("c_apdu", 360),
                     ("r_apdu", 360), ("sw", 60), ("ok", 60)]:
            self.nfc_trace.heading(c, text=c)
            self.nfc_trace.column(c, width=w, anchor="w")
        ysb = ttk.Scrollbar(trace_frame, orient="vertical",
                            command=self.nfc_trace.yview)
        self.nfc_trace.configure(yscrollcommand=ysb.set)
        self.nfc_trace.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

        # DG / readback panel.
        dg_frame = ttk.LabelFrame(f, text="读取结果 / Data Groups")
        dg_frame.pack(fill="x", padx=4, pady=4)
        self.nfc_dg_text = tk.Text(dg_frame, height=8, font=("Menlo", 10))
        self.nfc_dg_text.pack(fill="x")

    def _nfc_pick_script(self):
        path = filedialog.askopenfilename(
            initialdir=str(ROOT / "4_nfc_reader" / "data"),
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.nfc_script_var.set(path)

    def _nfc_run(self):
        script = self.nfc_script_var.get().strip()
        if not script or not Path(script).exists():
            messagebox.showerror("错误", f"找不到 mock 脚本:\n{script}")
            return
        for r in self.nfc_trace.get_children():
            self.nfc_trace.delete(r)
        self.nfc_dg_text.delete("1.0", "end")
        self.nfc_status_var.set("运行中…")
        def worker():
            try:
                # Add /v verbose to surface per-APDU details.
                proc = subprocess.run(
                    [str(NFC_TOOL), script],
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
        """Build an APDU-like trace from the nfc_tool key=value output.

        nfc_tool emits a handful of key:value lines describing the
        end-to-end BAC happy path (select, challenge, mutual auth,
        select DG1, read DG1, etc). We synthesize an APDU-style trace
        table on top of those results.
        """
        ok = (rc == 0)
        self.nfc_status_var.set(
            f"\u2713 全部成功（{rc}）" if ok else f"\u2717 异常退出 (rc={rc})")
        kv = {}
        for ln in stdout.splitlines():
            ln = ln.strip()
            if ":" not in ln:
                continue
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
        steps = [
            ("1", "SELECT (MF)", "ATR / \u5361\u7c7b\u578b", "9000"),
            ("2", "GET CHALLENGE", "\u968f\u673a 8 \u5b57\u8282", "9000"),
            ("3", "MUTUAL AUTH (BAC)",
                "\u6d3e\u751f Kenc/Kmac \u4f1a\u8bdd\u5bc6\u94a5", "9000"),
            ("4", "SELECT EF.DG1", kv.get("result.detail",
                "\u627e\u5230 DG1 \u6587\u4ef6"), "9000"),
            ("5", "READ BINARY (DG1)",
                (kv.get("result.dg1_mrz", "")[:60]
                 ).replace("\n", " / "), "9000"),
            ("6", "SELECT EF.DG15/SOD",
                "SOD = " + kv.get("result.sod_head", ""), "9000"),
            ("7", "READ BINARY (SOD)",
                kv.get("result.sod_head", "") + "\u2026", "9000"),
            ("8", "CLOSE / \u91ca\u653e\u4fe1\u9053",
                f"\u672a\u5339\u914d mock: {kv.get('mock.unexpected','0')}",
                "9000"),
        ]
        for i, capdu, rapdu, sw in steps:
            self.nfc_trace.insert("", "end", values=(
                i, capdu, rapdu, sw,
                "OK" if ok else "FAIL"))
        dg_text = stdout + (("\n[stderr]\n" + stderr) if stderr else "")
        self.nfc_dg_text.delete("1.0", "end")
        self.nfc_dg_text.insert("1.0", dg_text)

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
        top = ttk.Frame(f); top.pack(fill="x", padx=4)
        ttk.Label(top, text="照片A:").pack(side="left")
        self.face_a = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.face_a, width=50).pack(side="left", padx=2, fill="x", expand=True)
        ttk.Button(top, text="浏览…", command=lambda: self._face_pick("a")).pack(side="left")
        row2 = ttk.Frame(f); row2.pack(fill="x", padx=4, pady=2)
        ttk.Label(row2, text="照片B:").pack(side="left")
        self.face_b = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.face_b, width=50).pack(side="left", padx=2, fill="x", expand=True)
        ttk.Button(row2, text="浏览…", command=lambda: self._face_pick("b")).pack(side="left")
        ttk.Button(row2, text="比对", command=self._face_run).pack(side="left", padx=4)
        self.face_out = tk.Text(f, height=12, font=("Menlo", 11))
        self.face_out.pack(fill="both", expand=True, padx=4, pady=4)

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
