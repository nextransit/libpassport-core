#!/usr/bin/env python3
"""Resident engine pool for the MRZ photo benchmark.

Why this exists
---------------
Every engine used to run as a fresh process per photo.  Measured on
this machine (2026-10-29):

    mrz_ocr_cnn_tool one-shot      8 ms   (6 ms of it is process start)
    mrz_ocr_cnn_tool --serve       4 ms   (model already loaded)
    12_crnn/mrz_crnn_tool.py      137 ms   (79 ms is onnxruntime import)

The official path runs seven engines per photo, so the process/model
start-up cost was paid seven times per photo and dominated the
measured throughput.  This module keeps the C tools resident and
loads the CRNN ONNX model in-process, so a benchmark measures
recognition time rather than start-up time.

The C tools speak a line protocol on stdin/stdout: one image path per
line in, one result block out.  The CRNN ONNX model is imported
directly and reused.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ResidentTool:
    """One resident C tool process, driven over stdin/stdout."""

    def __init__(self, argv, env):
        self.argv = [str(a) for a in argv]
        self.env = env
        self.proc = None
        self.lock = threading.Lock()

    def _start(self):
        self.proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=self.env,
        )

    def run(self, image_path):
        """Recognise one image; returns (stdout_text, error_text)."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            try:
                self.proc.stdin.write(f"{image_path}\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                self._start()
                self.proc.stdin.write(f"{image_path}\n")
                self.proc.stdin.flush()
            lines = []
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                lines.append(line)
                if line.startswith("result.done"):
                    break
            return "".join(lines), ""

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


_POOL_LOCK = threading.Lock()
_POOL = {}


def _tool_env(**extra):
    env = dict(os.environ)
    env.setdefault("MRZ_OCR_LINE_MODELS", "1")
    env.update({k: v for k, v in extra.items() if v is not None})
    return env


def get_cnn_tool(fuse=False):
    """Resident CNN/fuse tool.  `fuse` selects the fused head."""
    key = "fuse" if fuse else "cnn"
    with _POOL_LOCK:
        tool = _POOL.get(key)
        if tool is None:
            argv = [ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool", "--serve"]
            env = _tool_env(MRZ_OCR_CRNN="0", MRZ_OCR_FUSE="1" if fuse else None)
            tool = ResidentTool(argv, env)
            _POOL[key] = tool
        return tool


def get_traditional_tool():
    with _POOL_LOCK:
        tool = _POOL.get("traditional")
        if tool is None:
            argv = [ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool", "--serve"]
            tool = ResidentTool(argv, _tool_env())
            _POOL["traditional"] = tool
        return tool


_CRNN_MODULE = None
_CRNN_LOCK = threading.Lock()


def get_crnn_module():
    """Import the CRNN ONNX reader once and reuse its session."""
    global _CRNN_MODULE
    with _CRNN_LOCK:
        if _CRNN_MODULE is None:
            import importlib.util
            path = ROOT / "12_crnn" / "mrz_crnn_tool.py"
            spec = importlib.util.spec_from_file_location("mrz_crnn_tool", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.session()          # force the ONNX session to load now
            _CRNN_MODULE = module
        return _CRNN_MODULE


_PADDLE_MODULE = None
_PADDLE_LOCK = threading.Lock()


def get_paddle_module():
    """Import the Paddle recogniser in-process and keep it loaded.

    Running paddle_v6.py as a subprocess paid the interpreter start-up
    and the onnxruntime import on every single photo; importing it once
    leaves only the recognition cost.
    """
    global _PADDLE_MODULE
    with _PADDLE_LOCK:
        if _PADDLE_MODULE is None:
            import importlib.util
            path = ROOT / "6_1_PaddleOCR" / "paddle_v6.py"
            pkg_dir = str(path.parent)
            if pkg_dir not in sys.path:
                sys.path.insert(0, pkg_dir)
            spec = importlib.util.spec_from_file_location("paddle_v6", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module._get_rec()          # force the ONNX model to load now
            _PADDLE_MODULE = module
        return _PADDLE_MODULE


# A small synthetic band used to force one real inference through every
# engine.  Loading the model is not enough: the first ONNX run and the
# first Tesseract call each pay their own one-off initialisation, which
# otherwise lands on the first real photo and inflates its time.
_WARMUP_BAND = None


def _warmup_band():
    """Write a tiny two-row MRZ-like band to a temp file."""
    global _WARMUP_BAND
    if _WARMUP_BAND is not None and _WARMUP_BAND.exists():
        return _WARMUP_BAND
    import tempfile
    import numpy as np
    from PIL import Image
    band = np.full((40, 960, 3), 255, dtype=np.uint8)
    band[8:20, 20:900] = 0
    band[24:36, 20:900] = 0
    path = Path(tempfile.gettempdir()) / "_mrz_pool_warmup.ppm"
    Image.fromarray(band).save(str(path), format="PPM")
    _WARMUP_BAND = path
    return path


def warmup_all(gui_module=None):
    """Load every model AND run one inference, so later timings are
    recognition-only."""
    started = time.perf_counter()
    get_traditional_tool()
    get_cnn_tool(fuse=False)
    get_cnn_tool(fuse=True)
    crnn = get_crnn_module()
    get_paddle_module()
    band = _warmup_band()
    try:
        get_traditional_tool().run(str(band))
        get_cnn_tool(fuse=False).run(str(band))
        get_cnn_tool(fuse=True).run(str(band))
        crnn.read_band(str(band))
    except Exception:
        # Warm-up failures are not fatal: the engines report their own
        # errors during the real run.
        pass
    warmup_gui_engines(gui_module)
    return (time.perf_counter() - started) * 1000.0


def warmup_gui_engines(gui_module=None):
    """Run one real recognition through the GUI-side engines.

    Paddle and Indra live in tests/gui.py and load their models on
    first use.  Their first *inference* also carries a one-off cost
    (ONNX graph setup, Tesseract's own init) that would otherwise land
    on the first real photo and inflate its measured time.

    Pass the already-imported gui module so the warm-up does not import
    it a second time.
    """
    try:
        import numpy as np
        if gui_module is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "mrz_gui_for_warmup", ROOT / "tests" / "gui.py")
            gui_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gui_module)
        band = _warmup_band()
        gui_module._rst_run_paddle_inprocess(str(band))
        gui_module._rst_tesseract_candidates(
            np.zeros((40, 960, 3), dtype=np.uint8), [(8, 20), (24, 36)], 1.0)
    except Exception:
        # Warm-up failures are not fatal: the engines report their own
        # errors during the real run.
        pass


def close_all():
    with _POOL_LOCK:
        for tool in _POOL.values():
            tool.close()
        _POOL.clear()
