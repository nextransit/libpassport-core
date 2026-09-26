# 6_1_PaddleOCR

PaddleOCR-backed MRZ recognizer. **Status: placeholder.**

The GUI exposes this back-end in the OCR tab drop-down but
invoking it currently fails because `paddlepaddle` does not
publish a Python 3.14 wheel for macOS arm64 (as of 2026-09-27).
Until the upstream project ships a wheel, the back-end row in
the OCR method picker is disabled with the message
"PaddleOCR (待 paddlepaddle 支持 Python 3.14)".

When `paddlepaddle` becomes installable on this platform:

1.  `pip install paddlepaddle`
2.  Implement `paddle_ocr_tool.py` next to this README, mirroring
    the stdout contract in 6_2_Tesseract/tesseract_tool.py.
3.  Add the tool path to `tests/gui.py` and `tests/gui_ctk.py`:

    ```python
    PADDLE_TOOL = ROOT / "6_1_PaddleOCR" / "paddle_ocr_tool.py"
    ```
4.  Register the back-end in
    `tests/ocr_bench_runner.METHODS["paddle"] = ("PaddleOCR", PADDLE_TOOL)`.
