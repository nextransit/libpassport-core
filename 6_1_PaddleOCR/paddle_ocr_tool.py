#!/usr/bin/env python3
"""paddle_ocr_tool -- PLACEHOLDER.

Will use PaddleOCR (paddlepaddle) once the runtime is installable on
the target platform. Until then the GUI disables this entry in the
OCR method picker. See ../README.md for details.
"""
import sys
print("paddle_ocr_tool: paddlepaddle unavailable on this platform",
      file=sys.stderr)
sys.exit(2)
