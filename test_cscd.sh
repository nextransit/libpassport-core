#!/bin/bash
export MRZ_OCR_SEGDBG=1
echo "=== Band I ==="
6_mrz_ocr/build/mrz_ocr_cnn_tool 6_mrz_ocr/data/corpus_cscd_photo/2026-9-30-I.ppm 2>&1
echo "=== Band J ==="
6_mrz_ocr/build/mrz_ocr_cnn_tool 6_mrz_ocr/data/corpus_cscd_photo/2026-9-30-J.ppm 2>&1
echo "=== Band K ==="
6_mrz_ocr/build/mrz_ocr_cnn_tool 6_mrz_ocr/data/corpus_cscd_photo/2026-9-30-K.ppm 2>&1
