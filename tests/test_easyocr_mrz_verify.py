#!/usr/bin/env python3
"""Verify EasyOCR custom_mrz model on real MRZ band images.

Uses restore.detect_structure to properly locate MRZ band, and uses
mrz_pair coordinates to precisely extract line1/line2 regions
(not just split by midpoint).

Tests direct model inference with fixed imgW=440 scaling.
"""
import json
import os
import sys
import time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "passport_photo_restore", "src"))

os.environ["MRZ_FAST_STRICT"] = "0"

from restore import _gray, detect_structure

# --- Load GT ---
gt_path = os.path.join(ROOT, "gt_photo_passport_official.json")
with open(gt_path) as f:
    gt_data = json.load(f)
gt = gt_data["gt"]

def load_rgb(path):
    img = Image.open(str(path)).convert("RGB")
    return np.array(img)

# --- Load EasyOCR custom_mrz reader ---
print("Loading EasyOCR custom_mrz reader...")
import easyocr
reader = easyocr.Reader(['en'], recog_network='custom_mrz',
                         detector=False, gpu=False, verbose=False)
print("Reader loaded.")

character = reader.character
imgH = 32
imgW = 440

import torch

model = reader.recognizer
device = reader.device

def normalize_pad(image_np, imgH=32, imgW=440):
    img = Image.fromarray(image_np, 'L')
    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.5
    return img

def recognize_direct(strip_np):
    img = normalize_pad(strip_np, imgH, imgW)
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img_tensor)
    preds = preds.log_softmax(2)
    preds_argmax = preds[0].argmax(1)
    decoded = []
    probs = []
    prev = -1
    for t in range(preds_argmax.size(0)):
        idx = preds_argmax[t].item()
        if idx != prev and idx != 0:
            decoded.append(idx)
            prob = torch.exp(preds[0, t, idx]).item()
            probs.append(prob)
        prev = idx
    text = ''.join(character[i-1] for i in decoded)
    if probs:
        probs_arr = np.array(probs)
        conf = float(probs_arr.prod() ** (2.0 / np.sqrt(len(probs_arr))))
    else:
        conf = 0.0
    return text, conf

charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

def sanitise(s):
    s = s.upper().replace(' ', '')
    return ''.join(ch if ch in charset else '<' for ch in s)

def char_acc(pred, gt_str):
    correct = sum(p == g for p, g in zip(pred, gt_str))
    return correct / max(1, len(gt_str))

results = []
failed = []

print("\n--- Testing on PassPortPhoto images (precise line extraction) ---")
for fname in sorted(gt.keys()):
    gt_entry = gt[fname]
    gt_l1 = gt_entry["l1"]
    gt_l2 = gt_entry["l2"]
    img_path = gt_entry["label"]

    try:
        rgb = load_rgb(img_path)
        work = rgb
        if max(rgb.shape[:2]) > 1250:
            scale_f = 1250.0 / max(rgb.shape[:2])
            work = np.array(Image.fromarray(rgb).resize(
                (int(rgb.shape[1] * scale_f), int(rgb.shape[0] * scale_f)),
                Image.BILINEAR))
        st = detect_structure(_gray(work))
        pair = st.get("mrz_pair")
        if not pair:
            single = st.get("mrz_single")
            if single:
                pair = (single, single)
            else:
                print(f"{fname}: NO PAIR")
                failed.append(fname)
                continue
        is_single = pair[0] == pair[1]
        scale = work.shape[0] / max(1, rgb.shape[0])
        H, W = rgb.shape[:2]

        # Precise line extraction using pair coordinates
        # pair = ((y0_start, y0_end), (y1_start, y1_end)) in work image coords
        # Map to original resolution
        l1_top = max(0, int(round(pair[0][0] / scale)))
        l1_bot = min(H, int(round((pair[0][1] + 1) / scale)))
        l2_top = max(0, int(round(pair[1][0] / scale)))
        l2_bot = min(H, int(round((pair[1][1] + 1) / scale)))

        # Add small padding (3px) to each line for context
        pad_px = 3
        l1_top = max(0, l1_top - pad_px)
        l1_bot = min(H, l1_bot + pad_px)
        l2_top = max(0, l2_top - pad_px)
        l2_bot = min(H, l2_bot + pad_px)

        line1_img = rgb[l1_top:l1_bot, :]
        line2_img = rgb[l2_top:l2_bot, :]

        # Convert to grayscale
        line1_gray = np.array(Image.fromarray(line1_img).convert('L'))
        line2_gray = np.array(Image.fromarray(line2_img).convert('L'))

    except Exception as e:
        print(f"{fname}: ERROR {e}")
        failed.append(fname)
        continue

    # Direct model inference
    t0 = time.time()
    l1_raw, c1 = recognize_direct(line1_gray)
    l2_raw, c2 = recognize_direct(line2_gray)
    t_di = time.time() - t0

    l1 = sanitise(l1_raw)[:44].ljust(44, '<')[:44]
    l2 = sanitise(l2_raw)[:44].ljust(44, '<')[:44]

    ca1 = char_acc(l1, gt_l1)
    ca2 = char_acc(l2, gt_l2)

    results.append((fname, ca1, ca2, c1, c2, t_di, l1, l2, line1_gray.shape, line2_gray.shape))

    status = ""
    if ca1 < 0.9 or ca2 < 0.9:
        status = " <-- LOW"
    print(f"{fname}: L1={ca1:.2f} L2={ca2:.2f} c={c1:.2f}/{c2:.2f} "
          f"shape={line1_gray.shape}/{line2_gray.shape}{status}")

# --- Summary ---
print("\n=== SUMMARY ===")
if results:
    l1_avg = np.mean([r[1] for r in results])
    l2_avg = np.mean([r[2] for r in results])
    time_avg = np.mean([r[5] for r in results])
    exact = sum(1 for r in results if r[1] == 1.0 and r[2] == 1.0)

    print(f"Direct inference (precise line extraction):")
    print(f"  L1 char_acc={l1_avg:.3f}, L2 char_acc={l2_avg:.3f}")
    print(f"  avg_time={time_avg*1000:.0f}ms")
    print(f"  exact: {exact}/{len(results)}")

    # Confidence analysis
    correct_confs = []
    wrong_confs = []
    for r in results:
        if r[1] < 1.0:
            wrong_confs.append(r[3])
        else:
            correct_confs.append(r[3])
        if r[2] < 1.0:
            wrong_confs.append(r[4])
        else:
            correct_confs.append(r[4])

    if correct_confs and wrong_confs:
        print(f"\nConfidence:")
        print(f"  Correct: mean conf = {np.mean(correct_confs):.3f}")
        print(f"  Wrong:   mean conf = {np.mean(wrong_confs):.3f}")
        print(f"  Separation: {np.mean(correct_confs) - np.mean(wrong_confs):.3f}")

    # Show worst cases
    print("\nWorst L1 cases:")
    worst_l1 = sorted(results, key=lambda r: r[1])[:5]
    for r in worst_l1:
        print(f"  {r[0]}: L1={r[1]:.2f} pred={r[6][:20]}... gt={gt[r[0]]['l1'][:20]}...")

    print("\nWorst L2 cases:")
    worst_l2 = sorted(results, key=lambda r: r[2])[:5]
    for r in worst_l2:
        print(f"  {r[0]}: L2={r[2]:.2f} pred={r[7][:20]}... gt={gt[r[0]]['l2'][:20]}...")

if failed:
    print(f"\nFailed: {failed}")
