#!/usr/bin/env python3
"""Verify EasyOCR custom_mrz model with sliding window approach.

The core problem: real MRZ strips are ~1440px wide but model was trained
on imgW=440. Compressing 1440→440 loses 3.3x horizontal detail.

Sliding window: scale strip to imgH=32 (preserving aspect ratio → ~1097px wide),
then slide a 440px window across with overlap, recognizing each window separately,
then merge results.

Also tests: single window at imgW=440 (current best), and full image at dynamic width.
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

gt_path = os.path.join(ROOT, "gt_photo_passport_official.json")
with open(gt_path) as f:
    gt_data = json.load(f)
gt = gt_data["gt"]

def load_rgb(path):
    img = Image.open(str(path)).convert("RGB")
    return np.array(img)

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

def recognize_tensor(img_tensor):
    """Run model on a preprocessed tensor [1,1,H,W] and return text, conf."""
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

def recognize_fixed(strip_np):
    """Recognize with fixed imgW=440 (current best approach)."""
    img = normalize_pad(strip_np, imgH, imgW)
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)
    return recognize_tensor(img_tensor)

def recognize_sliding(strip_np, window_w=440, stride=220):
    """Sliding window: scale to imgH=32 preserving aspect ratio, then slide."""
    img = Image.fromarray(strip_np, 'L')
    # Scale to imgH=32 preserving aspect ratio
    ratio = imgH / img.height
    new_w = max(1, int(img.width * ratio))
    img = img.resize((new_w, imgH), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32)
    img_np = img_np / 255.0
    img_np = (img_np - 0.5) / 0.5

    if new_w <= window_w:
        # Image fits in one window
        if new_w < window_w:
            pad = np.ones((imgH, window_w - new_w), dtype=np.float32)
            img_np = np.concatenate([img_np, pad], axis=1)
        img_tensor = torch.FloatTensor(img_np).unsqueeze(0).unsqueeze(0).to(device)
        return recognize_tensor(img_tensor)

    # Slide windows with overlap
    windows = []
    positions = []
    for start in range(0, new_w - window_w + 1, stride):
        end = start + window_w
        window = img_np[:, start:end]
        img_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
        text, conf = recognize_tensor(img_tensor)
        windows.append((text, conf))
        positions.append((start, end))

    # Also add last window if not covered
    if positions and positions[-1][1] < new_w:
        start = new_w - window_w
        end = new_w
        window = img_np[:, start:end]
        img_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
        text, conf = recognize_tensor(img_tensor)
        windows.append((text, conf))
        positions.append((start, end))

    # Merge: concatenate all window texts (simple approach)
    # TODO: smarter merge using overlap regions
    merged_text = ''.join(w[0] for w in windows)
    merged_conf = float(np.mean([w[1] for w in windows]))
    return merged_text, merged_conf

charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

def sanitise(s):
    s = s.upper().replace(' ', '')
    return ''.join(ch if ch in charset else '<' for ch in s)

def char_acc(pred, gt_str):
    correct = sum(p == g for p, g in zip(pred, gt_str))
    return correct / max(1, len(gt_str))

def native_band(rgb, pair, scale, single=False):
    H, W = rgb.shape[:2]
    first_top = max(0, int(round(pair[0][0] / scale)))
    second_bottom = min(H, int(round((pair[1][1] + 1) / scale)))
    heights = [
        max(1, int(round((y1 + 1) / scale)) - int(round(y0 / scale)))
        for y0, y1 in pair
    ]
    if single:
        top, bottom = first_top, second_bottom
        min_h = 30
        actual_h = bottom - top
        if actual_h < min_h:
            extra = (min_h - actual_h + 1) // 2
            top = max(0, top - extra)
            bottom = min(H, bottom + extra)
    else:
        pad = max(8, min(25, int(round(max(heights) * 0.70))))
        top = max(0, first_top - pad)
        bottom = min(H, second_bottom + pad)
    band = rgb[top:bottom, 0:W]
    return band

results_fixed = []
results_sliding = []

print("\n--- Testing fixed vs sliding window ---")
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
                continue
        is_single = pair[0] == pair[1]
        scale = work.shape[0] / max(1, rgb.shape[0])
        band_np = native_band(rgb, pair, scale, single=is_single)
        if band_np.size == 0:
            continue
    except Exception as e:
        continue

    band_gray = np.array(Image.fromarray(band_np).convert('L'))
    h, w = band_gray.shape
    mid = h // 2
    top_half = band_gray[:mid]
    bot_half = band_gray[mid:]

    # Fixed imgW=440
    l1_f, c1_f = recognize_fixed(top_half)
    l2_f, c2_f = recognize_fixed(bot_half)
    l1_f = sanitise(l1_f)[:44].ljust(44, '<')[:44]
    l2_f = sanitise(l2_f)[:44].ljust(44, '<')[:44]
    ca1_f = char_acc(l1_f, gt_l1)
    ca2_f = char_acc(l2_f, gt_l2)

    # Sliding window
    l1_s, c1_s = recognize_sliding(top_half)
    l2_s, c2_s = recognize_sliding(bot_half)
    l1_s = sanitise(l1_s)[:44].ljust(44, '<')[:44]
    l2_s = sanitise(l2_s)[:44].ljust(44, '<')[:44]
    ca1_s = char_acc(l1_s, gt_l1)
    ca2_s = char_acc(l2_s, gt_l2)

    results_fixed.append((fname, ca1_f, ca2_f, c1_f, c2_f))
    results_sliding.append((fname, ca1_s, ca2_s, c1_s, c2_s))

    flag = ""
    if ca1_s > ca1_f + 0.05 or ca2_s > ca2_f + 0.05:
        flag = " SLIDING_BETTER"
    elif ca1_f > ca1_s + 0.05 or ca2_f > ca2_s + 0.05:
        flag = " FIXED_BETTER"
    print(f"{fname}: FIXED L1={ca1_f:.2f} L2={ca2_f:.2f} | "
          f"SLIDE L1={ca1_s:.2f} L2={ca2_s:.2f}{flag}")

# Summary
print("\n=== SUMMARY ===")
if results_fixed:
    f_l1 = np.mean([r[1] for r in results_fixed])
    f_l2 = np.mean([r[2] for r in results_fixed])
    s_l1 = np.mean([r[1] for r in results_sliding])
    s_l2 = np.mean([r[2] for r in results_sliding])
    f_exact = sum(1 for r in results_fixed if r[1] == 1.0 and r[2] == 1.0)
    s_exact = sum(1 for r in results_sliding if r[1] == 1.0 and r[2] == 1.0)

    print(f"Fixed (imgW=440):    L1={f_l1:.3f} L2={f_l2:.3f} exact={f_exact}/{len(results_fixed)}")
    print(f"Sliding window:      L1={s_l1:.3f} L2={s_l2:.3f} exact={s_exact}/{len(results_sliding)}")
