#!/usr/bin/env python3
"""Launch EasyOCR MRZ custom training.

Steps:
1. Generate synthetic MRZ strip images (train + val)
2. Set up all_data directory structure for EasyOCR trainer
3. Run training with mrz_custom_config.yaml

Usage:
  python3 run_mrz_training.py --n_train 20000 --n_val 2000
  python3 run_mrz_training.py --n_train 20000 --n_val 2000 --ft
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAINER_DIR = ROOT / "tmp" / "easyocr_train" / "trainer"
GEN_SCRIPT = ROOT / "6_mrz_ocr" / "tools" / "train_easyocr_mrz" / "gen_mrz_strips.py"
CONFIG_FILE = TRAINER_DIR / "config_files" / "mrz_custom_config.yaml"
ALL_DATA = TRAINER_DIR / "all_data"
SAVED_MODELS = TRAINER_DIR / "saved_models"


def generate_data(n_train, n_val):
    """Generate train and val datasets."""
    train_out = ALL_DATA / "mrz_train"
    val_out = ALL_DATA / "mrz_val"

    print(f"\n{'='+ '='*60}")
    print(f"Generating training data: {n_train} samples")
    print(f"{'='*60}")
    subprocess.run([
        sys.executable, str(GEN_SCRIPT),
        "--train", "--out", str(train_out),
        "--n", str(n_train), "--seed", "1001"
    ], check=True)

    print(f"\n{'='*60}")
    print(f"Generating validation data: {n_val} samples")
    print(f"{'='*60}")
    subprocess.run([
        sys.executable, str(GEN_SCRIPT),
        "--eval", "--out", str(val_out),
        "--n", str(n_val), "--seed", "2002"
    ], check=True)


def setup_dirs():
    """Create necessary directories."""
    ALL_DATA.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS.mkdir(parents=True, exist_ok=True)
    (SAVED_MODELS / "mrz_custom").mkdir(exist_ok=True)


def run_training(ft=False, saved_model=""):
    """Run EasyOCR training."""
    cmd = [
        sys.executable, "train.py",
        "--config", str(CONFIG_FILE),
    ]
    if ft and saved_model:
        cmd += ["--saved_model", saved_model, "--FT"]
    elif saved_model:
        cmd += ["--saved_model", saved_model]

    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"  config: {CONFIG_FILE}")
    print(f"  FT: {ft}")
    print(f"  saved_model: {saved_model or '(none)'}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRAINER_DIR)

    subprocess.run(cmd, cwd=str(TRAINER_DIR), env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--ft", action="store_true", help="Fine-tune from saved model")
    parser.add_argument("--saved_model", default="", help="Path to pretrained model")
    parser.add_argument("--skip_data", action="store_true", help="Skip data generation")
    args = parser.parse_args()

    setup_dirs()
    if not args.skip_data:
        generate_data(args.n_train, args.n_val)
    run_training(ft=args.ft, saved_model=args.saved_model)
