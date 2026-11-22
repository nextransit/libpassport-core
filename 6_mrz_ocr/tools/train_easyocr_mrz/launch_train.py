#!/usr/bin/env python3
"""Launch EasyOCR MRZ custom training with proper arg parsing."""
from __future__ import annotations
import argparse
import os
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAINER_DIR = ROOT / "tmp" / "easyocr_train" / "trainer"
sys.path.insert(0, str(TRAINER_DIR))
os.chdir(TRAINER_DIR)

class Opt:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(TRAINER_DIR / "config_files" / "mrz_custom_config.yaml"))
    parser.add_argument("--saved_model", default="")
    parser.add_argument("--FT", action="store_true")
    parser.add_argument("--num_iter", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--valInterval", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        opt = Opt(yaml.safe_load(f))

    if args.saved_model:
        opt.saved_model = args.saved_model
    if args.FT:
        opt.FT = True
    if args.num_iter is not None:
        opt.num_iter = args.num_iter
    if args.batch_size is not None:
        opt.batch_size = args.batch_size
    if args.lr is not None:
        opt.lr = args.lr
    if args.valInterval is not None:
        opt.valInterval = args.valInterval

    opt.character = opt.number + opt.symbol + opt.lang_char

    from train import train
    train(opt)


if __name__ == "__main__":
    main()
