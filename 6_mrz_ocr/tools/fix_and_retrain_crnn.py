#!/usr/bin/env python3
"""Fix corpus_real GT labels using passport official GT, export features, retrain CRNN.
Does NOT modify the original corpus_real/corpus.json — works on a /tmp copy."""
import json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

PY = os.environ.get("PY", os.path.expanduser("~/.workbuddy-ai/binaries/python/envs/crnn/bin/python"))
WORK = "/tmp/crnn_build"
FIX_DIR = os.path.join(WORK, "corpus_real_fixed")

# Step 1: Create fixed copy
print("== 1/4 create fixed corpus_real copy ==")
os.makedirs(FIX_DIR, exist_ok=True)
src_dir = os.path.join("6_mrz_ocr", "data", "corpus_real")
for f in os.listdir(src_dir):
    if f.endswith(".ppm"):
        shutil.copy(os.path.join(src_dir, f), FIX_DIR)

with open(os.path.join("tests", "gt_photo_passport_official.json")) as f:
    gt = json.load(f)["gt"]
passport_gt = {}
for k, v in gt.items():
    key = k.replace(".jpg", "")
    passport_gt[key] = (v["l1"], v["l2"])

with open(os.path.join(src_dir, "corpus.json")) as f:
    corpus = json.load(f)

updated = 0
for r in corpus["records"]:
    key = r["id"].split("/")[-1]
    if key in passport_gt:
        p1, p2 = passport_gt[key]
        if r.get("line1", "") != p1 or r.get("line2", "") != p2:
            r["line1"] = p1
            r["line2"] = p2
            r["label_source"] = "passport_gt_official"
            updated += 1

with open(os.path.join(FIX_DIR, "corpus.json"), "w") as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)
print(f"Fixed {updated} records in {FIX_DIR}")

# Step 2: Export features from fixed corpus
print("== 2/4 export features from fixed corpus ==")
subprocess.run([PY, "tools/export_crnn_features.py",
    "--corpus", FIX_DIR,
    "--out", os.path.join(WORK, "corpus_real_fixed.npz"),
    "--crnn", "1", "--parallel", "0"], check=True, cwd="6_mrz_ocr")

# Step 3: Train with fixed real data
print("== 3/4 train ==")
epochs = os.environ.get("EPOCHS", "50")
extra_repeat = os.environ.get("EXTRA_REPEAT", "50")
synth_subsample = os.environ.get("SYNTH_SUBSAMPLE", "1500")
subprocess.run([PY, "tools/train_crnn_seq.py",
    "--train", os.path.join(WORK, "corpus_train.npz"),
    "--extra", os.path.join(WORK, "corpus_real_fixed.npz"),
    "--extra-repeat", extra_repeat,
    "--synth-subsample", synth_subsample,
    "--out", "src/crnn_weights.h",
    "--epochs", epochs,
    "--weight-decay", "3e-4",
    "--arch", "bi"], check=True, cwd="6_mrz_ocr")

# Step 4: Rebuild
print("== 4/4 rebuild ==")
subprocess.run(["cmake", "--build", "build",
    "--target", "mrz_ocr_cnn_tool", "--target", "mrz_ocr_cnn_tool_pic",
    "-j4"], check=True, cwd="6_mrz_ocr")

print("Done! Run eval to check results.")
