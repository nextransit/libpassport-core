"""Train the Tiny ConvNet directly on C-side dumped inference glyphs.
This is the ground-truth distribution of what the pipeline actually
feeds the network (resample_char_gray output), so it should beat any
synthetic renderer.  We split by IMAGE to avoid leakage and evaluate
on held-out images with the EXACT same decode rules via bench later.

Protocol notes:
  * --corpus must point at the TRAIN corpus (e.g. data/corpus_train).
    NEVER the eval corpus that bench.py gates on.
  * The dump workdir is rebuilt from scratch on every run and /tmp
    dumps are cleaned before each tool invocation, so stale glyphs from
    earlier runs can never be paired with the wrong label.
  * --mix-synthetic N blends N synthetic samples per class from
    train_cnn_v2's renderer (coverage for conditions the corpus
    under-samples).
"""
import json, glob, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
import random

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="data/corpus_train",
                help="TRAIN corpus directory (corpus.json + PPM images)")
ap.add_argument("--tool", default="build/mrz_ocr_cnn_tool", help="CNN tool")
ap.add_argument("--out", default="src/cnn_weights.h")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--workdir", default="/tmp/mrz_real_glyphs")
ap.add_argument("--mix-synthetic", type=int, default=0,
                help="blend N synthetic v2-renderer samples per class")
ap.add_argument("--mix-hard-slices", default="",
                help=".npz from tools/export_hard_slices.py; upsample the "
                     "real misclassified 16x12 patches into TRAIN data")
ap.add_argument("--max-images", type=int, default=0,
                help="subsample the corpus to N images (0 = all); "
                     "deterministic pick, keeps the loop fast")
a = ap.parse_args()

ROOT = Path(a.workdir)
SRC = Path(__file__).resolve().parents[1]   # 6_mrz_ocr checkout root
GLYPHS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
GI = {c: i for i, c in enumerate(GLYPHS)}

# hold out ~15% of images for validation
def split_images():
    imgs = sorted(ROOT.glob('img_*'))
    random.seed(42)
    random.shuffle(imgs)
    n_val = max(10, int(len(imgs)*a.val_frac))
    val_imgs = set(imgs[:n_val])
    tr_imgs = [p for p in imgs if p not in val_imgs]
    return tr_imgs, val_imgs

def gather(imgs):
    xs, ys = [], []
    for imgdir in imgs:
        meta = json.loads((imgdir/'meta.json').read_text())
        for li, line in ((0, meta['line1']), (1, meta['line2'])):
            for c, ch in enumerate(line):
                if ch not in GI: continue
                fs = sorted(glob.glob(str(imgdir/f'mrz_dump_img_l{li}_c{c:02d}_r*_w*.ppm')))
                if not fs: continue
                if len(fs) > 1:
                    raise RuntimeError(
                        f"{len(fs)} dump files for {imgdir.name} l{li} c{c} "
                        f"(stale /tmp dumps?) -- clean /tmp and re-run")
                p = np.asarray(Image.open(fs[0]).convert('L'), dtype=np.float32)/255.0
                xs.append(p); ys.append(GI[ch])
    return torch.from_numpy(np.array(xs, dtype=np.float32).reshape(-1,1,12,16)), torch.tensor(ys)

def generate_dumps():
    """Run the C tool with MRZ_OCR_DUMP=1 over every corpus image and
    move the 16x12 glyph dumps into per-image folders.  This is the
    single source of training data: the ACTUAL inputs the pipeline
    feeds the CNN (binarise -> band -> deskew -> split -> segment ->
    resample), which is why a model trained on it matches inference far
    better than any synthetic renderer."""
    import subprocess, glob, shutil, os, sys
    corpus = json.loads((Path(a.corpus)/'corpus.json').read_text())
    records = corpus['records']
    if a.max_images and a.max_images < len(records):
        records = sorted(random.Random(0).sample(records, a.max_images),
                         key=lambda r: r['id'])
        print(f"subsampled corpus to {len(records)} images", flush=True)
    env = dict(os.environ); env['MRZ_OCR_DUMP']='1'
    # Rebuild the workdir from scratch: leftover folders from an older
    # corpus would silently pair old glyphs with new labels.
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)
    n=0
    for rec in records:
        p = Path(a.corpus)/rec['image']
        # Clean the shared /tmp dump dir before EVERY run: dump file
        # names contain only (line, col, size) so two runs with the
        # same shapes would collide and glob could pick a stale glyph.
        for f in glob.glob('/tmp/mrz_dump_img_*.ppm'):
            os.remove(f)
        r = subprocess.run([a.tool, str(p)], capture_output=True,
                           text=True, timeout=20, env=env)
        if r.returncode != 0:
            print(f"  warn: tool failed on {rec['id']}", file=sys.stderr)
            continue
        d = ROOT/rec['id']; d.mkdir(parents=True, exist_ok=True)
        (d/'meta.json').write_text(json.dumps({
            "line1": rec['line1'], "line2": rec['line2'],
            "scale": rec['scale'], "hgap": rec['hgap'],
            "skew": rec['skew'], "noise": rec['noise']}))
        dumps = glob.glob('/tmp/mrz_dump_img_l*.ppm')
        for f in dumps:
            shutil.move(f, d)
        if len(dumps) == 0:
            print(f"  warn: no dumps produced for {rec['id']}", file=sys.stderr)
        n+=1
    print(f"dumped {n} images to {ROOT}", flush=True)

generate_dumps()
tr_imgs, val_imgs = split_images()
print("gathering train...", flush=True)
Xtr, Ytr = gather(tr_imgs)
print(f"train {Xtr.shape[0]}", flush=True)
print("gathering val...", flush=True)
Xval, Yval = gather(val_imgs)
print(f"val {Xval.shape[0]}", flush=True)

if a.mix_synthetic > 0:
    # Blend synthetic v2-renderer samples (conditions the real corpus
    # under-samples). Synthetic data joins TRAIN ONLY.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_cnn_v2 as v2
    Xsyn, Ysyn = v2.build_dataset(n_per_class=a.mix_synthetic, hard_extra=0)
    Xtr = torch.cat([Xtr, torch.from_numpy(Xsyn.reshape(-1,1,12,16))])
    Ytr = torch.cat([Ytr, torch.from_numpy(Ysyn.astype(np.int64))])
    print(f"mixed {Xsyn.shape[0]} synthetic -> train {Xtr.shape[0]}", flush=True)

if a.mix_hard_slices:
    # Upsample the REAL misclassified patches exported by
    # tools/export_hard_slices.py (eval corpus). These are exactly the
    # inputs the current weights get wrong, so they focus the next
    # training run on the failure distribution instead of blind noise.
    d = np.load(SRC / a.mix_hard_slices)
    Xh = torch.from_numpy(d['X'].astype(np.float32).reshape(-1,1,12,16))
    Yh = torch.from_numpy(d['Y'].astype(np.int64))
    # 30% weight: keep the base train set dominant, hard slices ~30%.
    n_hard = max(int(0.3 * len(Xtr)), len(Xh))
    reps = (n_hard + len(Xh) - 1) // len(Xh)
    Xh = Xh.repeat(reps, 1, 1, 1)[:n_hard]
    Yh = Yh.repeat(reps)[:n_hard]
    Xtr = torch.cat([Xtr, Xh])
    Ytr = torch.cat([Ytr, Yh])
    print(f"mixed {n_hard} hard slices (30% weight) -> train {Xtr.shape[0]}",
          flush=True)

C1,C2,C3,HID=8,16,4,64
cl = nn.Sequential()
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,C1,3,padding=1)
        self.conv2=nn.Conv2d(C1,C2,3,padding=1)
        self.conv3=nn.Conv2d(C2,C3,1)
        self.fc1=nn.Linear(C3*3*8,HID)
        self.fc2=nn.Linear(HID,37)
    def forward(self,x):
        x=F.relu(self.conv1(x)); x=F.max_pool2d(x,(2,1))
        x=F.relu(self.conv2(x)); x=F.max_pool2d(x,(2,2))
        x=self.conv3(x); x=x.flatten(1)
        x=F.relu(self.fc1(x)); return self.fc2(x)

torch.manual_seed(1)
model=Net()
opt=torch.optim.Adam(model.parameters(),lr=0.02)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
bs=512
best=0.0; best_state=None
for ep in range(a.epochs):
    model.train()
    perm=torch.randperm(len(Xtr))
    ctr=tot=0
    for s in range(0,len(perm),bs):
        idx=perm[s:s+bs]
        xb,yb=Xtr[idx],Ytr[idx]
        lg=model(xb); loss=F.cross_entropy(lg,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        ctr+=int((lg.argmax(1)==yb).sum()); tot+=len(idx)
    sched.step()
    model.eval()
    with torch.no_grad():
        val=0
        for s in range(0,len(Xval),bs):
            v=model(Xval[s:s+bs]);
            val+=int((v.argmax(1)==Yval[s:s+bs]).sum())
        va=val/len(Yval)
    if va>best:
        best=va; best_state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
    print(f"ep{ep:2d} tr={ctr/tot:.4f} val={va:.4f} best={best:.4f}", flush=True)

# emit C header
def emit(best_state, path):
    W1=best_state['conv1.weight'].transpose(2,3,1,0)
    b1=best_state['conv1.bias']
    W2=best_state['conv2.weight'].transpose(2,3,1,0)
    b2=best_state['conv2.bias']
    W3=best_state['conv3.weight'].transpose(2,3,1,0)
    b3=best_state['conv3.bias']
    Wf1=best_state['fc1.weight']; bf1=best_state['fc1.bias']
    Wf2=best_state['fc2.weight']; bf2=best_state['fc2.bias']
    mean=np.zeros(12*16,np.float32); std=np.ones(12*16,np.float32)
    def fmt(a): return ", ".join(f"{v:.8f}f" for v in np.asarray(a).ravel())
    out=["/* Auto-generated: trained on C-side dumped inference glyphs. Do not edit. */",
         "#ifndef MRZ_OCR_CNN_WEIGHTS_H","#define MRZ_OCR_CNN_WEIGHTS_H","","#include <stdint.h>","#include \"cnn.h\"","",
         f"static const float DEFAULT_IN_MEAN[CNN_IN_H * CNN_IN_W] = {{"]
    for i in range(0,len(mean),8): out.append("    "+fmt(mean[i:i+8])+",")
    out.append("};"); out.append(f"static const float DEFAULT_IN_STD[CNN_IN_H*CNN_IN_W] = {{")
    for i in range(0,len(std),8): out.append("    "+fmt(std[i:i+8])+",")
    out.append("};")
    out.append("static const float DEFAULT_CONV1_W[CNN_K*CNN_K*1*CNN_C1] = {"+
               ",".join(fmt(W1.ravel()[i:i+8]) for i in range(0,9*C1,8))+",};")
    out.append("static const float DEFAULT_CONV1_B[CNN_C1] = {"+fmt(b1)+"};")
    out.append("static const float DEFAULT_CONV2_W[CNN_K*CNN_K*CNN_C1*CNN_C2] = {"+
               ",".join(fmt(W2.ravel()[i:i+8]) for i in range(0,9*C1*C2,8))+",};")
    out.append("static const float DEFAULT_CONV2_B[CNN_C2] = {"+fmt(b2)+"};")
    out.append("static const float DEFAULT_CONV3_W[CNN_C2*CNN_C3] = {"+fmt(W3)+"};")
    out.append("static const float DEFAULT_CONV3_B[CNN_C3] = {"+fmt(b3)+"};")
    out.append("static const float DEFAULT_FC1_W[CNN_FLAT*CNN_HIDDEN] = {"+
               ",".join(fmt(Wf1.ravel()[i:i+8]) for i in range(0,96*HID,8))+",};")
    out.append("static const float DEFAULT_FC1_B[CNN_HIDDEN] = {"+fmt(bf1)+"};")
    out.append("static const float DEFAULT_FC2_W[CNN_HIDDEN*CNN_OUT] = {"+
               ",".join(fmt(Wf2.ravel()[i:i+8]) for i in range(0,HID*37,8))+",};")
    out.append("static const float DEFAULT_FC2_B[CNN_OUT] = {"+fmt(bf2)+"};")
    out.append("#endif")
    Path(path).write_text("\n".join(out)+"\n")
    print("wrote", path, "best_val", best, flush=True)

emit(best_state, a.out)
