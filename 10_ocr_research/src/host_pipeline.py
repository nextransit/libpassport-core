"""端到端 host pipeline: Photo -> restore (mrz_pair) -> crop_only -> CNN/decode.

策略：
  - 用 passport_photo_restore 检测结构先验 + find_mrz_pair 严格双行（天然剔除倒影）
  - 不做 rectify, 直接 crop 包含两行的最小矩形 (保留自然形态)
  - 喂给 mrz_ocr_cnn_tool (其内部 deskew + 字符切分)
"""
import sys, json, time, subprocess
sys.path.insert(0, 'passport_photo_restore/src')
import cv2
from pathlib import Path
from restore import load, _gray, detect_structure

ROOT = Path('.')
TOOL = ROOT / '6_mrz_ocr' / 'build' / 'mrz_ocr_cnn_tool'
TMP = ROOT / '10_ocr_research' / '_host_pipe.ppm'


def write_ppm(rgb, p):
    from PIL import Image
    Image.fromarray(rgb).save(p, format='PPM')


def run_cnn(ppm):
    t0 = time.perf_counter()
    r = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True, timeout=80)
    ms = (time.perf_counter() - t0) * 1000
    out = {}
    for ln in r.stdout.splitlines():
        if ':' in ln:
            k, _, v = ln.partition(':')
            out[k.strip()] = v.strip()
    return {
        'ok': r.returncode == 0 and out.get('result.ok') == 'OK',
        'line1': out.get('result.line1', ''),
        'line2': out.get('result.line2', ''),
        'conf1': out.get('result.conf1', ''),
        'conf2': out.get('result.conf2', ''),
        'ms': round(ms, 1),
        'err': out.get('result.ok', ''),
    }


def crop_pair(work_rgb, st, pad=0.06):
    pair = st.get('mrz_pair')
    if not pair: return None
    (l1y0, _), (l2y0, l2y1) = pair
    y0, y1 = l1y0, l2y1
    pad_px = max(3, int((y1 - y0) * pad))
    cy0 = max(0, y0 - pad_px)
    cy1 = min(work_rgb.shape[0] - 1, y1 + pad_px)
    return work_rgb[cy0:cy1 + 1]


PHOTO = sorted(ROOT.glob('Photo/*/*.jpg'))
results = []
total_ms = 0
for img in PHOTO:
    rgb = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    work, sc = load(rgb)
    g = _gray(work)
    st = detect_structure(g)
    crop = crop_pair(work, st)
    restore_ms = (time.perf_counter() - t0) * 1000
    e = {'file': str(img.relative_to(ROOT))}
    e['scale'] = round(sc, 3)
    e['mrz_pair'] = st.get('mrz_pair')
    e['restore_ms'] = round(restore_ms, 1)
    if crop is not None:
        e['crop_shape'] = list(crop.shape)
        write_ppm(crop, TMP)
        e['cnn'] = run_cnn(TMP)
        e['total_ms'] = round(restore_ms + e['cnn']['ms'], 1)
    else:
        e['cnn'] = {'ok': False, 'err': 'no_pair'}
    results.append(e)
    total_ms += e.get('total_ms', restore_ms)
    print(f'{e["file"][-30:]} pair={e["mrz_pair"]} restore={e["restore_ms"]}ms '
          f'crop_shape={e.get("crop_shape")} '
          f'cnn.ok={e["cnn"]["ok"]} conf=({e["cnn"].get("conf1","")},{e["cnn"].get("conf2","")}) '
          f'cnn.ms={e["cnn"].get("ms","")} '
          f'l1={e["cnn"].get("line1","")[:24]!r}')

json.dump(results, open(ROOT / '10_ocr_research' / 'host_pipeline_results.json', 'w'),
          ensure_ascii=False, indent=1)
print(f'\nsaved -> 10_ocr_research/host_pipeline_results.json')
n_ok = sum(1 for e in results if e['cnn']['ok'])
print(f'CNN ok: {n_ok}/{len(results)}  总流水线 ms={total_ms:.0f}')
