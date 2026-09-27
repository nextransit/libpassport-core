#!/usr/bin/env python3
"""Fetch passport photos from Wikimedia Commons for testing.

Downloads at least MIN_COUNT images into assets/pic/ with MD5 dedup and
jpg/png filtering. Usage: python3 tests/fetch_passports.py [--min N] [--out DIR]
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
CATS = [
    "Category:Passports",
    "Category:Passports of the world",
    "Category:Passport photographs",
    "Category:Machine-readable passports",
]
# Precise search terms targeting passport biographic/data pages (photo +
# MRZ + personal info), NOT covers / historical applications / portraits.
SEARCH_TERMS = [
    "passport data page",
    "passport information page",
    "passport biodata page",
    "passport MRZ page",
    "machine readable passport page",
    "passport specimen data page",
    "passport biographical page",
]
UA = {"User-Agent": "passport-bench-tool/1.0 (local testing)"}


def api_get(params, retries=3):
    params = {k: v for k, v in params.items() if v is not None}
    params = dict(params, format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if i == retries - 1:
                print(f"[warn] api_get failed {url}: {e}", file=sys.stderr)
                return None
            wait = 8 if "429" in str(e) else 1 * (i + 1)
            time.sleep(wait)
    return None


def cat_members(cat, budget=1500, max_depth=1, max_subcats=60):
    """Yield file titles from a category (+subcategories, max_depth levels).
    Stops after `budget` files yielded to bound API traffic."""
    seen = set()
    queued = {cat}
    stack = [(cat, 0)]
    yielded = 0
    subcats = 0
    while stack and yielded < budget:
        title, depth = stack.pop(0)
        cont = {}
        pages = 0
        while True:
            pages += 1
            if pages > 10:
                break
            p = dict(action="query", list="categorymembers",
                     cmtitle=title, cmtype="file|subcat", cmlimit="500",
                     cmcontinue=cont.get("cmcontinue"))
            d = api_get(p)
            if not d:
                break
            q = d.get("query", {}).get("categorymembers", [])
            for m in q:
                if m.get("ns") == 14:  # subcategory
                    if depth < max_depth and subcats < max_subcats \
                            and m["title"] not in queued:
                        queued.add(m["title"])
                        subcats += 1
                        stack.append((m["title"], depth + 1))
                elif m.get("ns") == 6:  # file
                    if m["title"] not in seen:
                        seen.add(m["title"])
                        yielded += 1
                        yield m["title"]
            cont = d.get("continue", {})
            if not cont:
                break
    return


def search_titles(term, budget=300):
    """Yield file titles from Commons full-text search (ns=6)."""
    seen = set()
    offset = 0
    yielded = 0
    while yielded < budget:
        d = api_get(dict(action="query", list="search",
                         srsearch=term, srnamespace="6", srlimit="50",
                         sroffset=str(offset)))
        if not d:
            break
        hits = d.get("query", {}).get("search", [])
        if not hits:
            break
        for h in hits:
            t = h.get("title")
            if t and t not in seen:
                seen.add(t)
                yielded += 1
                yield t
        offset += len(hits)
    return


def file_urls(titles):
    """Batch imageinfo -> {title: url}, 50 titles per request."""
    urls = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        d = api_get(dict(action="query", titles="|".join(batch),
                         prop="imageinfo", iiprop="url|size", iiurlwidth="800"))
        if not d:
            continue
        for pg in d.get("query", {}).get("pages", {}).values():
            tt = pg.get("title")
            ii = pg.get("imageinfo") or []
            if tt and ii:
                urls[tt] = ii[0].get("thumburl") or ii[0].get("url")
    return urls


def download(url, title):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def is_passport_page(data):
    """Return (ok, detail) — quick locator check that the image looks like a
    passport biographic page (photo + data fields + MRZ zone), not a cover /
    portrait / unrelated photo. Uses the locator on a downscaled thumbnail."""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((1000, 1000))
        import sys
        from pathlib import Path
        tests_dir = str(Path(__file__).resolve().parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from passport_locator import locate_regions  # type: ignore
        loc = locate_regions(img)
        mrz = loc.get("mrz")
        if not mrz:
            return False, "no mrz region"
        # MRZ band should be a reasonable strip near the bottom half.
        cy = mrz["y"] + mrz["h"] / 2.0
        if cy < img.height * 0.45:
            return False, f"mrz not bottom ({cy:.0f}/{img.height})"
        if mrz["h"] > img.height * 0.8:
            return False, f"mrz band too tall ({mrz['h']}/{img.height})"
        found = [k for k in ("photo", "data") if loc.get(k)]
        return True, f"mrz+{'/'.join(found)}"
    except Exception as e:  # noqa: BLE001
        return False, f"locator error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "pic"))
    ap.add_argument("--purge", action="store_true",
                    help="remove existing files that fail the passport-page check")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.purge:
        removed = kept = 0
        for f in sorted(os.listdir(args.out)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(args.out, f)
            with open(p, "rb") as fh:
                data = fh.read()
            ok, detail = is_passport_page(data)
            if ok:
                kept += 1
            else:
                os.unlink(p)
                removed += 1
                print(f"[purge] {f} ({detail})", file=sys.stderr)
        print(f"purge done: kept={kept} removed={removed}")
        sys.exit(0)

    existing = {f for f in os.listdir(args.out)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    print(f"existing: {len(existing)}")

    want = args.min - len(existing)
    if want <= 0:
        print(f"already >= {args.min} images, done")
        return

    titles = []
    for term in SEARCH_TERMS:
        if len(titles) >= want * 6:
            break
        for t in search_titles(term):
            if t not in titles:
                titles.append(t)
            if len(titles) >= want * 6:
                break
        print(f"[search] '{term}' -> {len(titles)} candidate titles so far",
              file=sys.stderr)
    if len(titles) < want * 2:
        for c in CATS:
            if len(titles) >= want * 6:
                break
            for t in cat_members(c):
                if t not in titles:
                    titles.append(t)
                if len(titles) >= want * 6:
                    break
            print(f"[{c}] collected {len(titles)} candidate titles so far",
                  file=sys.stderr)
    print(f"collected {len(titles)} candidate titles")

    dup = set()
    got = len(existing)
    seen_md5 = set()
    for f in existing:
        seen_md5.add(f.rsplit("_", 1)[0])
    n = 0

    urlmap = file_urls(titles)
    items = []
    for title in titles:
        name = title.split(":", 1)[1]
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        url = urlmap.get(title)
        if url:
            items.append((name, url))
    print(f"downloadable candidates: {len(items)}", file=sys.stderr)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch(item):
        name, url = item
        try:
            return name, download(url, name)
        except Exception as e:
            return name, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_fetch, it) for it in items]
        for fut in as_completed(futs):
            if got >= args.min:
                break
            name, data = fut.result()
            if not data:
                print(f"[skip] {name}", file=sys.stderr)
                continue
            ok, detail = is_passport_page(data)
            if not ok:
                print(f"[drop] {name} ({detail})", file=sys.stderr)
                continue
            digest = hashlib.md5(data).hexdigest()
            if digest in seen_md5:
                continue
            seen_md5.add(digest)
            ext = os.path.splitext(name)[1].lower()
            out = os.path.join(args.out, f"{digest[:16]}_{n:04d}{ext}")
            with open(out, "wb") as f:
                f.write(data)
            got += 1
            n += 1
            print(f"[{got}/{args.min}] {name} -> {os.path.basename(out)}", flush=True)

    print(f"done: {got} images in {args.out}")
    if got < args.min:
        print(f"[warn] only {got} (< {args.min}); rerun to collect more")
        sys.exit(1)


if __name__ == "__main__":
    main()