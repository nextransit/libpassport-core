#!/usr/bin/env python3
"""Build DETERMINISTIC MRZ corpora for the OCR module.

Key protocol (fixes the historical evaluation-integrity problems):

  * Every record is rendered with a fixed per-record noise seed, so a
    corpus directory is reproducible bit-for-bit and benchmark results
    are comparable across code changes.
  * Training and evaluation corpora are SEPARATE (different seeds and
    different ground-truth strings). bench.py only ever sees the eval
    corpus; the train corpus is consumed by tools/train_cnn_real.py.
  * All ground truth is ICAO 9303 TD3 valid: line 1 name fields contain
    only A-Z and '<', line 2 has real mod-7/3/1 check digits, numeric
    dates and M/F/< sex. This keeps the strict syntax whitelist
    consistent with the truth instead of fighting it.

Profiles:
  clean      axis-aligned binary ink (historical distribution)
  realistic  adds rotation, blur, illumination gradient, invert and
             heavier noise -- approximates real scanner output

Usage:
  python3 tests/gen_corpus.py --eval  --out data/corpus_eval  --seed 2002
  python3 tests/gen_corpus.py --train --out data/corpus_train --seed 1001
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN  = ROOT / "build" / "gen_mrz_image"

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGIT = "0123456789"
# Alnum pool for the personal-number field: G/D/K/M/L/R oversampled ~3x
# (measured-hard line-2 letter classes, see the pers drawing below).
PERS_ALNUM_POOL = "".join(
    ch * (3 if ch in "GDKMLR" else 1) for ch in ALNUM)


def check_digit_of(seg: str) -> str:
    """ICAO 9303 mod-7/3/1 weighted check digit. '<' = 0."""
    w = [7, 3, 1]
    total = 0
    for i, c in enumerate(seg):
        v = (ord(c) - 48) if '0' <= c <= '9' else (
            (ord(c) - 55) if 'A' <= c <= 'Z' else 0)
        total += v * w[i % 3]
    return str(total % 10)


def build_line2(pn: str, nat: str, bd: str, sex: str, ex: str,
                pers: str) -> str:
    """Assemble an ICAO-valid TD3 line 2 from its fields.

    Per ICAO 9303 the composite check digit covers the four fields that
    carry their own check digits (document number, date of birth, date
    of expiry, personal number) and excludes nationality and sex."""
    l2 = (pn + check_digit_of(pn) + nat + bd + check_digit_of(bd) +
          sex + ex + check_digit_of(ex) + pers + check_digit_of(pers))
    assert len(l2) == 43, l2
    comp = l2[0:10] + l2[13:20] + l2[21:43]
    l2 += check_digit_of(comp)
    assert len(l2) == 44
    return l2


def build_line1(issuer: str, surname: str, given: str) -> str:
    """Assemble an ICAO-valid TD3 line 1: P< + issuer(3) + name."""
    name = surname + "<<" + given if given else surname + "<"
    assert len(name) <= 39, name
    l1 = ("P<" + issuer + name).ljust(44, "<")
    assert len(l1) == 44
    return l1


def icao_valid(l1: str, l2: str) -> bool:
    """Structural validation used as a generation-time assert."""
    if len(l1) != 44 or len(l2) != 44:
        return False
    if not all(c in ALPHA + "<" for c in l1):
        return False
    if l2[9] != check_digit_of(l2[0:9]) or l2[19] != check_digit_of(l2[13:19]) \
       or l2[27] != check_digit_of(l2[21:27]) or l2[42] != check_digit_of(l2[28:42]) \
       or l2[43] != check_digit_of(l2[0:10] + l2[13:20] + l2[21:43]):
        return False
    if not (l2[13:19] + l2[21:27]).isdigit():
        return False
    if l2[20] not in "MF<":
        return False
    return True


# Base document "personalities" (same identities as the historical
# corpus, rebuilt as structurally valid TD3 records -- the originals
# were missing the birth/expiry check digits).
BASE_DOCS = [
    # (surname, given, issuer, pn, nat, bd, sex, ex, pers)
    ("ERIKSSON", "ANNA MARIA", "UTO", "L898902C3", "UTO",
     "690806", "F", "940623", "ZE184226B<<<<<"),
    ("WANG", "LI", "CHN", "E12345678", "CHN",
     "900101", "M", "301231", "1234567890<<<<"),
    ("SMITH", "JOHN", "USA", "123456789", "USA",
     "850215", "M", "250215", "A1B2C3D4E5<<<<"),
    ("MULLER", "HANS", "DEU", "C01X00T47", "DEU",
     "000101", "<", "320101", "00000000000000"),
    ("OBRIEN", "PATRICK", "GBR", "GBR123456", "IRL",
     "990101", "F", "291231", "DOCUMENT0123<<"),
]


def base_pairs():
    pairs = []
    for (surname, given, issuer, pn, nat, bd, sex, ex, pers) in BASE_DOCS:
        # "ANNA MARIA" is one field on the real document; ICAO renders
        # the space as a filler '<'.
        l1 = build_line1(issuer, surname, given.replace(" ", "<"))
        l2 = build_line2(pn, nat, bd, sex, ex, pers)
        assert icao_valid(l1, l2), (l1, l2)
        pairs.append((l1, l2))
    return pairs


def iso3_pool() -> list[str]:
    """Real ISO-3166-1 alpha-3 codes, parsed from the tool's own table.

    Single source of truth: the same k_iso3 that mrz_code3_plausible()
    (src/mrz_ocr_cnn.c) checks against.  Generating nationalities from
    this pool keeps the corpus inside the priors the production pipeline
    correctly applies to real documents.  Falls back to a short built-in
    list if the header cannot be parsed."""
    import re
    try:
        src = (ROOT / "src" / "mrz_ocr_cnn.c").read_text()
        m = re.search(r'static const char k_iso3\[\]\s*=\s*"([^"]+)"', src)
        if m:
            s = m.group(1)
            return [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]
    except OSError:
        pass
    return ["AFG", "CHN", "DEU", "GBR", "USA", "JPN", "KOR", "FRA", "ITA",
            "ESP", "RUS", "IND", "BRA", "CAN", "AUS", "UTO"]


def random_date(rnd: random.Random, lo_year: int, hi_year: int) -> str:
    """Calendar-valid YYMMDD.  Feb 29 only on leap years."""
    y = rnd.randint(lo_year, hi_year)
    m = rnd.randint(1, 12)
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    dmax = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return f"{y % 100:02d}{m:02d}{rnd.randint(1, dmax):02d}"


def random_pair(rnd: random.Random, plausible: bool = True) -> tuple[str, str]:
    """ICAO-valid random record: alpha-only line 1, checked line 2.

    plausible=True (default) draws every field from a real-document
    distribution: nationality/issuer from the ISO-3166 pool (they differ
    on ~20% of real passports), calendar-valid dates, digit-dominant
    document numbers ('<'-filled when short), and a personal-number field
    that mixes letters, digits and filler the way real optional data
    does.  The previous fully-random drawing produced month-43 birthdays
    and nationalities no state issues, which collide with the pipeline's
    structural priors (issuer fix / strict ICAO / checksum beam) -- the
    benchmark then measured that collision, not recognition."""
    if plausible:
        pool = iso3_pool()
        issuer = rnd.choice(pool)
        nat = issuer if rnd.random() < 0.8 else rnd.choice(pool)
        surname = "".join(rnd.choice(ALPHA) for _ in range(rnd.randint(3, 12)))
        given = "".join(rnd.choice(ALPHA) for _ in range(rnd.randint(0, 15)))
        l1 = build_line1(issuer, surname, given)
        r = rnd.random()
        if r < 0.6:
            pn = "".join(rnd.choice(DIGIT) for _ in range(9))
        elif r < 0.8:
            pn = "".join(rnd.choice(ALNUM) for _ in range(9))
        else:
            k = rnd.randint(7, 8)
            pn = ("".join(rnd.choice(ALNUM) for _ in range(k))).ljust(9, "<")
        bd = random_date(rnd, 1940, 2009)
        ex = random_date(rnd, 2015, 2039)
        # Personal-number field.  Real prior: mostly '<'-filled or numeric;
        # letters appear in the alnum tail.  G/D/K/M/L/R are the measured-
        # hard classes in this zone (line-2 letter cells err 14.2% vs
        # digits 5.6% on 106 real product bands, 2026-11-05), so the alnum
        # drawing samples them ~3x -- WITHOUT changing the filler/digit
        # shares, which is what keeps the '<'/'0' checksum-equivalence
        # prior realistic.
        u = rnd.random()
        if u < 0.35:
            pers = "<" * 14
        elif u < 0.60:
            pers = "".join(rnd.choice(DIGIT) for _ in range(14))
        elif u < 0.90:
            pers = "".join(rnd.choice(PERS_ALNUM_POOL) for _ in range(14))
        else:
            pers = ("".join(rnd.choice(PERS_ALNUM_POOL)
                            for _ in range(rnd.randint(6, 12))
                            )).ljust(14, "<")
        l2 = build_line2(pn, nat, bd, rnd.choice("MF<"), ex, pers)
        assert icao_valid(l1, l2), (l1, l2)
        return l1, l2
    issuer = "".join(rnd.choice(ALPHA) for _ in range(3))
    surname = "".join(rnd.choice(ALPHA) for _ in range(rnd.randint(3, 12)))
    given = "".join(rnd.choice(ALPHA) for _ in range(rnd.randint(0, 15)))
    l1 = build_line1(issuer, surname, given)
    pn = "".join(rnd.choice(ALNUM) for _ in range(9))
    nat = "".join(rnd.choice(ALPHA) for _ in range(3))
    bd = "".join(rnd.choice(DIGIT) for _ in range(6))
    ex = "".join(rnd.choice(DIGIT) for _ in range(6))
    pers = "".join(rnd.choice(ALNUM + "<") for _ in range(14))
    l2 = build_line2(pn, nat, bd, rnd.choice("MF<"), ex, pers)
    assert icao_valid(l1, l2), (l1, l2)
    return l1, l2


# Image variants: (scale, hgap, vgap, skew, noise, rot, blur, illum, invert)
VARIANTS_CLEAN = [
    (4, 2, 6, 0, 0.00, 0.0, 0, 0, 0),
    (4, 2, 6, 0, 0.01, 0.0, 0, 0, 0),
    (4, 2, 6, 2, 0.00, 0.0, 0, 0, 0),
    (4, 2, 6, -2, 0.00, 0.0, 0, 0, 0),
    (5, 3, 8, 0, 0.00, 0.0, 0, 0, 0),
    (5, 3, 8, 0, 0.02, 0.0, 0, 0, 0),
    (6, 2, 10, 0, 0.00, 0.0, 0, 0, 0),
    (6, 3, 10, 1, 0.01, 0.0, 0, 0, 0),
    (4, 4, 8, 0, 0.00, 0.0, 0, 0, 0),
    (3, 2, 6, 0, 0.00, 0.0, 0, 0, 0),
    (4, 2, 6, 0, 0.03, 0.0, 0, 0, 0),
    (4, 2, 6, 0, 0.05, 0.0, 0, 0, 0),
]

VARIANTS_REALISTIC = [
    # scanner tilt + defocus + shading
    (4, 2, 6, 0, 0.01, 1.0, 1, 30, 0),
    (4, 2, 6, 1, 0.02, -1.5, 1, 0, 0),
    (5, 3, 8, 0, 0.03, 2.0, 2, 50, 0),
    (6, 2, 10, 2, 0.00, -2.5, 0, 20, 0),
    (3, 2, 6, 0, 0.05, 0.0, 1, 40, 0),
    (4, 4, 8, -1, 0.01, 3.0, 2, 60, 0),
    (5, 2, 8, 0, 0.04, -1.0, 1, 10, 0),
    (4, 2, 6, -2, 0.02, 1.5, 0, 0, 0),
    (4, 2, 6, 0, 0.03, -3.0, 2, 30, 0),
    (6, 3, 10, 1, 0.01, 0.5, 1, 0, 1),    # inverted (white-on-black)
    (4, 2, 6, 0, 0.02, 2.5, 1, 80, 0),
    (5, 3, 8, 2, 0.05, -0.5, 2, 0, 0),
]

# Training-only, harder-than-eval variants targeting the measured weak axes
# (corpus_eval_v2 axis report, post-retrain baseline: blur=2 full 35.4%,
# illum>=40 full 32.5%, scale=3 full 49.2%).  The eval corpus never carries
# these, so the axis buckets on eval_v2 honestly measure transfer.
#
# Measured effect (2026-11-05, corpus_train_v3, CNN retrained on the v3
# slices): the "letter -> 'B'" attractor on real photos fell 95 -> 60 cells
# and "letter -> '<'" fell 90 -> 69, i.e. the degradation axis was the real
# defect, not the class prior.  End-to-end product metrics were unchanged
# (official PASS 32/verified 38 on the 44-photo set, identical to the
# baseline), so this is infrastructure for the next round, not a win by
# itself.
VARIANTS_DEGRADED = [
    (3, 2, 6, 0, 0.02, 1.5, 2, 40, 0),    # scale3 + blur2 + illum40 + rot
    (3, 2, 6, 1, 0.03, -2.0, 1, 60, 0),   # scale3 + illum60
    (4, 2, 6, 0, 0.02, 2.5, 3, 40, 0),    # blur3 (beyond the eval range)
    (4, 2, 6, -1, 0.04, -2.5, 2, 80, 0),  # illum80 + blur2
    (4, 4, 8, 1, 0.01, 3.0, 2, 50, 0),    # skew + rot3 + blur2
    (5, 3, 8, 0, 0.03, 1.0, 3, 60, 0),    # blur3 + illum60
    (4, 2, 6, 0, 0.05, 0.0, 2, 30, 0),    # heavy noise + blur2
    (6, 3, 10, -1, 0.02, -1.0, 2, 40, 0), # scale6 + blur2 + illum40
]


def render(out_dir: Path, name: str, l1: str, l2: str, v: tuple,
           noise_seed: int, font: str = "bitmap", aa: bool = False) -> Path:
    scale, hgap, vgap, skew, noise, rot, blur, illum, invert = v
    p = out_dir / f"{name}.ppm"
    cmd = [str(GEN), str(p), l1, l2,
           f"scale={scale}", f"hgap={hgap}", f"vgap={vgap}",
           f"skew={skew}", f"noise={noise}", f"noise-seed={noise_seed}"]
    if rot:
        cmd.append(f"rot={rot}")
    if blur:
        cmd.append(f"blur={blur}")
    if illum:
        cmd.append(f"illum={illum}")
    if invert:
        cmd.append("invert=1")
    if font == "ocrb":
        cmd.append("font=ocrb")
    if aa:
        cmd.append("aa=1")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true",
                    help="build the frozen evaluation corpus")
    ap.add_argument("--train", action="store_true",
                    help="build the (larger) training corpus")
    ap.add_argument("--out", required=True, help="output corpus directory")
    ap.add_argument("--seed", type=int, required=True,
                    help="corpus seed (ground truth + variant + noise)")
    ap.add_argument("--extra-pairs", type=int, default=None,
                    help="override the number of random records")
    ap.add_argument("--aa", action="store_true",
                    help="anti-aliased strokes (grey edges, like a scan)")
    ap.add_argument("--font", choices=("bitmap", "ocrb"), default="bitmap",
                    help="glyph shapes: bitmap = the hand-simplified bank "
                         "in src/template.c (historical default), ocrb = the "
                         "real OCR-B TrueType rendering (tools/ocrb_glyphs.h)")
    ap.add_argument("--degraded", action="store_true",
                    help="also render the training-only VARIANTS_DEGRADED "
                         "set (hard blur/illum/scale combos) for every "
                         "document; eval corpora must not pass this")
    ap.add_argument("--legacy-random", action="store_true",
                    help="draw fully-random fields (random 3-letter "
                         "nationalities, non-calendar dates) instead of the "
                         "plausible real-document distribution.  Kept only "
                         "to reproduce the pre-2026-11-05 corpora.")
    a = ap.parse_args()
    if a.eval == a.train:
        ap.error("pick exactly one of --eval / --train")

    if not GEN.exists():
        print("gen_mrz_image not built -- run cmake --build build first",
              file=sys.stderr)
        return 1

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rnd_master = random.Random(a.seed)

    records = []
    i = 0

    def add(l1, l2, v, pair_tag, var_idx, profile):
        nonlocal i
        i += 1
        name = f"img_{i:04d}_{pair_tag}_v{var_idx}"
        noise_seed = a.seed * 100000 + i * 7919
        render(out_dir, name, l1, l2, v, noise_seed, a.font, a.aa)
        scale, hgap, vgap, skew, noise, rot, blur, illum, invert = v
        records.append({
            "id": name, "profile": profile,
            "line1": l1, "line2": l2,
            "scale": scale, "hgap": hgap, "vgap": vgap,
            "skew": skew, "noise": noise, "rot": rot,
            "blur": blur, "illum": illum, "invert": invert,
            "image": f"{name}.ppm",
        })

    base = base_pairs()

    if a.eval:
        # Frozen evaluation corpus.  The BASE_DOCS identities live in the
        # TRAINING corpus, so they must NOT appear here: when the seed-2002
        # corpus carried them (plus the identical clean renders), 60 of 1380
        # eval images leaked into training and the seen texts scored ~88%
        # full vs ~13% for unseen ones -- the headline number was then a
        # leak measurement.  Eval is plausible-random docs only, drawn from
        # a seed disjoint from the training corpus's text seeds.
        n_extra = a.extra_pairs if a.extra_pairs is not None else 60
        for p_idx in range(n_extra):
            l1, l2 = random_pair(random.Random(a.seed * 31 + p_idx),
                                 plausible=not a.legacy_random)
            for v_idx, v in enumerate(VARIANTS_CLEAN):
                add(l1, l2, v, f"x{p_idx}", v_idx, "clean")
            for v_idx, v in enumerate(VARIANTS_REALISTIC):
                add(l1, l2, v, f"x{p_idx}", v_idx, "realistic")
    else:
        # Training corpus: base docs + more identities, heavier realistic
        # mix.  Eval texts use seed*31, training uses seed*131, so the two
        # random-document pools cannot collide for distinct corpus seeds.
        for p_idx, (l1, l2) in enumerate(base):
            for v_idx, v in enumerate(VARIANTS_CLEAN):
                add(l1, l2, v, f"b{p_idx}", v_idx, "clean")
            for v_idx, v in enumerate(VARIANTS_REALISTIC):
                add(l1, l2, v, f"b{p_idx}", v_idx, "realistic")
            if a.degraded:
                for v_idx, v in enumerate(VARIANTS_DEGRADED):
                    add(l1, l2, v, f"b{p_idx}", v_idx, "degraded")
        n_extra = a.extra_pairs if a.extra_pairs is not None else 120
        for p_idx in range(n_extra):
            l1, l2 = random_pair(random.Random(a.seed * 131 + p_idx),
                                 plausible=not a.legacy_random)
            for v_idx, v in enumerate(VARIANTS_CLEAN):
                add(l1, l2, v, f"x{p_idx}", v_idx, "clean")
            for v_idx, v in enumerate(VARIANTS_REALISTIC):
                add(l1, l2, v, f"x{p_idx}", v_idx, "realistic")
            if a.degraded:
                for v_idx, v in enumerate(VARIANTS_DEGRADED):
                    add(l1, l2, v, f"x{p_idx}", v_idx, "degraded")

    out = out_dir / "corpus.json"
    out.write_text(json.dumps({"version": 2, "seed": a.seed,
                               "font": a.font,
                               "aa": bool(a.aa),
                               "plausible": not a.legacy_random,
                               "count": len(records),
                               "records": records}, indent=2))
    n_clean = sum(1 for r in records if r["profile"] == "clean")
    print(f"generated {len(records)} test images "
          f"({n_clean} clean / {len(records) - n_clean} realistic) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
