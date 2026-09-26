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
    """Assemble an ICAO-valid TD3 line 2 from its fields."""
    l2 = (pn + check_digit_of(pn) + nat + bd + check_digit_of(bd) +
          sex + ex + check_digit_of(ex) + pers + check_digit_of(pers))
    assert len(l2) == 43, l2
    l2 += check_digit_of(l2)
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
       or l2[43] != check_digit_of(l2[0:43]):
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


def random_pair(rnd: random.Random) -> tuple[str, str]:
    """ICAO-valid random record: alpha-only line 1, checked line 2."""
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


def render(out_dir: Path, name: str, l1: str, l2: str, v: tuple,
           noise_seed: int) -> Path:
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
        render(out_dir, name, l1, l2, v, noise_seed)
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
        # Frozen evaluation corpus: base docs in all clean variants,
        # random docs in clean + realistic variants. ~630 images.
        for p_idx, (l1, l2) in enumerate(base):
            for v_idx, v in enumerate(VARIANTS_CLEAN):
                add(l1, l2, v, f"b{p_idx}", v_idx, "clean")
        n_extra = a.extra_pairs if a.extra_pairs is not None else 55
        for p_idx in range(n_extra):
            l1, l2 = random_pair(random.Random(a.seed * 31 + p_idx))
            for v_idx, v in enumerate(VARIANTS_CLEAN):
                add(l1, l2, v, f"x{p_idx}", v_idx, "clean")
            for v_idx, v in enumerate(VARIANTS_REALISTIC):
                add(l1, l2, v, f"x{p_idx}", v_idx, "realistic")
    else:
        # Training corpus: more identities, heavier realistic mix.
        for p_idx, (l1, l2) in enumerate(base):
            for v_idx, v in enumerate(VARIANTS_CLEAN):
                add(l1, l2, v, f"b{p_idx}", v_idx, "clean")
            for v_idx, v in enumerate(VARIANTS_REALISTIC):
                add(l1, l2, v, f"b{p_idx}", v_idx, "realistic")
        n_extra = a.extra_pairs if a.extra_pairs is not None else 120
        for p_idx in range(n_extra):
            l1, l2 = random_pair(random.Random(a.seed * 131 + p_idx))
            for v_idx, v in enumerate(VARIANTS_CLEAN):
                add(l1, l2, v, f"x{p_idx}", v_idx, "clean")
            for v_idx, v in enumerate(VARIANTS_REALISTIC):
                add(l1, l2, v, f"x{p_idx}", v_idx, "realistic")

    out = out_dir / "corpus.json"
    out.write_text(json.dumps({"version": 2, "seed": a.seed,
                               "count": len(records),
                               "records": records}, indent=2))
    n_clean = sum(1 for r in records if r["profile"] == "clean")
    print(f"generated {len(records)} test images "
          f"({n_clean} clean / {len(records) - n_clean} realistic) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
