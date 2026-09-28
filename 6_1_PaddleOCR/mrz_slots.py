#!/usr/bin/env python3
"""mrz_slots -- ICAO 9303 TD3 slot geometry + OCR-B confusion classes.

Single source of truth for the passport MRZ (TD3) layout, shared by
paddle_v6 / paddle_local / paddle_ocr_tool. Historically each module
carried its own copy of these tables, and the copies disagreed — most
notably a Line-1 table that marked positions 13..43 as digit-only,
which corrupted every name it touched ("BROWN" -> "BR0WN").

TD3 layout (44 chars per line):

    line1  [0] doc type alpha      [1] '<'          [2..4] issuing alpha
           [5..43] name alpha
    line2  [0..8] passport_no alnum    [9] cd
           [10..12] nationality alpha  [13..18] dob   [19] cd
           [20] sex M/F/<              [21..26] expiry [27] cd
           [28..41] personal_no alnum  [42] cd        [43] composite cd

The confusion classes below are the empirically observed OCR-B misread
pairs on the corpus (GT->pred): S->5, O->0, W->H, Z->2, I->1, Q->0,
U->0, B->8, M->N, M->H, N->H, W->A, R->8, L->C, W->4, T->1, V->0,
R->B, W->U, K->B, O->D, G->6, U->D, X->8. Each pair is registered in
both directions so a lookup answers "this predicted char could really
be what?".
"""
from __future__ import annotations

LINE_LEN = 44
ICAO = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
DOC_TYPES = set("PIVACDR")
SEX = set("MF<")

# Slot domains (positions not listed are alnum / fixed).
ALPHA_L1 = {0} | set(range(2, LINE_LEN))
DIGIT_L1 = set()
ALPHA_L2 = {10, 11, 12}
DIGIT_L2 = {9} | set(range(13, 20)) | set(range(21, 28)) | {42, 43}

# (a, b, cost): predicted `a` may really be `b` and vice versa.
# cost 1 = classic OCR-B glyph ambiguity, 2 = plausible visual mixup.
_PAIRS = (
    ("0", "O", 1), ("0", "Q", 2), ("0", "U", 2), ("0", "D", 2),
    ("0", "V", 2), ("0", "G", 2),
    ("1", "I", 1), ("1", "L", 1), ("1", "T", 1), ("1", "7", 1),
    ("5", "S", 1),
    ("2", "Z", 1),
    ("8", "B", 1), ("8", "R", 1), ("8", "X", 2), ("8", "3", 2),
    ("6", "G", 1),
    ("4", "W", 2), ("4", "A", 2),
    ("W", "H", 1), ("W", "A", 2), ("W", "U", 2), ("W", "M", 2),
    ("H", "N", 1), ("H", "M", 1),
    ("M", "N", 1),
    ("R", "B", 1),
    ("L", "C", 1),
    ("K", "B", 1), ("K", "X", 2),
    ("D", "U", 1), ("D", "O", 1),
    ("U", "V", 1), ("U", "O", 2),
    ("V", "Y", 2),
    ("Q", "O", 1),
    ("T", "I", 2), ("T", "7", 2),
    ("X", "K", 2),
)


def _build_confuse():
    table = {}
    for a, b, cost in _PAIRS:
        for src, dst in ((a, b), (b, a)):
            cur = table.setdefault(src, {})
            if dst not in cur or cost < cur[dst]:
                cur[dst] = cost
    return table


CONFUSE = _build_confuse()


def confusions(c):
    """Candidates for predicted char `c`, cheapest first.

    Returns [(candidate, cost), ...]; empty when `c` has no known
    confusion. Ordering is stable for equal costs (declaration order).
    """
    d = CONFUSE.get(c, {})
    return sorted(d.items(), key=lambda kv: kv[1])


def digit_for(c):
    """Cheapest digit this (mis-read) char could be; (digit, cost) or None."""
    for cand, cost in confusions(c):
        if cand.isdigit():
            return cand, cost
    return None


def alpha_for(c):
    """Cheapest letter this (mis-read) char could be; (letter, cost) or None."""
    for cand, cost in confusions(c):
        if cand.isalpha():
            return cand, cost
    return None


def enforce_slots(line, alpha_set, digit_set):
    """Swap OCR-B confusions so the line satisfies the slot domains.

    A digit-only slot holding a letter is swapped to its cheapest digit
    confusion (O->0, S->5, ...); an alpha-only slot holding a digit is
    swapped to its cheapest letter confusion (0->O, 1->I, ...). '<' and
    free (alnum) positions are untouched.
    """
    out = list(line)
    for i, ch in enumerate(out):
        if ch == "<" or not ch:
            continue
        if i in digit_set and ch.isalpha():
            cand = digit_for(ch)
            if cand:
                out[i] = cand[0]
        elif i in alpha_set and ch.isdigit():
            cand = alpha_for(ch)
            if cand:
                out[i] = cand[0]
    return "".join(out)


def slot_violations(line, alpha_set, digit_set):
    """Count positions that still violate their slot domain."""
    n = 0
    for i, ch in enumerate(line):
        if ch == "<" or not ch:
            continue
        if i in digit_set and not ch.isdigit():
            n += 1
        elif i in alpha_set and not ch.isalpha():
            n += 1
    return n


if __name__ == "__main__":
    # Slot domains match the corrected TD3 layout: Line 1 is alpha
    # everywhere except the '<' at position 1.
    assert 13 in ALPHA_L1 and not DIGIT_L1
    assert 5 in ALPHA_L1 and 21 in ALPHA_L1
    assert 13 in DIGIT_L2 and 19 in DIGIT_L2 and 43 in DIGIT_L2
    assert 10 in ALPHA_L2 and 10 not in DIGIT_L2
    # Confusion classes are bidirectional.
    assert digit_for("O") == ("0", 1)
    assert digit_for("S") == ("5", 1)
    assert alpha_for("0") == ("O", 1)
    assert alpha_for("5") == ("S", 1)
    # The historical name-corruption case must now survive.
    bad = "P<GBRBROWN<<JOHN<MICHAEL<SMITH<<<<<<<<<<<<<<<<<<<<<<<<"
    fixed = enforce_slots(bad, ALPHA_L1, DIGIT_L1)
    assert "BROWN" in fixed and "JOHN" in fixed, fixed
    assert fixed == bad  # no digit/alpha slot under Line 1 at all
    print("mrz_slots self-test OK")
