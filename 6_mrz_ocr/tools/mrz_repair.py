"""ICAO check-digit guided repair of a mis-read MRZ line 2.

The character heads read the optional-data field of TD3 line 2 with a
strong digit bias: a printed "O" comes out as "0", "S" as "6", "Z" as
"1", "G" as "Q" ...  Every one of those characters carries its own ICAO
7-3-1 check digit, so the ambiguity is resolvable without a better
model: flip the candidates that make the check digit match.

Only repairs that satisfy the failing field's own check digit *and* the
composite check digit are accepted, and only when exactly one candidate
survives.  A line that already passes every check is never touched.
"""
from __future__ import annotations

# Classic 8x12 OCR-B shape confusions, as (char -> alternatives).
_ALTS = {
    "0": "ODQ",
    "1": "ILJZ",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "GSB",
    "7": "T",
    "8": "B",
    "9": "GQ",
    "A": "4",
    "B": "86",
    "C": "GO0",
    "D": "0",
    "E": "F",
    "F": "E",
    "G": "69QCO",
    "I": "1",
    "J": "1",
    "L": "1",
    "O": "0DQCG",
    "Q": "09G",
    "S": "56",
    "T": "7",
    "U": "V",
    "V": "UY",
    "Y": "V",
    "Z": "21",
}
_ALT_OF = {c: tuple(a) for c, a in _ALTS.items()}
_ALT_BACK = {}
for _c, _as in _ALT_OF.items():
    for _a in _as:
        _ALT_BACK.setdefault(_a, set()).add(_c)
        _ALT_BACK.setdefault(_c, set()).add(_a)
for _c in list(_ALT_BACK):
    _ALT_BACK[_c] = tuple(sorted(_ALT_BACK[_c]))

# field start/end (half open) and the index of its check digit
_FIELDS = ((0, 9, 9), (13, 19, 19), (21, 27, 27), (28, 42, 42))
_MAX_COMBOS = 4096


def _weight_map():
    w = {}
    for i in range(10):
        w[str(i)] = i
    for i in range(26):
        w[chr(ord("A") + i)] = 10 + i
    return w


_W = _weight_map()


def check_digit(s: str) -> str:
    """ICAO 9303 7-3-1 weighted check digit ('<' counts as 0)."""
    total = 0
    for i, ch in enumerate(s):
        total += _W.get(ch, 0) * (7, 3, 1)[i % 3]
    return str(total % 10)


def _composite(line2: str) -> str:
    return line2[0:10] + line2[13:20] + line2[21:28] + line2[28:43]


def bad_checks(line2: str) -> list:
    """Names of the TD3 line2 check digits that do not match."""
    if len(line2) != 44:
        return ["length"]
    bad = []
    for start, end, ck in _FIELDS:
        if check_digit(line2[start:end]) != line2[ck]:
            bad.append({(0, 9, 9): "document_number", (13, 19, 19): "birth_date",
                        (21, 27, 27): "expiry_date",
                        (28, 42, 42): "personal_number"}[(start, end, ck)])
    if check_digit(_composite(line2)) != line2[43]:
        bad.append("composite")
    return bad


def _variants(field: str):
    """All substitutions of the ambiguous characters of one field."""
    slots = []
    for i, ch in enumerate(field):
        alts = _ALT_BACK.get(ch)
        if alts:
            slots.append((i, alts))
    combos = 1
    for _, alts in slots:
        combos *= len(alts)
    if combos > _MAX_COMBOS:
        return []
    out = [""]
    for i, alts in slots:
        nxt = []
        for prefix in out:
            for a in alts:
                nxt.append(prefix[:i] + a + prefix[i + 1:])
        out = nxt
    return out


def _repair_field(line2: str, start: int, end: int, ck: int) -> list:
    """Candidate line2 strings whose field check digit matches."""
    field = line2[start:end]
    want = line2[ck]
    out = []
    for var in _variants(field):
        if check_digit(var) != want:
            continue
        cand = line2[:start] + var + line2[end:]
        if check_digit(_composite(cand)) == cand[43]:
            out.append(cand)
    return out


def repair_line2(line2: str) -> tuple:
    """Return (repaired_line2_or_None, info_dict).

    line2 is returned unchanged (None) when it already passes every check
    or when the repair is not unique.
    """
    info = {"changed": False, "fields": [], "candidates": 0}
    if not line2 or len(line2) != 44:
        return None, info
    bad = bad_checks(line2)
    if not bad:
        return None, info
    names = {(0, 9, 9): "document_number", (13, 19, 19): "birth_date",
             (21, 27, 27): "expiry_date", (28, 42, 42): "personal_number"}
    failing = [f for f in _FIELDS if names[f] in bad]
    cands = [line2]
    for f in failing:
        nxt = []
        for c in cands:
            nxt.extend(_repair_field(c, *f))
        if not nxt:
            return None, info
        cands = nxt
        info["fields"].append(names[f])
        if len(cands) > 64:
            return None, info
    info["candidates"] = len(cands)
    if len(cands) != 1:
        return None, info
    fixed = cands[0]
    if bad_checks(fixed):
        return None, info
    info["changed"] = True
    return fixed, info
