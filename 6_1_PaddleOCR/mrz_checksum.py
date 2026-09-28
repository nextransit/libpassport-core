#!/usr/bin/env python3
"""mrz_checksum -- ICAO 9303 TD3 Mod-10 check-digit primitives + repair.

The TD3 (standard passport) MRZ layout has two 44-character lines. Line 1
has NO check digits; the four local check digits and the composite check
digit all live in Line 2:

    Position  | Field              | Length | Notes
    --------  | -----------------  | ------  | -----
    0..8      | passport_no        |  9      | check digit at pos 9
    9         | check digit (cd_1) |  1      |
    10..12    | nationality        |  3      | alpha-3
    13..18    | date_of_birth      |  6      | YYMMDD, check digit at pos 19
    19        | check digit (cd_2) |  1      |
    20        | sex                |  1      | M/F/<
    21..26    | date_of_expiry     |  6      | YYMMDD, check digit at pos 27
    27        | check digit (cd_3) |  1      |
    28..41    | personal_no        | 14      | check digit at pos 42
    42        | check digit (cd_4) |  1      |
    43        | composite_check    |  1      | check over all four fields above

    The composite check digit is computed over (passport_no + cd_1) +
    (DOB + cd_2) + (expiry + cd_3) + (personal_no + cd_4), i.e. the
    39-char string s[0:10] + s[13:20] + s[21:28] + s[28:43], with
    weights cycling 7, 3, 1 across the whole 39.

Mod-10 algorithm:
    1. Map each character to a digit:
         '0'..'9' -> 0..9
         'A'..'Z' -> 10..35
         '<'      -> 0
    2. Multiply each digit by a weight cycling 7, 3, 1.
    3. Sum all weighted digits.
    4. check_digit = sum mod 10.

Repair strategy (line2_correct_with_checksums): for each failing field,
enumerate plausible OCR-B confusions (mrz_slots.CONFUSE) at every position
— single substitutions, then restricted two-substitution combinations —
plus a "the data is right, the check glyph is misread" candidate. The
per-field candidates are then combined globally (itertools.product over
the four fields, each capped) and ranked by an empirical edit score:
log10 of the smoothed (GT char, OCR char) count from mrz_confusions
(measured on the training corpus), minus a small per-cost penalty, plus
a bonus when the composite check verifies without a rewrite. This
replaces the old first-match-wins solver that (a) could never fix
digit->digit misreads such as 3<->8, (b) never questioned a misread
check digit, and (c) silently made the wrong choice when two fields were
both broken. The prior is essential because Mod-10 alone cannot
discriminate: on passport_no / personal_no (weight phase aligned with
the composite check) several coincidental edits verify just as well as
the true confusion.

repair_line2(raw) additionally aligns near-44 inputs: a 43-char line gets
a checksum-guided single insertion (44 positions x ICAO chars), a
45-char line a single deletion; the result must be fully check-valid.

This module exposes:
    char_to_digit(c)              -- ICAO char -> 0..35
    digit_to_char(d)              -- 0..35 -> ICAO char
    compute_check_digit(field)    -- field str -> 1-char check
    verify_mod10(field, check)    -- bool
    solve_ambiguity(...)          -- classic single-slot solver
    line2_correct_with_checksums(line2)  -- global Mod-10 repair
    align_line2(raw)              -- 43/44/45 chars -> 44-char valid line
    repair_line2(raw)             -- align + correct, returns full report
"""
from __future__ import annotations
import itertools
import math
from typing import Iterable, Optional

from mrz_slots import ALPHA_L2, DIGIT_L2, ICAO, SEX, confusions
from mrz_confusions import prior

# Weight cycle: 7, 3, 1, 7, 3, 1, ...
WEIGHTS = (7, 3, 1)

# ICAO character -> digit value (0..35)
_CHAR_TO_DIGIT = {}
for _i in range(10):
    _CHAR_TO_DIGIT[str(_i)] = _i
for _i in range(26):
    _CHAR_TO_DIGIT[chr(ord("A") + _i)] = 10 + _i
_CHAR_TO_DIGIT["<"] = 0

# Reverse map: digit value -> ICAO character (we only emit digits 0-9
# for check-digit math, since checks are always digit positions).
_DIGIT_TO_CHAR = {v: k for k, v in _CHAR_TO_DIGIT.items()}


def char_to_digit(c: str) -> int:
    """Map a single ICAO character to its numeric value (0..35).

    '0'..'9' -> 0..9, 'A'..'Z' -> 10..35, '<' -> 0.
    Anything else raises ValueError -- caller is responsible for
    ICAO whitelist filtering.
    """
    try:
        return _CHAR_TO_DIGIT[c]
    except KeyError:
        raise ValueError(f"non-ICAO character: {c!r}")


def digit_to_char(d: int) -> str:
    """Reverse of char_to_digit for the 0..9 range (mod-10 output)."""
    if not 0 <= d <= 9:
        raise ValueError(f"check digit out of range: {d}")
    return str(d)


def compute_check_digit(field: str) -> str:
    """Compute the Mod-10 check digit for an arbitrary-length field."""
    total = 0
    for i, c in enumerate(field):
        total += char_to_digit(c) * WEIGHTS[i % 3]
    return digit_to_char(total % 10)


def verify_mod10(field: str, check: str) -> bool:
    """True iff `check` is the correct Mod-10 check digit for `field`."""
    if not field or not check:
        return False
    try:
        return compute_check_digit(field) == check
    except ValueError:
        return False


def solve_ambiguity(
    field_with_marker: str,
    check: str,
    candidates: Iterable[str],
    marker: str = "?",
) -> Optional[str]:
    """Find the candidate that satisfies Mod-10 against `check`.

    `field_with_marker` must contain exactly one occurrence of `marker`
    (the ambiguous slot); that slot is replaced with each candidate
    in turn. Returns the candidate that produces a matching check
    digit, or None if zero or >1 candidates match (no unique solution).
    """
    if field_with_marker.count(marker) != 1:
        return None
    matches = []
    for cand in candidates:
        candidate_field = field_with_marker.replace(marker, cand, 1)
        try:
            if verify_mod10(candidate_field, check):
                matches.append(cand)
        except ValueError:
            continue
    if len(matches) == 1:
        return matches[0]
    return None


# ----------------------------------------------------------------------
# Line 2 repair: global Mod-10 search across the four checked fields.
# ----------------------------------------------------------------------

LINE_LEN = 44

# (label, field_start, field_end, check_pos) for the four local checks.
_LINE2_FIELDS = (
    ("passport_no", 0, 9, 9),
    ("date_of_birth", 13, 19, 19),
    ("date_of_expiry", 21, 27, 27),
    ("personal_no", 28, 42, 42),
)

# Composite spans s[0:10] + s[13:20] + s[21:28] + s[28:43].
_COMPOSITE_SLICES = ((0, 10), (13, 20), (21, 28), (28, 43))


def _composite(s: str) -> str:
    return "".join(s[a:b] for a, b in _COMPOSITE_SLICES)


def _line2_valid(s: str) -> bool:
    """True iff all four local checks and the composite check verify."""
    if len(s) != LINE_LEN:
        return False
    for _label, start, end, cpos in _LINE2_FIELDS:
        if not verify_mod10(s[start:end], s[cpos]):
            return False
    return verify_mod10(_composite(s), s[43])


def _confusion_cost(ch: str, target: str) -> int:
    """Cost of explaining `ch` as `target` (3 = no known confusion)."""
    for cand, cost in confusions(ch):
        if cand == target:
            return cost
    return 3


# Weight of one cost unit against the log-prior evidence of an edit.
_EDIT_PENALTY = 0.5

# Floor for the composite-check-digit rewrite penalty: the composite is a
# single glyph and the empirical table is a weak proxy for it, so one
# implausible pair (e.g. "0" read as "4") must not dominate the search.
_CD_REWRITE_CAP = 1.5


def _edit_score(old: str, new: str) -> float:
    """Ranking evidence for explaining the observed `old` as `new`.

    log10 likelihood ratio of "GT was `new` and got misread as `old`"
    against "GT was `old` and was read correctly", minus a small
    per-cost-unit penalty. Normalising by the diagonal keeps globally
    common confusions (e.g. Z read as 2) from outweighing the local
    evidence of a *specific* glyph: a coincidental edit that merely
    satisfies Mod-10 has a low ratio, the true confusion a high one.
    """
    return (
        math.log10(prior(new, old))
        - math.log10(prior(old, old))
        - _EDIT_PENALTY * _confusion_cost(old, new)
    )


def _field_options(s, start, end, check_pos, trusted):
    """Candidate repairs that make one (field, check) pair verify.

    Returns [(cost, ((abs_pos, new_char), ...)), ...] ordered by edit
    score (empirical confusion prior, see _edit_score), capped at 8
    options. An empty edit tuple means "already valid". Options contain
    single edits, restricted two-edit combinations, and a "the check
    glyph was misread" candidate.
    """
    field = s[start:end]
    check = s[check_pos]
    if verify_mod10(field, check):
        return [(0, ())]

    opts = []
    edits = []  # (cost, pos, cand) plausible single edits in the field
    for i in range(start, end):
        ch = s[i]
        for cand, cost in confusions(ch):
            if cand == ch:
                continue
            edits.append((cost, i, cand))
            trial = s[start:i] + cand + s[i + 1:end]
            if verify_mod10(trial, check):
                opts.append((cost, ((i, cand),)))

    # Two substitutions. Enumerated even when single fixes exist: a
    # coincidental single edit that merely satisfies Mod-10 can be beaten
    # by the true two-edit explanation once _edit_score ranks the pair.
    for (c1, i1, a1), (c2, i2, a2) in itertools.combinations(edits, 2):
        if i1 == i2 or c1 + c2 > 4:
            continue
        trial = s[start:end]
        trial = trial[:i1 - start] + a1 + trial[i1 - start + 1:]
        trial = trial[:i2 - start] + a2 + trial[i2 - start + 1:]
        if verify_mod10(trial, check):
            opts.append((c1 + c2, ((i1, a1), (i2, a2))))

    # "Data wins": rewrite the check glyph to the computed digit.
    if check_pos not in trusted:
        comp = compute_check_digit(field)
        if comp != check:
            opts.append((_confusion_cost(check, comp), ((check_pos, comp),)))

    # Dedupe, best edit score first (true confusions beat coincidental
    # Mod-10 fixes), then cap.
    seen = set()
    uniq = []
    for opt in sorted(opts, key=lambda o: -sum(
            _edit_score(s[p], c) for p, c in o[1])):
        key = tuple(opt[1])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(opt)
    return uniq[:8]


def line2_correct_with_checksums(
    line2: str,
    *,
    trusted_check_positions: Optional[Iterable[int]] = None,
    max_cost: int = 4,
) -> tuple:
    """Run Mod-10 repair over Line 2.

    Args:
        line2: the 44-character Line 2 string (ICAO whitelist already
            applied -- characters outside the whitelist should have
            been replaced upstream).
        trusted_check_positions: check digits at these positions are
            taken as ground truth and never rewritten.
        max_cost: total edit budget; combinations costing more are
            rejected and the line is returned untouched.

    Returns:
        (corrected_line2, n_corrected, report) where corrected_line2 is
        fully check-valid whenever n_corrected > 0.
    """
    if len(line2) != LINE_LEN:
        return line2, 0, [f"len != {LINE_LEN} (got {len(line2)}), skipped"]

    trusted = set(trusted_check_positions or ())
    report = []
    per_field = []
    for label, start, end, cpos in _LINE2_FIELDS:
        per_field.append((label, _field_options(line2, start, end, cpos, trusted)))

    best = None  # (sort_key, trial_str, edits, needed_composite, comp_ok)
    for combo in itertools.product(*(opts for _label, opts in per_field)):
        total = sum(cost for cost, _ed in combo)
        if total > max_cost:
            continue
        trial = list(line2)
        edits = []
        score = 0.0
        for (label, _opts), (cost, ed) in zip(per_field, combo):
            for pos, cand in ed:
                score += _edit_score(trial[pos], cand)
                edits.append((label, pos, trial[pos], cand))
                trial[pos] = cand
        t = "".join(trial)
        needed = compute_check_digit(_composite(t))
        comp_ok = t[43] == needed
        if not comp_ok and 43 in trusted:
            continue
        if comp_ok:
            score += 0.5  # bonus: no composite rewrite needed
        else:
            score += max(_edit_score(t[43], needed), -_CD_REWRITE_CAP)
            total += _confusion_cost(t[43], needed)
        sort_key = (-score, total, len(edits))
        if best is None or sort_key < best[0]:
            best = (sort_key, t, edits, needed, comp_ok)

    if best is None:
        return line2, 0, ["no Mod-10 fix within budget (kept)"]

    _key, t, edits, needed, comp_ok = best
    if not edits and comp_ok:
        return line2, 0, ["all checks valid"]
    s2 = list(t)
    for label, pos, old, new in edits:
        report.append(f"{label}: pos {pos} fixed {old}->{new}")
    if not comp_ok:
        s2[43] = needed
        report.append(f"composite: pos 43 fixed {t[43]}->{needed}")
    n = len(edits) + (0 if comp_ok else 1)
    return "".join(s2), n, report


def _domain_cost(pos: int, cand: str) -> int:
    """How well `cand` fits the TD3 slot at `pos` (0 = fits, 1 = not)."""
    if pos in DIGIT_L2:
        return 0 if cand.isdigit() else 1
    if pos in ALPHA_L2:
        return 0 if cand.isalpha() else 1
    if pos == 20:
        return 0 if cand in SEX else 1
    return 0


def align_line2(raw: str) -> Optional[str]:
    """Align a near-44-char line to a fully check-valid 44-char line.

    Handles a single missing char (43 chars -> one insertion) and a
    single spurious char (45 chars -> one deletion). Longer/shorter
    inputs return None; the caller keeps its pad/trim path.
    """
    n = len(raw)
    if n == LINE_LEN:
        return raw
    if n == LINE_LEN - 1:
        hits = []
        for pos in range(LINE_LEN):
            for cand in sorted(ICAO):
                trial = raw[:pos] + cand + raw[pos:]
                if _line2_valid(trial):
                    hits.append((_domain_cost(pos, cand), pos, cand, trial))
        if hits:
            hits.sort()
            return hits[0][3]
        return None
    if n == LINE_LEN + 1:
        hits = []
        for pos in range(n):
            trial = raw[:pos] + raw[pos + 1:]
            if _line2_valid(trial):
                dup = (pos > 0 and raw[pos - 1] == raw[pos]) or \
                      (pos + 1 < n and raw[pos + 1] == raw[pos])
                hits.append((0 if dup else 1, pos, trial))
        if hits:
            hits.sort()
            return hits[0][2]
        return None
    return None


def repair_line2(raw: str, **kwargs) -> tuple:
    """Align + checksum-repair a raw Line 2 read.

    Returns (line44_or_raw, n_fixed, report, fully_valid).
    """
    raw = "".join(c if c in ICAO else "<" for c in (raw or "").upper())
    aligned = align_line2(raw)
    if aligned is None:
        return raw, 0, [f"cannot align len {len(raw)} to {LINE_LEN}"], False
    line, n, report = line2_correct_with_checksums(aligned, **kwargs)
    if len(line) == LINE_LEN and _line2_valid(line):
        return line, n, report, True
    return line, n, report, False


# ----------------------------------------------------------------------
# Self-test.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ICAO 9303 worked example for a 9-char passport_no field.
    sample = "D23145890"
    cd = compute_check_digit(sample)
    assert cd == "7", f"expected 7, got {cd}"

    # solve_ambiguity: force one bad char, pick the right digit back.
    ambig = list(sample)
    ambig[1] = "?"
    field_marker = "".join(ambig)
    cand = solve_ambiguity(field_marker, "7", "0123456789", marker="?")
    assert cand == "2", f"expected 2, got {cand}"

    # Build a valid synthetic TD3 line 2 and corrupt it the way the
    # corpus actually showed: letters misread as their digit lookalike
    # (O->0 count 448, Z->2 count 286 -- the reverse direction is rare),
    # plus a misread composite digit. Pairs whose char-code delta is a
    # multiple of 10 (L<->1, 6<->G, U<->0, T<->1) are checksum-invisible
    # and intentionally not used here.
    def _fld(v):
        return v + compute_check_digit(v)

    body = (_fld("L8SO92C34") + "UTO" + _fld("690806") + "F"
            + _fld("940623") + _fld("ZE184226B<<<<<"))
    line = body + compute_check_digit(_composite(body))
    assert len(line) == 44 and _line2_valid(line), line

    corrupt = list(line)
    corrupt[3] = "0"    # O->0 in passport_no
    corrupt[28] = "2"   # Z->2 in personal_no
    corrupt[43] = "0" if line[43] != "0" else "1"
    fixed, n, rep = line2_correct_with_checksums("".join(corrupt))
    assert fixed == line, (fixed, line, rep)
    assert _line2_valid(fixed)

    # Length alignment: delete a checked char, reinsert it.
    short = line[:14] + line[15:]
    assert align_line2(short) == line
    # Duplicate a checked char, drop it again.
    long_ = line[:8] + line[8] + line[8:]
    assert align_line2(long_) == line
    print("mrz_checksum self-test OK")
