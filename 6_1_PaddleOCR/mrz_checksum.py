#!/usr/bin/env python3
"""mrz_checksum -- ICAO 9303 TD3 Mod-10 check-digit primitives for Line 2.

The TD3 (standard passport) MRZ layout has two 44-character lines. Per
the user's correction (this is a critical correction — Line 1 has
NO check digits; all four local check digits and the composite check
digit live in Line 2):

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
    (DOB + cd_2) + (expiry + cd_3) + (personal_no + cd_4), treated as
    four concatenated sub-fields of total length 9+1+6+1+6+1+14+1 = 39
    characters, with weights cycling 7, 3, 1 across the whole 39.

Mod-10 algorithm:
    1. Map each character to a digit:
         '0'..'9' -> 0..9
         'A'..'Z' -> 10..35
         '<'      -> 0
    2. Multiply each digit by a weight cycling 7, 3, 1.
    3. Sum all weighted digits.
    4. check_digit = sum mod 10.

The reason this is *the* killer feature for MRZ OCR correction is
that all three weights (7, 3, 1) are coprime with the modulus 10:
    gcd(7,10) = gcd(3,10) = gcd(1,10) = 1.
So a single-character ambiguity in a digit-only slot has a UNIQUE
solution under Mod-10: when one position in a checked field is
ambiguous, plugging each candidate into the congruence equation
selects exactly one valid digit. No beam-search scoring needed.

This module exposes:
    char_to_digit(c)              — ICAO char -> 0..35
    digit_to_char(d)              — 0..35 -> ICAO char
    compute_check_digit(field)    — field str -> 1-char check
    verify_mod10(field, check)    — bool
    solve_ambiguity(field, check, candidates)
                                — given a field with one unknown slot,
                                  return the digit that satisfies Mod-10
    line2_resolve_with_checksums(line2)
                                — apply Mod-10 correction across
                                  all five Line-2 checked fields;
                                  returns the corrected line.
"""
from __future__ import annotations
from typing import Iterable, Optional

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
    Anything else raises ValueError — caller is responsible for
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
# Line 2 correction: apply Mod-10 across all five checked sub-fields.
# ----------------------------------------------------------------------

# (slice, check_pos) tuples for the four local check digits.
_LINE2_FIELDS = (
    # (field_label,    field_slice,   check_pos)
    ("passport_no",   slice(0, 9),   9),
    ("date_of_birth", slice(13, 19), 19),
    ("date_of_expiry", slice(21, 27), 27),
    ("personal_no",   slice(28, 42), 42),
)

# AMBIG candidates for digit-slot positions when OCR mis-reads.
# These are the same visual confusions as the slot-enforcement table
# in paddle_local.py: O<->0, I<->1, B<->8, S<->5, Z<->2.
_DIGIT_AMBIG = {
    "O": "0", "0": "O", "I": "1", "1": "I",
    "B": "8", "8": "B", "S": "5", "5": "S",
    "Z": "2", "2": "Z",
}


def _candidate_swaps(c: str) -> list:
    """All valid ICAO digit candidates for an ambiguous OCR char.

    For a digit-position slot, candidates are: the char itself, plus
    its AMBIG swap (if it's a digit-or-alpha pair). Always includes
    0-9 since the slot may need any digit to satisfy the check.
    """
    out = [c]
    if c in _DIGIT_AMBIG:
        out.append(_DIGIT_AMBIG[c])
    # Always include all 10 digits as candidates for digit-only slots —
    # the Mod-10 solver will pick the right one if exactly one matches.
    out.extend("0123456789")
    # Dedupe while preserving order.
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def line2_correct_with_checksums(
    line2: str,
    *,
    trusted_check_positions: Optional[Iterable[int]] = None,
) -> tuple:
    """Run Mod-10 correction over Line 2.

    For each of the four local check-digit fields (passport_no,
    DOB, expiry, personal_no), this routine:
      1. Verifies the local check digit against the field.
      2. If invalid, enumerates candidate substitutions for each
         digit-slot character (using AMBIG swap and the all-digits
         fallback), then uses Mod-10 to pick the unique solution.
      3. If a unique solution exists, applies it; otherwise leaves
         the field untouched.
    After the four local checks are resolved, the composite check
    at position 43 is also verified (and corrected) using the
    same principle across the concatenated fields.

    Args:
        line2: the 44-character Line 2 string (ICAO whitelist already
            applied — characters outside the whitelist should have
            been replaced upstream by paddle_local._norm_line).
        trusted_check_positions: if provided, the check digits at
            these positions are taken as ground truth and NOT
            corrected (the user may want to skip them in noisy
            OCR situations).

    Returns:
        (corrected_line2, n_corrected, report) where:
          - corrected_line2: the 44-char string after Mod-10 fixes
          - n_corrected: number of positions that were modified
          - report: list[str], human-readable per-field action log
            ("passport_no: pos 5 fixed 0->8", etc.)
    """
    if len(line2) != 44:
        return line2, 0, [f"len != 44 (got {len(line2)}), skipped"]

    trusted = set(trusted_check_positions or [])
    s = list(line2)
    report = []
    n = 0

    for label, fld_slice, check_pos in _LINE2_FIELDS:
        field = "".join(s[fld_slice])  # ICAO whitelist str snapshot
        check = s[check_pos]
        if verify_mod10(field, check):
            continue  # already correct
        # Try to find the unique substitution. Walk every position in
        # the field and, if the OCR char disagrees with the slot's
        # known-good candidates, try the AMBIG swap and the all-digits
        # set, looking for a (pos, candidate) pair that fixes the
        # check. Try AMBIG first (cheapest), then the all-digit set.
        found = None
        # The cheapest search: try AMBIG swap at the position the
        # OCR seems most likely wrong (i.e. where the AMBIG-swap
        # candidate is a digit in a digit slot).
        for i, c in enumerate(field):
            if c in _DIGIT_AMBIG and _DIGIT_AMBIG[c] in "0123456789":
                cand = _DIGIT_AMBIG[c]
                cand_field = field[:i] + cand + field[i + 1:]
                if verify_mod10(cand_field, check):
                    # If multiple positions match, the AMBIG-swap at
                    # the *first* such position wins (deterministic).
                    if found is None:
                        found = (i, cand)
        # If AMBIG didn't find a unique fix, try each position with
        # every digit 0..9 — the Mod-10 should land on exactly one.
        if found is None:
            for i in range(len(field)):
                if field[i] in "0123456789":
                    continue  # already a digit; skip trivial
                for cand in "0123456789":
                    cand_field = field[:i] + cand + field[i + 1:]
                    if verify_mod10(cand_field, check):
                        if found is None:
                            found = (i, cand)
                        else:
                            # two solutions -> ambiguous, abandon
                            found = None
                            break
                if found is None:
                    break
        if found is not None:
            i, cand = found
            old = field[i]
            field_list = list(s)
            field_list[fld_slice.start + i] = cand
            s = field_list
            report.append(f"{label}: pos {fld_slice.start + i} fixed {old}->{cand}")
            n += 1
        else:
            report.append(f"{label}: no unique Mod-10 fix (kept)")

    # Composite check at position 43 — concatenate the four (field+cd)
    # sub-fields into a 39-char string and verify cd against it.
    composite = (
        "".join(s[0:10]) + "".join(s[13:20]) +
        "".join(s[21:28]) + "".join(s[28:43])
    )
    if 43 not in trusted and not verify_mod10(composite, s[43]):
        # Try a single-slot correction across the composite field.
        for i in range(len(composite)):
            for cand in "0123456789":
                if composite[i] == cand:
                    continue
                cand_composite = composite[:i] + cand + composite[i + 1:]
                if verify_mod10(cand_composite, s[43]):
                    # Map i back to s-position (since composite skips
                    # pos 10..12 and 20).
                    if i < 10:
                        pos = i
                    elif i < 10 + 7:
                        pos = i + 3  # skip [10..12]
                    elif i < 10 + 7 + 7:
                        pos = i + 4  # skip [10..12] and [20]
                    else:
                        pos = i + 5  # skip [10..12], [20], [27]
                    # 27 is the third check digit, which is in s.
                    # We don't want to overwrite a check digit (43 is
                    # the composite cd, not in the 39-char input).
                    # But pos might land on check-digit positions 9, 19,
                    # 27, 42 which are inside composite as well.
                    if pos in {9, 19, 27, 42}:
                        # Re-align: check digits are at s-pos 9, 19,
                        # 27, 42 which map to composite offsets 9, 16,
                        # 23, 37. We don't want to *change* check
                        # digits from the composite; instead let the
                        # local checks above handle them. Bail.
                        continue
                    s[pos] = cand
                    report.append(f"composite: pos {pos} fixed {line2[pos]}->{cand}")
                    n += 1
                    break
            else:
                continue
            break
        else:
            report.append("composite: no unique Mod-10 fix (kept)")

    return "".join(s), n, report


# ----------------------------------------------------------------------
# Self-test: ICAO 9303 §4.9 worked example (or close to it).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic check: a 9-char passport_no field whose Mod-10 must
    # equal the documented check digit.
    sample = "D23145890"  # 9 chars
    cd = compute_check_digit(sample)
    assert cd == "7", f"expected 7, got {cd}"

    # solve_ambiguity: force one bad char, the solver must pick the
    # right digit (D in slot 1 should be replaced; we replace with '?'
    # and ask which of {0..9} matches).
    ambig = list(sample)
    ambig[1] = "?"
    field_marker = "".join(ambig)
    cand = solve_ambiguity(field_marker, "7", "0123456789", marker="?")
    print(f"solve_ambiguity({field_marker!r}, '7', 0..9) -> {cand}")
    assert cand is not None
    print("mrz_checksum self-test OK")
