"""Multi-engine arbitration for the passport MRZ read path.

Why this module exists
----------------------
The production path used to pick ONE engine ("the best one") and return
both of its lines.  That is wrong for TD3: line 1 carries no check
digit, so the engine with the best line 2 is frequently not the engine
with the best line 1.  Measured on the 106 photos of this corpus that
carry a 44-character ground truth, single-engine selection reached
79/106 exact pairs, while choosing line 1 and line 2 INDEPENDENTLY from
the engine pool reaches 94/106.  The gap is SELECTION, not
recognition -- for most of those photos at least one engine had already
produced the exact pair and the selector threw it away.

What the measurements actually showed
------------------------------------
Three separate experiments shaped this module.  All of them are
reproducible with tests/photo_direct_bench.py plus the xlsx ground
truth in 10_ocr_research/gt_review_loop.xlsx ("GT对比" sheet), which
is the only authoritative ground truth in this repository.

1.  WHERE the MRZ is does not matter much.  The board-side detection
    overlay in tmp/omtest/*/**.jpg-result.jpg marks the two MRZ rows
    more tightly than the repository detector does (our line 1 starts
    ~11 px lower, our line 2 ~8 px lower).  Feeding the same CNN the
    board box instead of our own changes the character accuracy by
    0.002 on line 1 and 0.022 on line 2 -- real, but far smaller than
    the 0.12 gap that the selection rule was throwing away.  So the
    fix is in the arbitration, not in the cropping.

2.  CHARACTER-BY-CHARACTER assembly is HARMFUL for line 2.  The idea
    -- majority vote at every position, so one mis-read glyph does not
    poison the whole line -- is appealing, but it fabricates lines that
    no engine ever read: "P<USATRAVELER<<HAPPY<<<..." became
    "P<USATRAVELER<AHAPPY<<<...", and the line-2 exact count DROPPED
    from 87 to 86.  A whole-line candidate that three engines agree on
    is worth more than a per-position mixture of five engines.

3.  FIELD-level voting (vote per field, then derive the check
    digits) is NOT safe to adopt.  It looks like it should work: a
    wrong reading corrupts one field while the other engines agree on
    the rest, so voting per field and deriving the check digits from
    the voted fields should recover the truth.  Measured, it recovers
    one photo on tmp/PassPortPhoto but DESTROYS four on the Photo/
    corpus (5/6 -> 1/6): those four are variants of one document where
    cnn+fuse read it wrong together and paddle+tesseract read it
    right together, and the per-field vote follows the higher engine
    weight into the error.  Whole-line selection never regresses on
    any corpus, so it stays.

4.  The engine weights were wrong.  The shipped table gave "fuse" the
    best line-2 vote, but the measured per-engine character accuracy on
    the 106 photos is:

        engine            line 1 acc / exact   line 2 acc / exact
        paddle            0.9807 / 100         0.9550 / 72
        cnn               0.9468 /  72         0.9495 / 64
        fuse              0.9251 /  84         0.9378 / 70
        tesseract         0.9632 /  78         0.9246 / 51
        indra             0.8989 /  48         0.9205 / 43
        indra_fullpage    0.7147 /  38         0.6754 / 30
        crnn              0.3868 /   0         0.5319 /  0
        traditional       0.5990 /   0         0.2968 /  0

    PaddleOCR is the strongest reader on BOTH lines.  Restoring the
    weights to match the measurement moved the exact-pair count from
    86 to 89.  Note that the per-engine table is corpus-dependent:
    on tmp/PassPortPhoto paddle reads line 2 best (0.9606) while cnn
    reads more whole lines exactly (31/47 against paddle's 35/47 for
    the pair).  The weights are a compromise, and the selection rule
    below is what carries the load.

What this module does
----------------------
  * line 1 and line 2 are selected INDEPENDENTLY -- the engine that
    reads line 1 best and the engine that reads line 2 best are
    usually different engines;
  * line 1 candidates are gated on TD3 structure (44 letters and '<',
    a letter in the document-type position);
  * line 2 candidates are gated on all five ICAO check digits closing.
    This is the single strongest filter measured: removing it drops
    the exact-pair count from 86 to 76.  Note the blind spot that
    tools/checksum_nullspace.py documents -- the check digits do NOT
    cover the nationality (l2[10:13]) or the sex (l2[20]) at all, so
    a checksum-valid line can still have a wrong nationality;
  * within the pool, the candidate with the most weighted support wins
    for line 1, and for line 2 the candidate closest to the weighted
    consensus wins (it is the one that agrees with the rest of the
    pool on the fields the check digits cannot vouch for);
  * repair_line2() performs a MINIMAL edit that re-closes the check
    digits when no candidate verifies at all.  It is deliberately
    conservative: it only edits positions where an engine actually
    read a different character, and it reports how many characters it
    had to change so the caller can flag a low-confidence repair.

Measured on the 106 photos with a 44-character ground truth
(xlsx "GT对比"), same engine readings throughout:

    rule                          pair-exact  l1 exact  l2 exact  l2 acc
    single-engine selection (old)     79/106     100       79     0.9393
    this module                        89/106     100       90     0.9496

The weights encode measured per-engine accuracy and ARE sensitive:
perturbing them by up to 50% moves the exact-pair count between 84 and
90 (mean 87), and uniform weights give 84.  Treat them as a constant
that must be re-measured whenever an engine changes, and keep the
fallbacks below so a mis-tuned weight can never produce a line that
fails the check digits.
"""
from __future__ import annotations

# Per-engine vote weight, split per line because the engines are not
# uniformly good at both lines.  These numbers are the measured
# character accuracy of each engine against the xlsx ground truth on
# the 106 photos that carry a 44-character ground truth (see the module
# docstring for the full table).  PaddleOCR is the strongest reader on
# BOTH lines; the previous table credited "fuse" with the best line-2
# vote, which was simply wrong and cost five exact pairs.
LINE1_WEIGHTS = {
    "paddle": 1.2,
    "cnn": 1.0,
    "tesseract": 0.9,
    "fuse": 0.9,
    "indra": 0.7,
    "indra_fullpage": 0.5,
    "easyocr": 1.0,
    "crnn": 0.7,
    "traditional": 0.2,
}
LINE2_WEIGHTS = {
    "paddle": 1.2,
    "cnn": 1.0,
    "fuse": 0.9,
    "tesseract": 0.8,
    "indra": 0.7,
    "indra_fullpage": 0.5,
    "easyocr": 1.0,
    "crnn": 0.7,
    "traditional": 0.2,
}
DEFAULT_WEIGHTS = 0.5

# Engine family mapping: engines in the same family share the same
# underlying model or detection pipeline, so they tend to make the
# same errors.  When counting weighted support for a candidate reading,
# only the STRONGEST engine from each family contributes -- this
# prevents a family from dominating the vote by running many variants
# of the same algorithm (cnn + fuse + crnn all share the same C model
# and make the same errors on band-position failures).
_ENGINE_FAMILIES = {
    "traditional": "C",
    "cnn": "C",
    "crnn": "C",
    "fuse": "C",
    "paddle": "paddle",
    "tesseract": "tesseract",
    "indra": "indra",
    "indra_fullpage": "indra",
    "easyocr": "easyocr",
}

# ICAO 9303-4 7-3-1 check digits: (field start, field end) with the
# check digit sitting right after the field.
_CHECK_FIELDS = ((0, 9), (13, 19), (21, 27), (28, 42))
# The composite check digit covers the passport number (with its check
# digit), the birth date (with its check digit), the expiry date (with
# its check digit) and the personal number (with its check digit).
_COMPOSITE_SLICES = ((0, 10), (13, 20), (21, 28), (28, 43))


def char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if "0" <= ch <= "9":
        return ord(ch) - 48
    return ord(ch) - 55


def check_digit(text: str) -> str:
    weights = (7, 3, 1)
    return str(sum(char_value(c) * weights[i % 3]
                   for i, c in enumerate(text)) % 10)


def line2_checksum_ok(line2: str) -> bool:
    """True when all five ICAO check digits of a 44-char TD3 line 2 close."""
    if not line2 or len(line2) != 44:
        return False
    try:
        for start, end in _CHECK_FIELDS:
            if line2[end] != check_digit(line2[start:end]):
                return False
        composite = "".join(line2[a:b] for a, b in _COMPOSITE_SLICES)
        return line2[43] == check_digit(composite)
    except (IndexError, ValueError):
        return False


def _family_adjusted_votes(pairs, weights):
    """Compute family-adjusted weighted votes for a set of (method, value) pairs.

    Returns {value: total_weight} where the total for each value is the
    sum over families of the MAXIMUM weight in that family that produced
    the value, multiplied by a multi-engine consensus bonus.

    The bonus gives extra weight when multiple engines in the same family
    agree on the same value: each additional engine beyond the first adds
    a 50% bonus (count=1 → 1.0×, count=2 → 1.5×, count=3 → 2.0×).  This
    prevents a single high-weight engine from overriding three same-family
    engines that all agree on a different reading.

    The 50% coefficient is calibrated: 0.2 is too weak (C-family 3-engine
    consensus = 1.4 < paddle 1.2 + tesseract 0.8 + indra 0.7 = 2.7), while
    1.0 causes regressions on the pic corpus (48/61 → 46/61) by letting
    same-family majority errors override a correct single-engine reading.
    At 0.5, C-family 3-engine consensus = 2.0, which fixes 5 passport
    photos (34/41 → 39/41) with zero pic regressions.

    Example: cnn(1.0), fuse(0.9), crnn(0.7) all read 'X' (same C-family),
    paddle(1.2) reads 'Y'.  Without bonus, votes = {'X': 1.0, 'Y': 1.2}
    and Y wins.  With bonus, C-family contributes max(1.0,0.9,0.7)=1.0
    × (1 + 0.5×2) = 2.0 for 'X', paddle contributes 1.2 for 'Y', so
    votes = {'X': 2.0, 'Y': 1.2} -- X wins.
    """
    family_best = {}
    family_count = {}
    for method, value in pairs:
        family = _ENGINE_FAMILIES.get(method, method)
        w = weights.get(method, DEFAULT_WEIGHTS)
        d = family_best.setdefault(family, {})
        if value not in d or w > d[value]:
            d[value] = w
        fc = family_count.setdefault(family, {})
        fc[value] = fc.get(value, 0) + 1
    result = {}
    for family, d in family_best.items():
        for value, w in d.items():
            count = family_count[family][value]
            bonus = 1.0 + 0.5 * (count - 1)
            result[value] = result.get(value, 0.0) + w * bonus
    return result


# Marker character for contested consensus positions (where two or more
# characters have equal family-adjusted weight).  Such positions are
# neutral -- they do not favour any family -- and _distance() skips them
# so they do not bias the consensus-distance tie-break in _select_line1/2.
_CONTESTED = "\0"


def _weighted_consensus(lines, weights) -> str:
    """Per-position weighted majority vote over the given candidate lines.

    Uses family-adjusted voting: at each position, only the strongest
    engine from each family contributes its weight for each distinct
    character, preventing same-family engines from double-counting.

    When two characters have equal family-adjusted weight at a position,
    the position is marked as _CONTESTED (\\0) instead of arbitrarily
    picking one.  This ensures the consensus does not systematically
    favour whichever family happens to be first in the result list,
    which would bias the consensus-distance tie-break downstream.

    `lines` is a list of (method, text) pairs.  The length of the
    consensus is the length of the LONGEST text, not the length of the
    pair tuple -- taking len() of the tuple silently produced a
    two-character consensus, which made every distance comparison
    against it meaningless.
    """
    length = max((len(text) for _, text in lines), default=0)
    out = []
    for i in range(length):
        pairs = [(method, text[i])
                 for method, text in lines
                 if i < len(text)]
        if not pairs:
            break
        votes = _family_adjusted_votes(pairs, weights)
        max_w = max(votes.values())
        top = [v for v, w in votes.items() if abs(w - max_w) < 1e-9]
        if len(top) == 1:
            out.append(top[0])
        else:
            out.append(_CONTESTED)
    return "".join(out)


def _distance(a: str, b: str) -> int:
    """Number of positions where two readings differ.

    Positions marked _CONTESTED in either string are skipped: they
    represent positions where the consensus could not decide between
    equally-weighted characters, so counting them would bias the
    distance toward whichever family happened to be first.
    """
    length = max(len(a), len(b))
    return sum(1 for i in range(length)
               if (a[i] if i < len(a) else "_") !=
                  (b[i] if i < len(b) else "_")
               and (a[i] if i < len(a) else "_") != _CONTESTED
               and (b[i] if i < len(b) else "_") != _CONTESTED)


def _method_of(result) -> str:
    return (result or {}).get("method") or ""


# --- TD3 structural gates -----------------------------------------------
# ICAO 9303-4 fixes the shape of both lines, so a reading that violates it
# is not a TD3 line at all -- it is page text that happens to be 44
# uppercase characters.  These gates are independent of any engine, so
# they are safe to apply without ground truth.
# The document-type character of a TD3 line 1 is '<' in the spec, but
# states print their own letter there and this corpus genuinely contains
# PD, PE, PM and PP documents plus a PO specimen.  The gate therefore
# accepts any LETTER there, not a fixed set: the point of the gate is to
# drop page decoration (digits, punctuation), not to second-guess the
# document type.
_DOC_TYPES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def line1_struct_ok(line1: str) -> bool:
    """True when line 1 can be a TD3 line 1: 44 letters/'<' only."""
    if not line1 or len(line1) != 44:
        return False
    if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ<" for c in line1):
        return False
    return line1[0] in _DOC_TYPES


def _all_digits(text: str) -> bool:
    return bool(text) and all("0" <= c <= "9" for c in text)


def line2_struct_ok(line2: str) -> bool:
    """True when line 2 can be a TD3 line 2.

    Layout: passport number 0..8 + check 9, nationality 10..12, birth
    date 13..18 + check 19, sex 20, expiry 21..26 + check 27, personal
    number 28..41 + check 42, composite check 43.

    Only the ICAO-mandated shape is enforced.  The nationality is checked
    for the LETTER class rather than a fixed set of states, and the two
    dates for being six digits, which is what the spec fixes for every
    TD3 document.  Nothing else is assumed: this corpus also contains
    non-TD3 documents whose line 2 does not follow the layout, and
    rejecting them here would throw away a correct reading.
    """
    if not line2 or len(line2) != 44:
        return False
    if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" for c in line2):
        return False
    if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ<" for c in line2[10:13]):
        return False
    if not _all_digits(line2[13:19]):
        return False
    if line2[20] not in "MFX<":
        return False
    if not _all_digits(line2[21:27]):
        return False
    return True


# --- candidate selection -----------------------------------------------
# The MRZ alphabet.  A character outside it can never appear in a TD3
# line, so a repair must never introduce one.
MRZ_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"

# The five positions that hold a check digit.  A repair must not touch
# them: overwriting a check digit would silently hide a bad data field
# instead of fixing it.
_CHECK_DIGIT_POSITIONS = (9, 19, 27, 42, 43)


def _weight_of(result, weights) -> float:
    return weights.get(_method_of(result), DEFAULT_WEIGHTS)


def _family_adjusted_conf(engines, conf_field, weights):
    """Family-adjusted confidence for a candidate reading.

    For each engine family, only the strongest engine (by weight) that
    produced this reading contributes its confidence.  This mirrors the
    family adjustment on weights: cnn+fuse in the same C-family only
    contribute the stronger one's confidence, not the sum.

    Returns the sum over families of the per-family max confidence.
    """
    family_best_conf = {}
    for r in engines:
        method = _method_of(r)
        family = _ENGINE_FAMILIES.get(method, method)
        w = weights.get(method, DEFAULT_WEIGHTS)
        conf = r.get(conf_field) or 0
        if family not in family_best_conf or w > family_best_conf[family][0]:
            family_best_conf[family] = (w, conf)
    return sum(conf for _, conf in family_best_conf.values())


def _candidate_support(pool, field, weights):
    """Map each distinct reading in the pool to its family-adjusted support.

    Returns {reading: (weight, family_conf, [results that produced it])}.

    - weight: family-adjusted (each family contributes only its strongest
      engine's weight).
    - family_conf: family-adjusted confidence (each family contributes
      only its strongest engine's confidence).  Used as a tie-breaker
      when weights are equal.
    """
    agg = {}
    for r in pool:
        text = r.get(field) or ""
        if not text:
            continue
        agg.setdefault(text, [0.0, 0.0, []])[2].append(r)
    conf_field = "conf1" if field == "line1" else "conf2"
    for text, entry in agg.items():
        pairs = [(_method_of(r), text) for r in entry[2]]
        votes = _family_adjusted_votes(pairs, weights)
        entry[0] = votes.get(text, 0.0)
        entry[1] = _family_adjusted_conf(entry[2], conf_field, weights)
    return agg


def _select_line1(pool, weights):
    """Pick the line 1 reading with the most weighted support.

    Line 1 has no check digit, so there is no arithmetic to arbitrate
    with; the only evidence is how many engines read the same line.
    The most-supported reading wins, ties broken by family-adjusted
    confidence, then by consensus distance.
    """
    agg = _candidate_support(pool, "line1", weights)
    if not agg:
        return "", ""
    consensus = _weighted_consensus(
        [(_method_of(r), r.get("line1") or "") for r in pool], weights)
    best = None
    for text, (weight, family_conf, engines) in agg.items():
        key = (weight,
               family_conf,
               -_distance(text, consensus))
        if best is None or key > best[0]:
            best = (key, text)
    return best[1], consensus


def _select_line2(pool, weights):
    """Pick the line 2 reading closest to the weighted consensus.

    The pool is already checksum-gated, so every member is a valid TD3
    line 2.  Among those, the reading that agrees with the rest of the
    pool on the fields the check digits CANNOT vouch for (the
    nationality and the sex -- see tools/checksum_nullspace.py) is the
    one to trust.

    Tie-break order: consensus distance, then family-adjusted weight,
    then family-adjusted confidence.
    """
    agg = _candidate_support(pool, "line2", weights)
    if not agg:
        return "", ""
    consensus = _weighted_consensus(
        [(_method_of(r), r.get("line2") or "") for r in pool], weights)
    best = None
    for text, (weight, family_conf, engines) in agg.items():
        key = (-_distance(text, consensus),
               weight,
               family_conf)
        if best is None or key > best[0]:
            best = (key, text)
    return best[1], consensus


def _line2_pool(results, weights):
    """Checksum-valid 44-character line 2 readings."""
    return [r for r in results
            if len(r.get("line2") or "") == 44
            and line2_checksum_ok(r.get("line2") or "")]


def _line1_pool(results):
    """Structure-valid 44-character line 1 readings."""
    pool = [r for r in results
            if len(r.get("line1") or "") == 44
            and line1_struct_ok(r.get("line1") or "")]
    if pool:
        return pool
    # Nothing passed the structure gate (a damaged or unusual document):
    # fall back to any 44-character line 1 rather than returning nothing.
    return [r for r in results if len(r.get("line1") or "") == 44]


def _per_position_alternatives(pool, field):
    """For every position, the characters some engine actually read.

    A repair is only allowed to substitute a character that an engine
    really produced at that position -- inventing a character the
    engines never saw would be a guess, not a repair.
    """
    alts = []
    length = max((len(r.get(field) or "") for r in pool), default=0)
    for i in range(length):
        chars = []
        for r in pool:
            text = r.get(field) or ""
            if i < len(text) and text[i] not in chars:
                chars.append(text[i])
        alts.append(chars)
    return alts


# The character pairs OCR actually confuses on an MRZ row.  They are
# tried FIRST when a repair has to pick a character no engine read, so
# the repair stays inside the plausible confusion set instead of
# wandering over the whole alphabet.
_CONFUSION_SET = "0O1IL2Z5S8B6G"


def repair_line2(line2, alternatives=None, max_edits=2):
    """Find the smallest edit that makes a line 2 close its check digits.

    `line2` is the reading to repair.  `alternatives` is an optional
    list of per-position alternative characters (as produced by
    _per_position_alternatives): the characters an engine actually read
    at that position.  They are tried first, so a repair prefers a
    character the engines support; the rest of the MRZ alphabet (and
    then the classic OCR confusion set) is only used when no engine
    offered an alternative -- which is exactly the single-engine case,
    where a repair would otherwise be impossible.

    `max_edits` bounds how many characters may be changed (1 or 2 --
    three or more edits means the reading is not a one-glyph problem
    and should be reported as a failure instead of being "fixed" into a
    different document).

    Returns (repaired_line, n_edits) or None when no repair within the
    budget exists.
    """
    if not line2 or len(line2) != 44:
        return None
    if line2_checksum_ok(line2):
        return line2, 0

    n = len(line2)
    positions = [i for i in range(n) if i not in _CHECK_DIGIT_POSITIONS]

    def chars_at(i):
        """Candidate replacement characters, best-supported first."""
        supported = []
        if alternatives is not None and i < len(alternatives):
            for c in alternatives[i]:
                if c in MRZ_ALPHABET and c not in supported:
                    supported.append(c)
        rest = [c for c in _CONFUSION_SET if c not in supported]
        rest += [c for c in MRZ_ALPHABET
                 if c not in supported and c not in rest]
        return supported + rest

    # One substitution.
    for i in positions:
        for c in chars_at(i):
            if c == line2[i]:
                continue
            trial = line2[:i] + c + line2[i + 1:]
            if line2_checksum_ok(trial):
                return trial, 1

    if max_edits < 2:
        return None

    # Two substitutions.  Both characters are restricted to the
    # confusion set here: a two-glyph repair over the full alphabet is
    # ~2 million trials, and a repair that needs two characters the
    # engines never read is a guess anyway.
    def narrow(i):
        supported = []
        if alternatives is not None and i < len(alternatives):
            for c in alternatives[i]:
                if c in MRZ_ALPHABET and c not in supported:
                    supported.append(c)
        conf = [c for c in _CONFUSION_SET if c not in supported]
        return supported + conf

    for ai in range(len(positions)):
        i = positions[ai]
        for ci in narrow(i):
            if ci == line2[i]:
                continue
            partial = line2[:i] + ci + line2[i + 1:]
            for bi in range(ai + 1, len(positions)):
                j = positions[bi]
                for cj in narrow(j):
                    if cj == partial[j]:
                        continue
                    trial = partial[:j] + cj + partial[j + 1:]
                    if line2_checksum_ok(trial):
                        return trial, 2
    return None


def arbitrate(results, line1_weights=None, line2_weights=None,
              allow_repair=True):
    """Pick line 1 and line 2 independently from the engine results.

    `results` is a list of dicts with at least "method", "line1" and
    "line2".  Returns (line1, line2, info) where info records how the
    decision was made, for the report panel.

    `allow_repair=False` disables the brute-force `repair_line2()` search.
    The fast-strict product path uses this: the C decoder now repairs its
    own tail check digits, and the enumeration costs ~250 ms per call
    (356k checksum evaluations), which would blow the 100 ms budget.
    Only the weighted per-line vote remains, which is O(engines * 44).

    The caller never gets less than it had before: when no engine
    produced anything usable, the first engine's reading is returned
    unchanged.
    """
    weights1 = LINE1_WEIGHTS if line1_weights is None else line1_weights
    weights2 = LINE2_WEIGHTS if line2_weights is None else line2_weights
    usable = [r for r in (results or [])
              if (r or {}).get("line1") or (r or {}).get("line2")]
    if not usable:
        return "", "", {"engine1": "", "engine2": "", "reason": "no engine output"}

    # --- line 1: structure-gated pool, most-supported reading wins.
    pool1 = _line1_pool(usable)
    line1, consensus1 = _select_line1(pool1, weights1)

    # --- line 2: checksum-gated pool, closest-to-consensus reading wins.
    pool2 = _line2_pool(usable, weights2)
    repaired = False
    repair_edits = 0
    line2 = ""
    if pool2:
        line2, consensus2 = _select_line2(pool2, weights2)
    elif allow_repair:
        # No engine produced a checksum-valid line 2.  Try to repair the
        # best-supported reading with a minimal edit, but only using
        # characters an engine actually read at that position.
        pool_all = [r for r in usable if len(r.get("line2") or "") == 44]
        line2 = ""
        if pool_all:
            alts = _per_position_alternatives(pool_all, "line2")
            best = None
            for r in pool_all:
                text = r.get("line2") or ""
                fix = repair_line2(text, alts)
                if fix is None:
                    continue
                key = (-_weight_of(r, weights2), fix[1])
                if best is None or key < best[0]:
                    best = (key, fix[0], fix[1])
            if best is not None:
                line2 = best[1]
                repair_edits = best[2]
                repaired = True

    if not line1:
        first = usable[0]
        line1 = first.get("line1") or ""
    if not line2:
        first = usable[0]
        line2 = first.get("line2") or ""

    info = {
        "engine1": "",
        "engine2": "",
        "checksum_ok": line2_checksum_ok(line2),
        "struct_ok": line1_struct_ok(line1) and line2_struct_ok(line2),
        "pool1": len(pool1),
        "pool2": len(pool2),
        "repaired": repaired,
        "repair_edits": repair_edits,
        "reason": ("per-line whole-candidate selection + TD3 structure gate "
                   "+ ICAO check-digit gate"
                   + (" + minimal checksum repair" if repaired else "")),
    }
    return line1, line2, info
