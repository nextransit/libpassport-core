#!/usr/bin/env python3
"""Tests for the per-position multi-engine MRZ arbitration.

The arbitration is what turns N independent engine readings into one
line 1 and one line 2.  It is the piece that decides the final MRZ, so
its failure modes are worth pinning:

  1. the ICAO check digit must be computed the way the spec defines it
     (7-3-1 weights over the composite span s[0:10] + s[13:20] +
     s[21:28] + s[28:43]) -- an earlier version sliced the composite
     wrongly and rejected valid line 2 readings;
  2. a reading that is not a TD3 line at all (page body text, digits in
     line 1) must be gated out of the candidate pool;
  3. per-position assembly must repair a single wrong glyph without
     discarding the whole line;
  4. the assembled line 2 must always close its check digits, and the
     module must never return a line that fails them;
  5. CS/CD travel permits (a single 30-character line) must not be
     mangled by the TD3 logic.

Run:  python3 tests/test_mrz_arbitration.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import mrz_arbitration as A

g_fail = 0
g_total = 0


def check(cond, msg):
    global g_fail, g_total
    g_total += 1
    if not cond:
        g_fail += 1
        print(f"  FAIL {msg}")
    else:
        print(f"  ok   {msg}")


# A real, checksum-valid TD3 line 1 / line 2 pair taken from the corpus
# (assets/pic/08806d29a6268e1f_0003.jpg).
GOOD_L2 = "S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06"
GOOD_L1 = "P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<"


def test_check_digit():
    print("check digit")
    # ICAO 9303-4 example: the check digit over "L898902C" is 3.
    check(A.check_digit("L898902C") == "3", "check digit of L898902C is 3")
    check(A.check_digit("<<<<<<<<") == "0", "check digit of all filler is 0")
    # A line whose composite check digit is right must verify.
    l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    check(A.line2_checksum_ok(l2),
          "a spec-conformant line 2 verifies")
    # Break the composite check digit -> must not verify.
    bad = l2[:-1] + ("0" if l2[-1] != "0" else "1")
    check(not A.line2_checksum_ok(bad),
          "a broken composite check digit is rejected")
    # Break one local check digit -> must not verify.
    broken = l2[:9] + ("0" if l2[9] != "0" else "1") + l2[10:]
    check(not A.line2_checksum_ok(broken),
          "a broken passport-number check digit is rejected")
    check(not A.line2_checksum_ok(""), "an empty line does not verify")
    check(not A.line2_checksum_ok("X" * 44), "a non-ICAO line does not verify")


def test_structure_gates():
    print("structure gates")
    check(A.line1_struct_ok(GOOD_L1), "a real line 1 passes the gate")
    check(not A.line1_struct_ok("P1CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"),
          "a line 1 with a digit is rejected")
    check(not A.line1_struct_ok(GOOD_L1[:43]), "a 43-character line 1 is rejected")
    # The document type is a letter in this corpus (PD / PE / PM / PP / PO),
    # so the gate must not be limited to 'P'.
    check(A.line1_struct_ok("PD" + GOOD_L1[2:]),
          "a PD document line 1 passes the gate")
    check(A.line2_struct_ok(GOOD_L2), "a real line 2 passes the gate")
    check(not A.line2_struct_ok("S2100023<6CHE7108019M2002293<<<<<<<<<<<<<0#"),
          "a line 2 with a non-ICAO character is rejected")
    # The birth date must be six digits.
    check(not A.line2_struct_ok(GOOD_L2[:13] + "71080A" + GOOD_L2[19:]),
          "a line 2 with a non-digit birth date is rejected")


def test_assembly_repairs_one_glyph():
    print("per-position assembly")
    wrong_l1 = "P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    # engine A mis-reads one glyph of line 1 and one glyph of line 2;
    # engine B reads both lines correctly.  The assembly must take the
    # majority character and end up with B's line.
    a1 = "P<CHEDE<MAIENFELD<<PIERRX<<<<<<<<<<<<<<<<<<<"  # 'X' instead of 'E'
    a2 = GOOD_L2[:30] + "0" + GOOD_L2[31:]               # one digit wrong
    b1 = GOOD_L1
    b2 = GOOD_L2
    results = [
        {"method": "cnn", "line1": a1, "line2": a2, "conf1": 90, "conf2": 90},
        {"method": "paddle", "line1": b1, "line2": b2, "conf1": 90, "conf2": 90},
        {"method": "fuse", "line1": b1, "line2": b2, "conf1": 90, "conf2": 90},
    ]
    l1, l2, info = A.arbitrate(results)
    check(l1 == GOOD_L1, f"line 1 is repaired (got {l1!r})")
    check(l2 == GOOD_L2, f"line 2 is repaired (got {l2!r})")
    check(info["checksum_ok"], "the assembled line 2 closes its check digits")


def test_gates_out_page_text():
    print("page-text rejection")
    # A page body read is 44 uppercase characters but has digits in the
    # line-1 slot and a line 2 whose checks do not close.  Neither may
    # reach the output.
    junk1 = "AALLLAL<P<1LWGGLO<<P3GAAE<<<<<<<<<<<<<<<<<<<"
    junk2 = "A24L9L23<LLLL74C8C4GP2E322G3<<<<<<<<<<<<<<<E"
    results = [
        {"method": "traditional", "line1": junk1, "line2": junk2,
         "conf1": 74, "conf2": 73},
    ]
    l1, l2, info = A.arbitrate(results)
    # There is no valid candidate, so the module must NOT invent one:
    # it returns the reading unchanged and flags it as failing both the
    # structure gate and the check digits, so the caller can reject it.
    check(not A.line1_struct_ok(l1) and not A.line2_struct_ok(l2),
          "a page-text read is not dressed up as a TD3 line")
    check(not info["checksum_ok"] and not info["struct_ok"],
          "the info block flags the reading as invalid")


def test_single_line_cscd_not_mangled():
    print("CS/CD travel permit")
    # A CS/CD code is a single 30-character line; the TD3 logic must not
    # invent a second line or corrupt the one it has.
    permit = "CDL000000007<2508197<8208031<6"
    results = [{"method": "cnn", "line1": permit, "line2": "",
                "conf1": 90, "conf2": 0}]
    l1, l2, _ = A.arbitrate(results)
    check(l1 == permit, "a 30-character line 1 is returned unchanged")
    check(l2 == "", "no line 2 is invented")


def test_empty_and_garbage_input():
    print("degenerate input")
    l1, l2, info = A.arbitrate([])
    check(l1 == "" and l2 == "", "no engines -> empty result")
    l1, l2, _ = A.arbitrate([{"method": "cnn", "line1": "", "line2": ""}])
    check(l1 == "" and l2 == "", "an engine with no output -> empty result")


def test_line1_and_line2_are_chosen_independently():
    """The whole point of the arbitration: the best line 1 and the best
    line 2 usually come from DIFFERENT engines, and taking both from
    one engine throws a correct reading away."""
    print("independent per-line selection")
    # engine "cnn" reads line 1 correctly but mangles line 2;
    # engine "paddle" reads line 2 correctly but mangles line 1.
    # Three engines read line 1 correctly so the majority carries it
    # even though the single best line-2 engine is a different one.
    cnn_l1 = GOOD_L1
    cnn_l2 = GOOD_L2[:30] + "1" + GOOD_L2[31:]        # one digit wrong
    pad_l1 = GOOD_L1[:10] + "X" + GOOD_L1[11:]        # one glyph wrong
    pad_l2 = GOOD_L2
    results = [
        {"method": "cnn", "line1": cnn_l1, "line2": cnn_l2,
         "conf1": 90, "conf2": 90},
        {"method": "paddle", "line1": pad_l1, "line2": pad_l2,
         "conf1": 90, "conf2": 90},
        {"method": "tesseract", "line1": cnn_l1, "line2": cnn_l2,
         "conf1": 90, "conf2": 90},
    ]
    l1, l2, _ = A.arbitrate(results)
    check(l1 == GOOD_L1, f"line 1 comes from the engines that read it "
                        f"best (got {l1!r})")
    check(l2 == GOOD_L2, f"line 2 comes from the engine that read it "
                        f"best (got {l2!r})")


def test_checksum_gate_drops_broken_line2():
    """A line 2 whose check digits do not close is not a TD3 line 2 and
    must not be able to contribute a character to the result."""
    print("checksum gate")
    broken2 = GOOD_L2[:30] + "9" + GOOD_L2[31:]        # checksum no longer closes
    results = [
        {"method": "paddle", "line1": GOOD_L1, "line2": broken2,
         "conf1": 95, "conf2": 95},
        {"method": "cnn", "line1": GOOD_L1, "line2": GOOD_L2,
         "conf1": 95, "conf2": 95},
    ]
    l1, l2, info = A.arbitrate(results)
    check(l2 == GOOD_L2, f"the checksum-valid reading wins "
                         f"(got {l2!r})")
    check(info["pool2"] == 1,
          f"the broken reading is not in the pool (pool2={info['pool2']})")


def test_repair_closes_the_check_digits():
    """When no engine produced a checksum-valid line 2, the module
    repairs the best reading with a MINIMAL edit instead of returning
    an unverifiable line."""
    print("automatic repair")
    # Break exactly one data character of a valid line 2.
    broken = GOOD_L2[:5] + ("7" if GOOD_L2[5] != "7" else "8") + GOOD_L2[6:]
    check(not A.line2_checksum_ok(broken),
          "the broken reading really does fail its check digits")
    results = [{"method": "cnn", "line1": GOOD_L1, "line2": broken,
                "conf1": 90, "conf2": 90}]
    l1, l2, info = A.arbitrate(results)
    check(A.line2_checksum_ok(l2),
          f"the repaired line 2 closes its check digits (got {l2!r})")
    check(info.get("repaired"), "the info block reports the repair")
    check(info.get("repair_edits", 0) >= 1,
          "the info block reports how many characters were changed")
    check(l2 != broken, "the repaired line differs from the broken one")
    # The repair must change as little as possible: one data character.
    check(sum(1 for a, b in zip(l2, broken) if a != b) == 1,
          "the repair changed exactly one character")


def test_repair_does_not_touch_check_digits():
    """A repair must never overwrite a check digit -- that would hide
    a bad data field instead of fixing it."""
    print("repair safety")
    broken = GOOD_L2[:30] + "9" + GOOD_L2[31:]
    fix = A.repair_line2(broken)
    check(fix is not None, "a one-character repair exists")
    if fix:
        changed = [i for i, (a, b) in enumerate(zip(fix[0], broken))
                   if a != b]
        check(all(i not in A._CHECK_DIGIT_POSITIONS for i in changed),
              f"the repair left the check digits alone (changed {changed})")


def test_repair_respects_the_edit_budget():
    """A line that needs three or more edits is not a one-glyph
    problem; the module must report failure rather than fabricate a
    different document."""
    print("repair budget")
    # Corrupt three separate data fields.
    bad = list(GOOD_L2)
    for i in (2, 15, 25):
        bad[i] = "0" if bad[i] != "0" else "1"
    bad = "".join(bad)
    fix = A.repair_line2(bad, max_edits=2)
    check(fix is None or fix[1] <= 2,
          "no repair with more than two edits is returned")


def test_weights_follow_the_measurement():
    """PaddleOCR measured the strongest reader on BOTH lines.  The
    shipped weights must say so, or the arbitration throws away the
    best reading."""
    print("engine weights")
    check(A.LINE1_WEIGHTS["paddle"] >= max(
        v for k, v in A.LINE1_WEIGHTS.items() if k != "paddle"),
        "paddle has the top line-1 vote")
    check(A.LINE2_WEIGHTS["paddle"] >= max(
        v for k, v in A.LINE2_WEIGHTS.items() if k != "paddle"),
        "paddle has the top line-2 vote")
    # The engines that never read the corpus correctly must not be able
    # to out-vote a correct reading.
    for weak in ("crnn", "traditional"):
        check(A.LINE1_WEIGHTS[weak] < A.LINE1_WEIGHTS["paddle"],
              f"{weak} cannot out-vote paddle on line 1")
        check(A.LINE2_WEIGHTS[weak] < A.LINE2_WEIGHTS["paddle"],
              f"{weak} cannot out-vote paddle on line 2")


if __name__ == "__main__":
    test_check_digit()
    test_structure_gates()
    test_assembly_repairs_one_glyph()
    test_gates_out_page_text()
    test_single_line_cscd_not_mangled()
    test_empty_and_garbage_input()
    test_line1_and_line2_are_chosen_independently()
    test_checksum_gate_drops_broken_line2()
    test_repair_closes_the_check_digits()
    test_repair_does_not_touch_check_digits()
    test_repair_respects_the_edit_budget()
    test_weights_follow_the_measurement()
    print(f"\n{g_total - g_fail}/{g_total} checks passed")
    sys.exit(1 if g_fail else 0)
