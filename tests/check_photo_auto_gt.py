#!/usr/bin/env python3
"""Check and conservatively repair tests/photo_auto_gt.json.

The file intentionally contains weak, human-review records.  This checker
therefore treats line-1 grammar problems in tier C/pending records as
warnings, while enforcing shape and checksum invariants for every populated
line.  ``--fix`` only clears unusable weak-label text; it never guesses
characters or changes a tier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "tests" / "photo_auto_gt.json"
ALPHA = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
ALNUM = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
WEIGHTS = (7, 3, 1)


def check_digit(value: str) -> int:
    total = 0
    for index, char in enumerate(value):
        if char == "<":
            number = 0
        elif char.isdigit():
            number = ord(char) - ord("0")
        else:
            number = ord(char) - ord("A") + 10
        total += number * WEIGHTS[index % 3]
    return total % 10


def line2_checksum_ok(line: str) -> bool:
    if len(line) != 44 or not set(line) <= ALNUM:
        return False
    for position, start, end in (
        (9, 0, 9),
        (19, 13, 19),
        (27, 21, 27),
        (42, 28, 42),
    ):
        if line[position] != str(check_digit(line[start:end])):
            return False
    composite = line[0:10] + line[13:20] + line[21:28] + line[28:43]
    return line[43] == str(check_digit(composite))


def line1_structurally_valid(line: str) -> bool:
    # ICAO 9303-4 TD3 line 1: 'P' followed by ONE additional document-type
    # character at the discretion of the issuing state ('<' or a letter --
    # e.g. PD diplomatic / PS service / PM; this corpus genuinely prints
    # PDRUSS…, PMKOR… and PPSVK… documents).  Only the issuer (cols 2-4)
    # must be pure letters.
    return (
        len(line) == 44
        and line[0] == "P"
        and (line[1] == "<" or "A" <= line[1] <= "Z")
        and set(line[2:5]) <= (ALPHA - {"<"})
        and set(line[5:]) <= ALPHA
    )


def expected_labels() -> set[str] | None:
    """Return the current photo corpus labels when the GUI helper is usable."""
    try:
        sys.path.insert(0, str(ROOT / "tests"))
        import gui  # noqa: PLC0415

        return {label for label, _path in gui.iter_passport_photos()}
    except (ImportError, RuntimeError):
        return None


def repair(data: dict) -> list[str]:
    changes: list[str] = []
    records = data.get("gt", {})
    if not isinstance(records, dict):
        return changes
    for name, record in records.items():
        if not isinstance(record, dict):
            continue
        tier = record.get("tier")
        if tier in {"A", "B"}:
            continue
        for field in ("l1", "l2"):
            value = record.get(field)
            if value is None and field in record:
                record[field] = ""
                changes.append(f"{name}.{field}: null -> empty")
            elif isinstance(value, str) and 0 < len(value) < 44:
                other = record.get("l2" if field == "l1" else "l1", "")
                if not other:
                    record[field] = ""
                    changes.append(
                        f"{name}.{field}: partial line length {len(value)} -> empty"
                    )
        l2 = record.get("l2", "")
        if (
            isinstance(l2, str)
            and len(l2) == 44
            and not line2_checksum_ok(l2)
            and record.get("l2_tier") not in {"A", "B"}
        ):
            record["l1"] = ""
            record["l2"] = ""
            record.setdefault("l1_frozen", False)
            record.setdefault("l2_frozen", False)
            changes.append(f"{name}: invalid weak-label l1/l2 pair -> empty")
    return changes


def validate(data: object) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        return ["root must be an object"], warnings
    records = data.get("gt")
    if not isinstance(records, dict):
        return ["root.gt must be an object"], warnings

    labels: set[str] = set()
    for name, record in records.items():
        prefix = f"{name}:"
        if not isinstance(name, str) or not isinstance(record, dict):
            errors.append(f"{prefix} record must be an object")
            continue
        label = record.get("label")
        if not isinstance(label, str) or not label:
            errors.append(f"{prefix} label must be a non-empty string")
        else:
            if label in labels:
                errors.append(f"{prefix} duplicate label {label}")
            labels.add(label)
            if Path(label).name != name:
                errors.append(f"{prefix} key does not match label basename {label}")
            if not (ROOT / label).is_file():
                errors.append(f"{prefix} image does not exist: {label}")

        tier = record.get("tier")
        if not isinstance(tier, str) or not tier:
            errors.append(f"{prefix} tier must be a non-empty string")
        l1 = record.get("l1", "")
        l2 = record.get("l2", "")
        if not isinstance(l1, str) or not isinstance(l2, str):
            errors.append(f"{prefix} l1/l2 must be strings")
            continue

        for field, line in (("l1", l1), ("l2", l2)):
            if line and len(line) != 44:
                errors.append(f"{prefix} {field} length {len(line)} != 44")
            if line and len(line) == 44:
                allowed = ALPHA if field == "l1" else ALNUM
                bad = sorted(set(line) - allowed)
                if bad:
                    message = f"{prefix} {field} illegal characters {bad}"
                    if tier in {"A", "B"}:
                        errors.append(message)
                    else:
                        warnings.append(message)

        if l2 and len(l2) == 44 and not line2_checksum_ok(l2):
            message = f"{prefix} l2 checksum is invalid"
            if tier in {"A", "B"} or record.get("l2_tier") in {"A", "B"}:
                errors.append(message)
            else:
                warnings.append(message)

        if l2 and len(l2) == 44:
            grammar_errors = []
            if not set(l2[10:13]) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                grammar_errors.append("nationality")
            if not set(l2[13:19]) <= set("0123456789"):
                grammar_errors.append("date of birth")
            if l2[20] not in "MF<":
                grammar_errors.append("sex")
            if not set(l2[21:27]) <= set("0123456789"):
                grammar_errors.append("expiry")
            if grammar_errors:
                message = f"{prefix} l2 grammar invalid: {', '.join(grammar_errors)}"
                warnings.append(message)

        if l1 and len(l1) == 44 and not line1_structurally_valid(l1):
            if tier in {"A", "B"}:
                errors.append(f"{prefix} trusted l1 grammar is invalid")
            else:
                warnings.append(f"{prefix} weak-label l1 grammar is invalid")

        if tier in {"A", "B"} and (len(l1) != 44 or len(l2) != 44):
            errors.append(f"{prefix} trusted tier requires two complete lines")

    expected = expected_labels()
    if expected is not None:
        actual = set(labels)
        for missing in sorted(expected - actual):
            errors.append(f"missing corpus record: {missing}")
        for extra in sorted(actual - expected):
            errors.append(f"record is outside current corpus: {extra}")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.path)

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {path}: {exc}")
        return 1

    if args.fix:
        changes = repair(data)
        if changes:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
            os.replace(temporary, path)
            for change in changes:
                print(f"FIX: {change}")
        else:
            print("FIX: no safe repairs needed")

    errors, warnings = validate(data)
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(
        f"SUMMARY: records={len(data.get('gt', {})) if isinstance(data, dict) and isinstance(data.get('gt'), dict) else 0} "
        f"warnings={len(warnings)} errors={len(errors)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
