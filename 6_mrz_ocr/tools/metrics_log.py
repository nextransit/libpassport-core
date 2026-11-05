#!/usr/bin/env python3
"""Append bench rows to metrics_history.md WITH the corpus fingerprint.

The 10-30 lesson: metrics_history.md had three generations of numbers
under the same nominal corpus name because data/corpus_eval is
regenerable + gitignored, and every historical comparison silently broke.
This helper stamps every appended row with the signed corpus fingerprint
(tools/corpus_manifest.py) and the weights-header md5, so a row can never
again be ambiguous about WHAT produced it.

Table shape (existing columns kept, provenance rides inside `corpus`):

  | date | corpus | method | line1 acc | line2 acc | full | cksum | ms p50 |
      corpus cell = "<name>@<fingerprint12>"  (or plain name when unsigned)
"""

import datetime
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "metrics_history.md"
HEADER = "| date | corpus | method | line1 acc | line2 acc | full | cksum | ms p50 |\n"


def append_rows(rows: list[dict], history: Path | None = None,
                corpus_fingerprint: str | None = None,
                weights_md5: str | None = None) -> None:
    """rows: dicts with keys corpus, method, l1, l2, full, cksum, p50."""
    history = history or HISTORY
    lines = []
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    prov = ""
    if weights_md5:
        prov += f" w={weights_md5[:12]}"
    for r in rows:
        corpus_cell = f"{r['corpus']}"
        if corpus_fingerprint:
            corpus_cell += f"@{corpus_fingerprint}"
        lines.append(
            f"| {stamp} | {corpus_cell}{prov} | {r['method']} | "
            f"{r['l1']*100:.2f}% | {r['l2']*100:.2f}% | {r['full']*100:.2f}% | "
            f"{r['cksum']*100:.2f}% | {r['p50']:.1f} |\n")
    text = history.read_text() if history.exists() else HEADER
    if not text.endswith("\n"):
        text += "\n"
    history.write_text(text + "".join(lines))
    print(f"[metrics-history] appended {len(lines)} row(s) -> {history.name}", file=__import__("sys").stderr)


def last_corpus_fingerprint(history: Path | None = None) -> str | None:
    """Most recent @fingerprint seen in the history (for sanity checks)."""
    history = history or HISTORY
    if not history.exists():
        return None
    hits = re.findall(r"@([0-9a-f]{12})", history.read_text())
    return hits[-1] if hits else None
