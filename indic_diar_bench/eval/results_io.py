"""Per-recording result persistence (JSONL).

The aggregate tables a run prints are means over recordings; the per-recording values
behind them live only in memory and are gone when the process exits. That is fine for a
short run and fatal for a long one, for two separate reasons:

  1. A corpus-wide result cannot be reconstructed from per-language summary tables.
     `aggregate()` means over recordings, so averaging N language-level means silently
     reweights every language to equal size regardless of how many recordings it holds.
  2. The paired significance tests need each recording's baseline-vs-variant pair. No
     amount of post-processing recovers those from printed aggregates.

Writing one JSON line per (variant, recording) as the run proceeds fixes both, and makes
`--resume` possible: a multi-day evaluation that dies on day nine restarts from the last
completed recording instead of from nothing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import TextIO

from eval.evaluate import RecordingResult

_FIELD_NAMES = {f.name for f in fields(RecordingResult)}


def open_results_file(path: str | Path) -> TextIO:
    """Append-mode so a resumed run adds to, rather than truncates, prior work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def write_result(handle: TextIO, variant: str, result: RecordingResult) -> None:
    """Write one record and flush immediately -- an unflushed buffer is exactly what gets
    lost in the crash this file exists to survive."""
    record = {"variant": variant, **asdict(result)}
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _to_result(record: dict) -> RecordingResult:
    """Build a RecordingResult from a stored record, tolerating both directions of version
    skew: unknown keys are dropped, and fields the writer did not have are filled in. Only
    `recording_id` is genuinely required."""
    known = {k: v for k, v in record.items() if k in _FIELD_NAMES}
    for f in fields(RecordingResult):
        if f.name in known:
            continue
        if f.name == "si_sdr":
            known[f.name] = None
        elif f.name in ("language", "condition"):
            known[f.name] = ""
        elif f.name == "num_ref_speakers":
            known[f.name] = 0
        elif f.name != "recording_id":
            known[f.name] = 0.0
    return RecordingResult(**known)


def read_results(paths: list[str | Path]) -> dict[str, list[RecordingResult]]:
    """Load one or more JSONL files into {variant: [RecordingResult, ...]}.

    Deduplicates on (variant, recording_id), keeping the last occurrence, so re-running a
    language after a crash and concatenating the logs does not double-count it.

    Unparseable lines are skipped with a warning rather than aborting the load: a process
    killed partway through a write leaves a truncated final line, and that is precisely the
    situation --resume exists to recover from -- refusing to read the file would throw away
    every completed recording to avoid one incomplete one."""
    merged: dict[tuple[str, str], tuple[str, RecordingResult]] = {}
    for path in paths:
        for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                variant = record.pop("variant")
                result = _to_result(record)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"  (skipping unreadable line {lineno} of {path}: {exc})", file=sys.stderr)
                continue
            merged[(variant, result.recording_id)] = (variant, result)

    out: dict[str, list[RecordingResult]] = {}
    for variant, result in merged.values():
        out.setdefault(variant, []).append(result)
    return out


def load_completed(path: str | Path) -> dict[str, dict[str, RecordingResult]]:
    """{variant: {recording_id: result}} for everything already on disk, for --resume.
    A missing file is not an error: it just means nothing has been done yet."""
    path = Path(path)
    if not path.exists():
        return {}
    return {
        variant: {r.recording_id: r for r in results}
        for variant, results in read_results([path]).items()
    }
