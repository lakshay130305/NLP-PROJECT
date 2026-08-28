"""Summarize a stress_test.py run: overall + per-language + per-condition metric
aggregates, and a breakdown of any failures by error type. Safe to run against an
in-progress run's results.jsonl/failures.jsonl (just reads whatever's there so far).

Usage:
  python scripts/summarize_stress_test.py --output-dir outputs/stress_test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._stdio import force_utf8_stdio

METRIC_FIELDS = ["der", "wder", "cpwer", "wer", "rtf", "osd_f1", "routed_fraction"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def print_group_table(title: str, records: list[dict], group_key: str) -> None:
    print(f"\n{title}")
    print("-" * 90)
    header = f"{group_key:<14} {'n':>4} " + " ".join(f"{m:>9}" for m in METRIC_FIELDS)
    print(header)
    print("-" * 90)

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[group_key]].append(r)

    for key in sorted(groups):
        rows = groups[key]
        line = f"{key:<14} {len(rows):>4} "
        line += " ".join(f"{mean([r[m] for r in rows]):>9.3f}" for m in METRIC_FIELDS)
        print(line)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    results = load_jsonl(out_dir / "results.jsonl")
    failures = load_jsonl(out_dir / "failures.jsonl")

    print(f"Stress test summary for: {out_dir}")
    print(f"  {len(results)} succeeded, {len(failures)} failed\n")

    if results:
        print("Overall means:")
        for m in METRIC_FIELDS:
            print(f"  {m:<16} {mean([r[m] for r in results]):.4f}")

        print_group_table("Per-language breakdown", results, "language")
        print_group_table("Per-condition breakdown", results, "condition")
    else:
        print("No successful recordings yet.")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        by_type: dict[str, list[dict]] = defaultdict(list)
        for f in failures:
            by_type[f["error_type"]].append(f)
        for error_type, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  {error_type} ({len(items)}x):")
            for item in items[:5]:
                print(f"    - {item['recording_id']} ({item['language']}/{item['condition']}): {item['error_message']}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
    else:
        print("\nNo failures recorded.")


def main():
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, default="outputs/stress_test")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
