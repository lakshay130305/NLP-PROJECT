"""CLI: pool per-recording results from one or more --save-results JSONL files into the
corpus-wide tables a single run cannot produce.

Why this exists: a long evaluation has to be split into per-language jobs (the loader
materializes every decoded recording in memory, so the whole corpus at once will not fit),
but the headline table and the paired significance test are properties of the whole corpus,
not of any one job. They cannot be recovered from the per-language summary tables:

  * `aggregate()` means over recordings. Averaging N per-language means reweights every
    language to equal size regardless of how many recordings it holds, which is a different
    number from the mean over all recordings whenever the counts differ -- and they do.
  * The paired Wilcoxon test needs each recording's baseline-vs-variant pair. Printed
    aggregates do not contain them at any level of post-processing.

Feeding this script the JSONL files instead pools the raw per-recording rows, so both come
out correct by construction.

Usage:
  python scripts/pool_results.py results/*.jsonl
  python scripts/pool_results.py results/*.jsonl --breakdown --baseline B1
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate import aggregate
from eval.results_io import read_results
from scripts._stdio import force_utf8_stdio
from scripts.run_variant import (
    BREAKDOWN_KEYS,
    _fmt,
    print_breakdown,
    print_significance,
)


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand globs ourselves: on Windows the shell does not, so `results/*.jsonl` would
    otherwise arrive as a literal unmatched string."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(m) for m in glob.glob(pattern)]
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [candidate]
            else:
                raise SystemExit(f"No results file matched '{pattern}'")
        paths.extend(matches)
    return sorted(set(paths))


def variant_sort_key(name: str):
    """B1..B5 in order, PROPOSED last, anything else alphabetically after."""
    order = ["B1", "B2", "B3", "B4", "B5", "PROPOSED"]
    base = name.split("@")[0]
    return (order.index(base) if base in order else len(order), name)


def main() -> None:
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", metavar="JSONL",
                        help="One or more JSONL files written by run_variant.py --save-results. "
                             "Globs are expanded by this script, so quoting is unnecessary.")
    parser.add_argument("--breakdown", nargs="*", default=None, choices=BREAKDOWN_KEYS, metavar="KEY",
                        help="Per-stratum tables to print. Choices: " + ", ".join(BREAKDOWN_KEYS) +
                             ". Pass with no value for all four.")
    parser.add_argument("--significance", action="store_true", default=True,
                        help="Paired significance tests vs --baseline (on by default; this is the "
                             "main reason to pool).")
    parser.add_argument("--no-significance", dest="significance", action="store_false")
    parser.add_argument("--baseline", type=str, default="B1",
                        help="Variant treated as the baseline for the significance tests (default: B1).")
    args = parser.parse_args()
    if args.breakdown is not None and not args.breakdown:
        args.breakdown = list(BREAKDOWN_KEYS)

    paths = expand_paths(args.results)
    results_by_variant = read_results(paths)
    if not results_by_variant:
        raise SystemExit("No records found in the given files.")

    results_by_variant = {
        k: results_by_variant[k] for k in sorted(results_by_variant, key=variant_sort_key)
    }

    counts = {k: len(v) for k, v in results_by_variant.items()}
    languages = sorted({r.language for rs in results_by_variant.values() for r in rs if r.language})
    print(f"Pooled {sum(counts.values())} per-recording result(s) from {len(paths)} file(s), "
          f"{len(results_by_variant)} variant(s), {len(languages)} language(s).\n")

    # Unequal per-variant coverage means the rows below are not measured over the same
    # recordings, which quietly invalidates a side-by-side reading of them.
    if len(set(counts.values())) > 1:
        print("WARNING: variants cover different numbers of recordings -- the pooled rows below are")
        print("         NOT computed over the same set, so comparing them directly is unsound.")
        print("         Per-variant counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        print("         (The paired significance test is unaffected: it intersects on recording_id.)\n")

    header = (
        f"{'System':<12} {'n':>5} {'DER':>7} {'DER-ov':>7} {'DER-no':>7} {'WDER':>7} {'cpWER':>7} "
        f"{'WER':>7} {'RTF':>7} {'OSD-F1':>7} {'Routed%':>8}"
    )
    print("Corpus-wide comparison (pooled over recordings, not over languages):\n")
    print(header)
    print("-" * len(header))
    for label, results in results_by_variant.items():
        agg = aggregate(results)
        print(
            f"{label:<12} {int(agg['n']):>5} {_fmt(agg['der'])} {_fmt(agg['der_overlap'])} "
            f"{_fmt(agg['der_nonoverlap'])} {_fmt(agg['wder'])} {_fmt(agg['cpwer'])} "
            f"{_fmt(agg['wer'])} {_fmt(agg['rtf'])} {_fmt(agg['osd_f1'])} "
            f"{agg['routed_fraction'] * 100:>7.1f}%"
        )

    print("\nDER breakdown (fraction of total reference speech time) + OSD precision/recall:\n")
    bd_header = (
        f"{'System':<12} {'DER':>7} {'Missed':>7} {'FalseAl':>7} {'Confus':>7} "
        f"{'OSD-P':>7} {'OSD-R':>7} {'OSD-F1':>7}"
    )
    print(bd_header)
    print("-" * len(bd_header))
    for label, results in results_by_variant.items():
        agg = aggregate(results)
        print(
            f"{label:<12} {_fmt(agg['der'])} {_fmt(agg['der_missed'])} {_fmt(agg['der_false_alarm'])} "
            f"{_fmt(agg['der_confusion'])} {_fmt(agg['osd_precision'])} {_fmt(agg['osd_recall'])} "
            f"{_fmt(agg['osd_f1'])}"
        )

    for key in (args.breakdown or []):
        print_breakdown(key, results_by_variant)

    if args.significance:
        if args.baseline not in results_by_variant:
            print(f"\n(--baseline '{args.baseline}' not present in these files "
                  f"{list(results_by_variant)} -- skipping significance.)")
        else:
            print_significance(args.baseline, results_by_variant[args.baseline], results_by_variant)


if __name__ == "__main__":
    main()
