"""CLI: run one or more system variants (B1..B5, PROPOSED) over a set of recordings
and print a section-13.1-style results table (DER / WDER / cpWER / WER / RTF).

Usage:
  # smoke test on synthetic audio, no network/model downloads (Backend.DUMMY):
  python scripts/run_variant.py --synthetic 5 --variants B1,PROPOSED

  # real dataset + real pretrained models (needs network, HF token for gated pyannote models):
  python scripts/run_variant.py --dataset --limit 10 --backend pretrained --variants B1,B2,B3,B4,B5,PROPOSED

  # score against a locally downloaded Parquet shard instead of streaming from the Hub:
  python scripts/run_variant.py --parquet path/to/hindi-test.parquet --backend pretrained --variants PROPOSED

  # does PROPOSED actually beat B1, or is a lower average just noise? (paired significance test)
  python scripts/run_variant.py --dataset --limit 30 --backend pretrained --variants B1,PROPOSED --significance

  # single clip: see the actual transcript (reference vs predicted) plus that clip's own numbers:
  python scripts/run_variant.py --dataset --languages Hindi --limit 1 --backend pretrained --variants PROPOSED --show-transcript
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prep.language_codes import language_to_whisper_code
from data.prep.synthetic import generate_conversation, synthetic_manifest_entry
from eval.evaluate import aggregate, evaluate_recording
from eval.significance import bootstrap_ci, cohens_d_paired, paired_test
from pipeline.config import Backend
from pipeline.orchestrator import OverlapAwarePipeline
from pipeline.variants import ALL_VARIANTS, build_variant
from scripts._env import load_env
from scripts._stdio import force_utf8_stdio
from scripts.demo import format_transcript

SIGNIFICANCE_METRICS = ("der", "wder", "cpwer")


def iter_synthetic_recordings(n: int):
    for i in range(n):
        num_speakers = 2 if i % 3 != 0 else 3
        convo = generate_conversation(
            num_speakers=num_speakers,
            num_turns=10,
            overlap_ratio=0.2,
            seed=i,
        )
        entry = synthetic_manifest_entry(f"synthetic_{i:03d}", convo)
        yield entry, convo.audio, convo.sample_rate


def iter_real_recordings(languages, conditions, limit):
    from data.prep.manifest import load_indic_diarbench

    yield from load_indic_diarbench(languages=languages, conditions=conditions, limit=limit)


def iter_local_parquet_recordings(parquet_path, limit):
    from data.prep.manifest import load_indic_diarbench_local

    yield from load_indic_diarbench_local(parquet_path, limit=limit)


def _paired_values(baseline_results, variant_results, metric: str) -> tuple[list[float], list[float]]:
    """Aligns two RecordingResult lists by recording_id (order isn't guaranteed to match
    if either run skipped/reordered a recording) and returns the paired metric values."""
    variant_by_id = {r.recording_id: r for r in variant_results}
    baseline_vals, variant_vals = [], []
    for b in baseline_results:
        v = variant_by_id.get(b.recording_id)
        if v is None:
            continue
        baseline_vals.append(getattr(b, metric))
        variant_vals.append(getattr(v, metric))
    return baseline_vals, variant_vals


def print_significance(baseline_name: str, baseline_results, results_by_variant: dict) -> None:
    print(f"\nSignificance vs baseline '{baseline_name}' (paired Wilcoxon signed-rank test, n={len(baseline_results)}):\n")
    if len(baseline_results) < 2:
        print("  (need at least 2 paired recordings to run a significance test -- skipping)")
        return

    header = (
        f"{'Variant':<10} {'Metric':<7} {'Baseline':>9} {'Variant':>9} {'95% CI (variant)':>18} "
        f"{'p-value':>9} {'Cohen d':>9}  Verdict"
    )
    print(header)
    print("-" * len(header))

    for variant_name, variant_results in results_by_variant.items():
        if variant_name == baseline_name:
            continue
        for metric in SIGNIFICANCE_METRICS:
            baseline_vals, variant_vals = _paired_values(baseline_results, variant_results, metric)
            if len(baseline_vals) < 2:
                continue
            test_result = paired_test(baseline_vals, variant_vals, test="wilcoxon")
            effect = cohens_d_paired(baseline_vals, variant_vals)
            ci = bootstrap_ci(variant_vals)
            baseline_mean = sum(baseline_vals) / len(baseline_vals)

            significant = test_result.p_value < 0.05
            if not significant:
                verdict = "no significant difference"
            elif ci.mean < baseline_mean:
                verdict = "significant IMPROVEMENT"
            else:
                verdict = "significant REGRESSION"

            ci_str = f"[{ci.lower:.3f}, {ci.upper:.3f}]"
            print(
                f"{variant_name:<10} {metric:<7} {baseline_mean:>9.3f} {ci.mean:>9.3f} {ci_str:>18} "
                f"{test_result.p_value:>9.4f} {effect:>9.3f}  {verdict}"
            )

    print(
        "\n(p < 0.05 required for significance; Cohen's d magnitude: ~0.2 small, ~0.5 medium, ~0.8 large. "
        "A lower mean alone is NOT sufficient evidence -- see README's 'what counts as beating B1' discussion.)"
    )


def print_transcript_comparison(variant_name: str, entry, transcript, stats, result) -> None:
    print(f"\n{'=' * 70}")
    print(f"[{variant_name}] {entry.recording_id}  ({entry.language}, {stats.audio_duration:.1f}s)")
    print(f"{'=' * 70}")

    if entry.reference_transcript is not None and entry.reference_transcript.utterances:
        print("\n--- Reference (ground truth) ---")
        print(format_transcript(entry.reference_transcript, entry.duration))
    else:
        print("\n--- Reference (ground truth) ---\n(none available for this recording)")

    print("\n--- Predicted (model output) ---")
    print(format_transcript(transcript, stats.audio_duration))

    print(
        f"\n--- Metrics ---\nDER={result.der:.3f}  WDER={result.wder:.3f}  cpWER={result.cpwer:.3f}  "
        f"WER={result.wer:.3f}  RTF={result.rtf:.2f}  OSD-F1={result.osd_f1:.3f}  "
        f"Routed={result.routed_fraction * 100:.1f}%"
    )


def run(args: argparse.Namespace) -> None:
    variants = args.variants.split(",") if args.variants else list(ALL_VARIANTS)
    backend = Backend.PRETRAINED if args.backend == "pretrained" else Backend.DUMMY

    if args.synthetic is not None:
        recordings = list(iter_synthetic_recordings(args.synthetic))
    elif args.parquet is not None:
        recordings = list(iter_local_parquet_recordings(args.parquet, args.limit))
    else:
        recordings = list(iter_real_recordings(args.languages, None, args.limit))

    print(f"Evaluating {len(recordings)} recording(s) x {len(variants)} variant(s), backend={backend.value}\n")

    header = f"{'System':<10} {'DER':>7} {'WDER':>7} {'cpWER':>7} {'WER':>7} {'RTF':>7} {'OSD-F1':>7} {'Routed%':>8}"
    print(header)
    print("-" * len(header))

    results_by_variant: dict[str, list] = {}

    for variant_name in variants:
        cfg = build_variant(variant_name, backend=backend, asr_model_size=args.asr_model_size, hf_token=args.hf_token)
        pipeline = OverlapAwarePipeline(cfg)

        results = []
        for entry, audio, sr in recordings:
            language_hint = language_to_whisper_code(entry.language) if backend == Backend.PRETRAINED else None
            transcript, stats, predicted_overlap = pipeline.run(
                audio, sr, recording_id=entry.recording_id, language=language_hint
            )
            result = evaluate_recording(entry, transcript, predicted_overlap, stats)
            results.append(result)
            if args.show_transcript:
                print_transcript_comparison(variant_name, entry, transcript, stats, result)

        results_by_variant[variant_name] = results
        agg = aggregate(results)
        print(
            f"{variant_name:<10} {agg['der']:>7.3f} {agg['wder']:>7.3f} {agg['cpwer']:>7.3f} "
            f"{agg['wer']:>7.3f} {agg['rtf']:>7.3f} {agg['osd_f1']:>7.3f} {agg['routed_fraction']*100:>7.1f}%"
        )

    if args.significance:
        if args.baseline not in results_by_variant:
            print(
                f"\n(--significance requested but baseline '{args.baseline}' wasn't in --variants "
                f"{list(results_by_variant)} -- skipping.)"
            )
        else:
            print_significance(args.baseline, results_by_variant[args.baseline], results_by_variant)


def main():
    force_utf8_stdio()
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", type=int, default=None, metavar="N",
                         help="Generate N synthetic conversations instead of loading the real dataset.")
    parser.add_argument("--dataset", action="store_true", help="Load the real Indic DiarBench dataset from HuggingFace (streamed).")
    parser.add_argument("--parquet", type=str, default=None, metavar="PATH",
                         help="Load recordings from a local Parquet shard of the dataset instead of streaming "
                              "from the Hub (e.g. a file downloaded by hand from the HF dataset viewer).")
    parser.add_argument("--languages", nargs="*", default=None, help="Restrict to these languages (--dataset only, not --parquet).")
    parser.add_argument("--limit", type=int, default=5, help="Max number of real-dataset recordings to load.")
    parser.add_argument("--variants", type=str, default=None, help=f"Comma-separated subset of {ALL_VARIANTS}.")
    parser.add_argument("--backend", choices=["dummy", "pretrained"], default="dummy",
                         help="dummy = no network/model downloads (default); pretrained = real pyannote/speechbrain/whisper models.")
    parser.add_argument("--asr-model-size", type=str, default="tiny", help="faster-whisper model size (pretrained backend only).")
    parser.add_argument("--hf-token", type=str, default=None,
                         help="HuggingFace token for gated pyannote models (pretrained backend's VAD/OSD only). "
                              "Defaults to the HF_TOKEN environment variable if set.")
    parser.add_argument("--significance", action="store_true",
                         help="After the results table, run a paired significance test (Wilcoxon signed-rank, "
                              "Cohen's d effect size, bootstrap 95%% CI) comparing each variant against "
                              "--baseline on DER/WDER/cpWER. Requires --baseline to be included in --variants "
                              "and at least 2 recordings.")
    parser.add_argument("--baseline", type=str, default="B1",
                         help="Variant to treat as the baseline for --significance (default: B1).")
    parser.add_argument("--show-transcript", action="store_true",
                         help="Print the reference (ground truth) and predicted (model output) transcript, "
                              "plus per-recording metrics, for every recording/variant -- not just the "
                              "aggregate table. Useful for a single clip: --limit 1 --show-transcript.")
    args = parser.parse_args()
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    if args.synthetic is None and not args.dataset and args.parquet is None:
        args.synthetic = 5  # default to a quick synthetic smoke test if nothing was specified

    run(args)


if __name__ == "__main__":
    main()
