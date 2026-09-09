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
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prep.language_codes import language_to_whisper_code
from data.prep.synthetic import generate_conversation, synthetic_manifest_entry
from eval.evaluate import aggregate, aggregate_by, evaluate_recording
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
    """Round-robins across the requested languages (defaulting to all 22) so a capped run
    gets broad language coverage instead of exhausting one language before starting the next
    -- `load_indic_diarbench` alone iterates languages sequentially and would otherwise fill
    `limit` entirely from the first language in the list."""
    from data.prep.manifest import ALL_LANGUAGES, round_robin_recordings

    yield from round_robin_recordings(languages or ALL_LANGUAGES, conditions, limit=limit)


def parse_conditions(values):
    """--conditions near_field far_field -> [AcousticCondition.NEAR_FIELD, ...]"""
    if not values:
        return None
    from schemas.types import AcousticCondition

    out = []
    for v in values:
        try:
            out.append(AcousticCondition(v.lower()))
        except ValueError:
            valid = [c.value for c in AcousticCondition]
            raise SystemExit(f"Unknown --conditions value '{v}'. Valid: {valid}")
    return out


def parse_taus(raw):
    """--osd-tau 0.3,0.5,0.7 -> [0.3, 0.5, 0.7]; None -> [None] (use each variant's default)."""
    if raw is None:
        return [None]
    taus = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            tau = float(part)
        except ValueError:
            raise SystemExit(f"--osd-tau expects numbers (e.g. 0.3,0.5,0.7), got '{part}'")
        if not 0.0 < tau < 1.0:
            raise SystemExit(f"--osd-tau must be in (0, 1), got {tau}")
        taus.append(tau)
    return taus or [None]


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
        f"{'Variant':<13} {'Metric':<7} {'Baseline':>9} {'Variant':>9} {'95% CI (variant)':>18} "
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
                f"{variant_name:<13} {metric:<7} {baseline_mean:>9.3f} {ci.mean:>9.3f} {ci_str:>18} "
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


BREAKDOWN_KEYS = ("language", "condition", "speakers", "overlap")

BREAKDOWN_TITLES = {
    "language": "Per-language breakdown (section 17.1)",
    "condition": "Per-acoustic-condition breakdown (section 17.2)",
    "speakers": "Per-speaker-count breakdown (section 17.3)",
    "overlap": "Per-overlap-intensity breakdown (overlap strata, section 3.5)",
}


def _fmt(value, width: int = 7, places: int = 3) -> str:
    """NaN-safe cell formatter -- an unscoreable column should read as '--', not 'nan'."""
    if value is None or math.isnan(value):
        return f"{'--':>{width}}"
    return f"{value:>{width}.{places}f}"


def build_jobs(variant_names, taus, backend, args):
    """One job per (variant, tau). Variants that don't run OSD are built once even under a
    tau sweep -- tau cannot change their output, and re-running them would burn GPU time on
    identical results and pad the table with rows that only look like a comparison."""
    jobs = []
    sweeping = len(taus) > 1
    seen_non_osd = set()
    for variant_name in variant_names:
        for tau in taus:
            overrides = {} if tau is None else {"osd_tau": tau}
            cfg = build_variant(
                variant_name, backend=backend, asr_model_size=args.asr_model_size,
                hf_token=args.hf_token, device=args.device,
                collect_separated_audio=args.si_sdr, **overrides,
            )
            if sweeping and not cfg.use_osd:
                if variant_name in seen_non_osd:
                    continue
                seen_non_osd.add(variant_name)
                jobs.append((variant_name, cfg))
            else:
                jobs.append((f"{variant_name}@{tau}" if sweeping else variant_name, cfg))
    return jobs


def print_breakdown(key: str, results_by_variant: dict) -> None:
    header = (
        f"{'System':<12} {'Group':<16} {'n':>4} {'DER':>7} {'DER-ov':>7} {'DER-no':>7} "
        f"{'WDER':>7} {'cpWER':>7} {'OSD-F1':>7} {'Routed%':>8}"
    )
    print(f"\n{BREAKDOWN_TITLES[key]}:\n")
    print(header)
    print("-" * len(header))
    for label, results in results_by_variant.items():
        for group, agg in aggregate_by(results, key).items():
            print(
                f"{label:<12} {group:<16} {int(agg['n']):>4} {_fmt(agg['der'])} "
                f"{_fmt(agg['der_overlap'])} {_fmt(agg['der_nonoverlap'])} {_fmt(agg['wder'])} "
                f"{_fmt(agg['cpwer'])} {_fmt(agg['osd_f1'])} {agg['routed_fraction'] * 100:>7.1f}%"
            )
    print(
        "\n(n is the number of recordings in that stratum -- a row with n=1 is an anecdote, not a "
        "per-group result. DER-ov / DER-no are DER restricted to reference overlap regions and to "
        "everything else.)"
    )


def print_si_sdr(results_by_variant: dict) -> None:
    print("\nSeparation quality (SI-SDR, dB, higher is better):\n")
    header = f"{'System':<12} {'SI-SDR':>8} {'scored':>7}"
    print(header)
    print("-" * len(header))
    any_scored = False
    for label, results in results_by_variant.items():
        agg = aggregate(results)
        any_scored = any_scored or agg["n_si_sdr"] > 0
        print(f"{label:<12} {_fmt(agg['si_sdr'], 8, 2)} {int(agg['n_si_sdr']):>7}")
    if not any_scored:
        print(
            "\n(No recording had ground-truth isolated sources, so SI-SDR is undefined here. Indic "
            "DiarBench publishes only the mixture and an RTTM -- separation quality is directly "
            "scorable on --synthetic runs, and only indirectly (via DER/WDER/cpWER) on real data.)"
        )


def run(args: argparse.Namespace) -> None:
    variants = args.variants.split(",") if args.variants else list(ALL_VARIANTS)
    backend = Backend.PRETRAINED if args.backend == "pretrained" else Backend.DUMMY
    taus = parse_taus(args.osd_tau)

    if args.synthetic is not None:
        recordings = list(iter_synthetic_recordings(args.synthetic))
    elif args.parquet is not None:
        recordings = list(iter_local_parquet_recordings(args.parquet, args.limit))
    else:
        recordings = list(iter_real_recordings(args.languages, parse_conditions(args.conditions), args.limit))

    jobs = build_jobs(variants, taus, backend, args)

    device_note = f", device={args.device}" if backend == Backend.PRETRAINED else ""
    tau_note = f", osd_tau sweep={taus}" if len(taus) > 1 else ""
    print(
        f"Evaluating {len(recordings)} recording(s) x {len(jobs)} run(s), "
        f"backend={backend.value}{device_note}{tau_note}\n"
    )

    header = (
        f"{'System':<12} {'DER':>7} {'DER-ov':>7} {'DER-no':>7} {'WDER':>7} {'cpWER':>7} "
        f"{'WER':>7} {'RTF':>7} {'OSD-F1':>7} {'Routed%':>8}"
    )
    print(header)
    print("-" * len(header))

    results_by_variant: dict[str, list] = {}

    for label, cfg in jobs:
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
                print_transcript_comparison(label, entry, transcript, stats, result)

        results_by_variant[label] = results
        agg = aggregate(results)
        print(
            f"{label:<12} {_fmt(agg['der'])} {_fmt(agg['der_overlap'])} {_fmt(agg['der_nonoverlap'])} "
            f"{_fmt(agg['wder'])} {_fmt(agg['cpwer'])} {_fmt(agg['wer'])} {_fmt(agg['rtf'])} "
            f"{_fmt(agg['osd_f1'])} {agg['routed_fraction'] * 100:>7.1f}%"
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

    if args.si_sdr:
        print_si_sdr(results_by_variant)

    for key in (args.breakdown or []):
        print_breakdown(key, results_by_variant)

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
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                         help="Device for pretrained VAD/OSD/embedding/separation/ASR models. "
                              "Defaults to 'cuda' if available, else 'cpu'.")
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
    parser.add_argument("--conditions", nargs="*", default=None, metavar="COND",
                         help="Restrict to these acoustic conditions (near_field, far_field, in_the_wild). "
                              "--dataset only. Needed for the per-condition comparison table.")
    parser.add_argument("--osd-tau", type=str, default=None, metavar="TAU[,TAU...]",
                         help="Override the OSD decision threshold. Accepts a comma-separated sweep "
                              "(e.g. --osd-tau 0.3,0.5,0.7), which runs every OSD-using variant once per "
                              "threshold and labels the rows VARIANT@TAU. Variants without OSD are run "
                              "once regardless, since tau cannot change their output.")
    parser.add_argument("--breakdown", nargs="*", default=None, choices=BREAKDOWN_KEYS, metavar="KEY",
                         help="Print per-stratum result tables in addition to the pooled one. "
                              "Choices: " + ", ".join(BREAKDOWN_KEYS) + ". Pass with no value for all four.")
    parser.add_argument("--si-sdr", action="store_true",
                         help="Score separation quality directly (SI-SDR) against ground-truth isolated "
                              "sources. Only meaningful on --synthetic runs: the real dataset ships the "
                              "mixture only. Holds separated audio in memory, hence off by default.")
    parser.add_argument("--show-transcript", action="store_true",
                         help="Print the reference (ground truth) and predicted (model output) transcript, "
                              "plus per-recording metrics, for every recording/variant -- not just the "
                              "aggregate table. Useful for a single clip: --limit 1 --show-transcript.")
    args = parser.parse_args()
    if args.breakdown is not None and not args.breakdown:
        args.breakdown = list(BREAKDOWN_KEYS)  # bare --breakdown means "all of them"
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.synthetic is None and not args.dataset and args.parquet is None:
        args.synthetic = 5  # default to a quick synthetic smoke test if nothing was specified

    run(args)


if __name__ == "__main__":
    main()
