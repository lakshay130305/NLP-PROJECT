"""CLI: run one or more system variants (B1..B5, PROPOSED) over a set of recordings
and print a section-13.1-style results table (DER / WDER / cpWER / WER / RTF).

Usage:
  # smoke test on synthetic audio, no network/model downloads (Backend.DUMMY):
  python scripts/run_variant.py --synthetic 5 --variants B1,PROPOSED

  # real dataset + real pretrained models (needs network, HF token for gated pyannote models):
  python scripts/run_variant.py --dataset --limit 10 --backend pretrained --variants B1,B2,B3,B4,B5,PROPOSED

  # score against a locally downloaded Parquet shard instead of streaming from the Hub:
  python scripts/run_variant.py --parquet path/to/hindi-test.parquet --backend pretrained --variants PROPOSED
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
from pipeline.config import Backend
from pipeline.orchestrator import OverlapAwarePipeline
from pipeline.variants import ALL_VARIANTS, build_variant
from scripts._stdio import force_utf8_stdio


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

    for variant_name in variants:
        cfg = build_variant(variant_name, backend=backend, asr_model_size=args.asr_model_size, hf_token=args.hf_token)
        pipeline = OverlapAwarePipeline(cfg)

        results = []
        for entry, audio, sr in recordings:
            language_hint = language_to_whisper_code(entry.language) if backend == Backend.PRETRAINED else None
            transcript, stats, predicted_overlap = pipeline.run(
                audio, sr, recording_id=entry.recording_id, language=language_hint
            )
            results.append(evaluate_recording(entry, transcript, predicted_overlap, stats))

        agg = aggregate(results)
        print(
            f"{variant_name:<10} {agg['der']:>7.3f} {agg['wder']:>7.3f} {agg['cpwer']:>7.3f} "
            f"{agg['wer']:>7.3f} {agg['rtf']:>7.3f} {agg['osd_f1']:>7.3f} {agg['routed_fraction']*100:>7.1f}%"
        )


def main():
    force_utf8_stdio()
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
    args = parser.parse_args()
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    if args.synthetic is None and not args.dataset and args.parquet is None:
        args.synthetic = 5  # default to a quick synthetic smoke test if nothing was specified

    run(args)


if __name__ == "__main__":
    main()
