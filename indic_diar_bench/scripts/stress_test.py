"""Long-running resilience/correctness test: runs the PROPOSED variant (pretrained
backend, the actual system this project proposes -- not a baseline) across a broad,
round-robin sample of real Indic DiarBench recordings spanning all requested languages
and acoustic conditions, catching and logging every per-recording failure instead of
crashing the whole run, so it can run unattended for hours.

Nothing about which languages/conditions/thresholds run is hardcoded into this script's
logic -- every one of those is a CLI flag with the full dataset's language list
(`data/prep/manifest.py::ALL_LANGUAGES`) and all three conditions as the default scope,
and the per-recording ASR language hint is resolved dynamically via
`data/prep/language_codes.py` rather than assumed.

Usage:
  # default: all 22 languages, all 3 conditions, round-robin, up to a 3-hour time budget
  python scripts/stress_test.py

  # narrower run for a quick check:
  python scripts/stress_test.py --languages Hindi Tamil --time-budget-hours 0.5

Output (in --output-dir, default outputs/stress_test/):
  results.jsonl   one line per successfully-processed recording (metrics + timing)
  failures.jsonl  one line per recording that raised an exception (full traceback)
  summary.json    running aggregate stats, rewritten after every recording
"""

from __future__ import annotations

import os

# MUST happen before the first `import torch` anywhere in the process (even a lazy one
# inside a function called much later) -- MKL/OMP read these env vars once, at native
# library init time, not per-call. Confirmed necessary this session: with the default
# (all 12 logical cores) thread count, `RuntimeError: mkl_malloc: failed to allocate
# memory` occurred on the very FIRST recording of a freshly-started process (not after
# many recordings, ruling out a slow leak as the sole cause) -- MKL allocates per-thread
# scratch buffers for BLAS/attention ops, so with 4 models loaded simultaneously
# (pyannote + ECAPA + SepFormer + Whisper) on a 7.3GB-RAM machine, 12x the scratch buffer
# size was enough to blow the memory budget on an ordinary transient allocation. Capped at
# 2 threads: a deliberate speed/memory tradeoff (slower matmuls, far less scratch memory)
# validated on this machine, not a universal constant -- raise it on a machine with more
# headroom.
_CPU_THREAD_LIMIT = "2"
os.environ.setdefault("OMP_NUM_THREADS", _CPU_THREAD_LIMIT)
os.environ.setdefault("MKL_NUM_THREADS", _CPU_THREAD_LIMIT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _CPU_THREAD_LIMIT)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _CPU_THREAD_LIMIT)

import argparse
import json
import sys
import time
import traceback
from itertools import cycle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prep.language_codes import language_to_whisper_code
from data.prep.manifest import ALL_LANGUAGES
from eval.evaluate import evaluate_recording
from pipeline.config import Backend
from pipeline.variants import build_variant
from schemas.types import (
    AcousticCondition,
    ManifestEntry,
    SpeakerAttributedTranscript,
    Utterance,
)
from scripts._stdio import force_utf8_stdio

_CONDITION_NAME_MAP = {
    "near_field": AcousticCondition.NEAR_FIELD,
    "far_field": AcousticCondition.FAR_FIELD,
    "in_the_wild": AcousticCondition.IN_THE_WILD,
}


def truncate_recording(entry: ManifestEntry, audio, sample_rate: int, max_seconds: float | None):
    """Clips a recording (and its reference annotations, for a fair eval) to at most
    `max_seconds`. Needed for broad coverage within a time budget: some real dataset
    recordings are long enough, and/or overlap-heavy enough to route a large fraction to
    the (CPU-expensive) separation branch, that processing one recording in full can take
    far longer than the per-recording budget implied by "cover all N languages in T hours"
    -- confirmed this session (one recording took >15 minutes with the fast 'tiny' ASR
    config). None/<=0 disables truncation (process the full recording)."""
    if max_seconds is None or max_seconds <= 0:
        return entry, audio
    max_samples = int(max_seconds * sample_rate)
    if len(audio) <= max_samples:
        return entry, audio

    truncated_audio = audio[:max_samples]
    kept_segments = [s for s in entry.reference_segments if s.start < max_seconds]
    kept_segments = [
        s if s.end <= max_seconds else type(s)(s.start, max_seconds, s.speaker) for s in kept_segments
    ]

    kept_utterances: list[Utterance] = []
    if entry.reference_transcript:
        for u in entry.reference_transcript.utterances:
            if u.start >= max_seconds:
                continue
            kept_words = [w for w in u.words if w.start < max_seconds]
            if kept_words:
                kept_utterances.append(Utterance(speaker=u.speaker, start=u.start, end=min(u.end, max_seconds), words=kept_words))
    truncated_transcript = SpeakerAttributedTranscript(recording_id=entry.recording_id, utterances=kept_utterances)

    truncated_entry = ManifestEntry(
        recording_id=entry.recording_id,
        audio_path=entry.audio_path,
        language=entry.language,
        condition=entry.condition,
        duration=min(entry.duration, max_seconds),
        reference_segments=kept_segments,
        reference_transcript=truncated_transcript,
    )
    return truncated_entry, truncated_audio


def round_robin_recordings(languages: list[str], conditions: list[AcousticCondition] | None):
    """Interleaves per-language streaming iterators so the run gets broad language
    coverage early instead of exhausting one language before starting the next.

    Network/stream errors (confirmed this session: a HuggingFace CDN read timing out mid
    -download of a large parquet shard escalated into a MemoryError deep inside `requests`,
    which is NOT a StopIteration and was NOT caught here originally -- it propagated straight
    out of this generator and crashed the entire multi-hour run after only 5 recordings).
    Any exception from a per-language sub-iterator now just drops that language from the
    rotation (logged, not silent) so a transient network failure on one language can't take
    down the whole run.
    """
    from data.prep.manifest import load_indic_diarbench

    iterators = {lang: load_indic_diarbench(languages=[lang], conditions=conditions, streaming=True) for lang in languages}
    active = dict(iterators)
    for lang in cycle(list(active.keys())):
        if not active:
            return
        if lang not in active:
            continue
        try:
            yield next(active[lang])
        except StopIteration:
            active.pop(lang, None)
        except Exception as e:  # noqa: BLE001 -- a broken stream for one language must not kill the whole run
            print(f"  (stream error for {lang}, dropping it from rotation: {type(e).__name__}: {e})")
            active.pop(lang, None)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_processed_ids(results_path: Path, failures_path: Path) -> set[str]:
    seen: set[str] = set()
    for path in (results_path, failures_path):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["recording_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def write_summary(summary_path: Path, stats: dict) -> None:
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    import torch

    torch.set_num_threads(int(_CPU_THREAD_LIMIT))  # reinforces the env vars set at module import time (see top of file)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    failures_path = out_dir / "failures.jsonl"
    summary_path = out_dir / "summary.json"

    languages = args.languages if args.languages else ALL_LANGUAGES
    conditions = (
        [_CONDITION_NAME_MAP[c] for c in args.conditions] if args.conditions else None
    )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cfg = build_variant(
        args.variant, backend=Backend.PRETRAINED, asr_model_size=args.asr_model_size,
        hf_token=hf_token, max_overlap_speakers=args.max_overlap_speakers,
    )

    print(f"Loading models for variant={args.variant} (this can take a while)...")
    from pipeline.orchestrator import OverlapAwarePipeline

    pipeline = OverlapAwarePipeline(cfg)

    already_processed = load_processed_ids(results_path, failures_path) if args.resume else set()
    if already_processed:
        print(f"Resuming: {len(already_processed)} recording(s) already processed, will be skipped.")

    time_budget_seconds = args.time_budget_hours * 3600
    start_time = time.monotonic()

    n_success = 0
    n_failure = 0
    n_skipped_short = 0
    per_language_counts: dict[str, int] = {}
    error_type_counts: dict[str, int] = {}
    der_values: list[float] = []
    rtf_values: list[float] = []

    print(f"Time budget: {args.time_budget_hours}h. Languages: {len(languages)}. "
          f"Conditions: {[c.value for c in conditions] if conditions else 'all'}.")
    print(f"Logging to: {results_path}, {failures_path}, {summary_path}\n")

    exit_reason = "dataset_exhausted"  # default if the round-robin generator itself runs out

    for entry, audio, sr in round_robin_recordings(languages, conditions):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_budget_seconds:
            print(f"\nTime budget of {args.time_budget_hours}h reached. Stopping.")
            exit_reason = "time_budget"
            break
        if args.max_recordings is not None and (n_success + n_failure) >= args.max_recordings:
            print(f"\nReached --max-recordings={args.max_recordings}. Stopping.")
            exit_reason = "max_recordings"
            break
        if args.recordings_per_attempt and (n_success + n_failure) >= args.recordings_per_attempt:
            print(f"\nReached --recordings-per-attempt={args.recordings_per_attempt}. "
                  f"Exiting cleanly for a memory-hygiene restart (supervisor will relaunch with --resume).")
            exit_reason = "periodic_restart"
            break

        if entry.recording_id in already_processed:
            continue

        if len(audio) == 0:
            n_skipped_short += 1
            continue

        entry, audio = truncate_recording(entry, audio, sr, args.max_audio_seconds)

        language_hint = language_to_whisper_code(entry.language)
        t0 = time.perf_counter()
        try:
            transcript, stats, predicted_overlap = pipeline.run(
                audio, sr, recording_id=entry.recording_id, language=language_hint
            )
            result = evaluate_recording(entry, transcript, predicted_overlap, stats)
            record = {
                "recording_id": entry.recording_id,
                "language": entry.language,
                "condition": entry.condition.value,
                "duration": entry.duration,
                "der": result.der,
                "wder": result.wder,
                "cpwer": result.cpwer,
                "wer": result.wer,
                "rtf": result.rtf,
                "osd_f1": result.osd_f1,
                "routed_fraction": result.routed_fraction,
                "wall_clock_seconds": time.perf_counter() - t0,
            }
            append_jsonl(results_path, record)
            n_success += 1
            der_values.append(result.der)
            rtf_values.append(result.rtf)
            per_language_counts[entry.language] = per_language_counts.get(entry.language, 0) + 1
            print(f"[OK {n_success}] {entry.recording_id} ({entry.language}/{entry.condition.value}, "
                  f"{entry.duration:.0f}s) DER={result.der:.3f} RTF={result.rtf:.2f}")
        except Exception as e:  # noqa: BLE001 -- must keep the multi-hour run alive on any single-recording failure
            tb = traceback.format_exc()
            error_type = type(e).__name__
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            failure_record = {
                "recording_id": entry.recording_id,
                "language": entry.language,
                "condition": entry.condition.value,
                "duration": entry.duration,
                "error_type": error_type,
                "error_message": str(e),
                "traceback": tb,
                "wall_clock_seconds": time.perf_counter() - t0,
            }
            append_jsonl(failures_path, failure_record)
            n_failure += 1
            print(f"[FAIL {n_failure}] {entry.recording_id} ({entry.language}/{entry.condition.value}) "
                  f"-- {error_type}: {e}")

        if (n_success + n_failure) % 5 == 0:
            write_summary(summary_path, {
                "elapsed_hours": round((time.monotonic() - start_time) / 3600, 3),
                "n_success": n_success,
                "n_failure": n_failure,
                "n_skipped_empty_audio": n_skipped_short,
                "mean_der": sum(der_values) / len(der_values) if der_values else None,
                "mean_rtf": sum(rtf_values) / len(rtf_values) if rtf_values else None,
                "per_language_counts": per_language_counts,
                "error_type_counts": error_type_counts,
            })

    # "finished" is the signal run_stress_test_supervised.sh watches to decide whether to
    # relaunch -- it must be True only when this attempt is GENUINELY done (ran out of
    # time/recordings/dataset), not for a periodic_restart, which is a voluntary exit to
    # release accumulated memory (confirmed this session: faster-whisper's `mkl_malloc:
    # failed to allocate memory` started recurring after enough recordings in one process
    # lifetime on this machine's 7.3GB RAM -- the per-recording try/except correctly kept
    # the run alive each time, but repeated failures meant memory pressure was building
    # faster than any single recording could account for). A periodic_restart must look
    # like "not finished yet" to the supervisor so it keeps relaunching with --resume.
    write_summary(summary_path, {
        "elapsed_hours": round((time.monotonic() - start_time) / 3600, 3),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_skipped_empty_audio": n_skipped_short,
        "mean_der": sum(der_values) / len(der_values) if der_values else None,
        "mean_rtf": sum(rtf_values) / len(rtf_values) if rtf_values else None,
        "per_language_counts": per_language_counts,
        "error_type_counts": error_type_counts,
        "exit_reason": exit_reason,
        "finished": exit_reason != "periodic_restart",
    })
    print(f"\nDone ({exit_reason}). {n_success} succeeded, {n_failure} failed, {n_skipped_short} skipped (empty audio).")
    if error_type_counts:
        print("Error types seen:", error_type_counts)


def main():
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="*", default=None,
                         help=f"Languages to test (default: all {len(ALL_LANGUAGES)} in the dataset).")
    parser.add_argument("--conditions", nargs="*", default=None, choices=list(_CONDITION_NAME_MAP),
                         help="Acoustic conditions to test (default: all three).")
    parser.add_argument("--variant", type=str, default="PROPOSED",
                         help="System variant to stress-test (default: PROPOSED, the actual proposed system).")
    parser.add_argument("--asr-model-size", type=str, default="tiny",
                         help="faster-whisper model size (default: tiny -- this run prioritizes broad "
                              "recording/language/condition coverage for bug-finding within the time "
                              "budget over transcription quality; 'small' took ~14x real-time per "
                              "recording in this session's validation, which would cap coverage too low).")
    parser.add_argument("--max-overlap-speakers", type=int, default=2)
    parser.add_argument("--hf-token", type=str, default=None, help="Defaults to HF_TOKEN env var.")
    parser.add_argument("--time-budget-hours", type=float, default=3.0,
                         help="Stop after roughly this many hours of wall-clock testing (default: 3.0).")
    parser.add_argument("--max-recordings", type=int, default=None,
                         help="Optional hard cap on the number of recordings processed, independent of time budget.")
    parser.add_argument("--recordings-per-attempt", type=int, default=12,
                         help="Voluntarily exit (not counted as 'finished' -- the supervisor relaunches with "
                              "--resume) after this many recordings, to release memory accumulated over a long "
                              "process lifetime before it causes a crash (confirmed this session: faster-whisper "
                              "started raising 'mkl_malloc: failed to allocate memory' after enough recordings "
                              "in one run on this machine's limited RAM). Set to 0 to disable periodic restarts "
                              "(only sensible with --time-budget-hours short enough that memory pressure can't "
                              "build up, or on a machine with much more RAM).")
    parser.add_argument("--max-audio-seconds", type=float, default=45.0,
                         help="Clip each recording to at most this many seconds before processing, so one long "
                              "or heavily-overlapping recording can't consume a disproportionate share of the "
                              "time budget and starve coverage of other languages (default: 45s; 0/negative "
                              "disables truncation and processes full recordings).")
    parser.add_argument("--output-dir", type=str, default="outputs/stress_test")
    parser.add_argument("--resume", action="store_true",
                         help="Skip recording_ids already present in results.jsonl/failures.jsonl from a prior run.")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
