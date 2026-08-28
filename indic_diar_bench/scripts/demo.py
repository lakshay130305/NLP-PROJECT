"""Interactive CLI demo: point at one or more audio files (or a directory of them) and
get back a speaker-attributed transcript for each -- no ground truth / dataset needed.

This is the "just show me it working on my own audio" entry point, as opposed to
run_variant.py which is for benchmarking against Indic DiarBench with ground truth.

Usage:
  # interactive mode -- prompts for file paths, one per line, blank line to run:
  python scripts/demo.py

  # non-interactive: pass file(s)/folder(s) directly as arguments:
  python scripts/demo.py meeting.wav
  python scripts/demo.py call1.mp3 call2.wav
  python scripts/demo.py ./my_recordings/

  # quick structural smoke test with no model downloads:
  python scripts/demo.py meeting.wav --backend dummy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prep.language_codes import (
    LANGUAGE_TO_ISO639_1,
    whisper_supported_language_codes,
)
from pipeline.audio_utils import load_audio_file
from pipeline.config import Backend
from pipeline.variants import build_variant
from scripts._stdio import force_utf8_stdio

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a", ".aac", ".opus", ".webm"}


def prompt_for_paths() -> list[Path]:
    print("Enter audio file path(s) -- one per line. Blank line when you're done.")
    paths: list[Path] = []
    while True:
        try:
            line = input("  > ").strip().strip('"')
        except EOFError:
            break
        if not line:
            break
        p = Path(line)
        if not p.exists():
            print(f"    (not found: {p}, skipping)")
            continue
        paths.append(p)
    return paths


def expand_paths(raw_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in raw_paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
            if not found:
                print(f"(no audio files found in {p})")
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            print(f"(not found: {p}, skipping)")
    return files


def resolve_hf_token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN")


def resolve_language(raw: str | None) -> str | None:
    """Accepts either a language name (e.g. "Hindi", case-insensitive) or a raw Whisper
    language code (e.g. "hi") and returns a code Whisper actually supports, or None
    (auto-detect) if `raw` is empty or doesn't resolve to a supported code. A hint here
    matters a lot for accuracy on short/noisy real-world audio: unconstrained language
    auto-detection can flip between languages mid-recording (confirmed this session on a
    real Hindi sample transcribed with the tiny model and no hint -- it hallucinated
    fragments in Japanese, Korean, and Urdu script alongside the actual Hindi)."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None

    supported = whisper_supported_language_codes()
    by_name = {name.lower(): code for name, code in LANGUAGE_TO_ISO639_1.items()}

    code = by_name.get(raw.lower()) or (raw.lower() if raw.lower() in supported else None)
    if code is None or code not in supported:
        print(f"  (language '{raw}' not recognized/supported by the ASR backend -- falling back to auto-detect)")
        return None
    return code


def format_transcript(transcript, audio_duration: float) -> str:
    lines = [f"Recording: {transcript.recording_id}  ({audio_duration:.1f}s)", "-" * 60]
    for u in transcript.sorted_by_time():
        ts = f"[{u.start:6.2f} - {u.end:6.2f}]"
        lines.append(f"{ts}  {u.speaker:<12} {u.text}")
    if not transcript.utterances:
        lines.append("(no speech detected)")
    return "\n".join(lines)


def transcript_to_dict(transcript) -> dict:
    return {
        "recording_id": transcript.recording_id,
        "utterances": [
            {
                "speaker": u.speaker,
                "start": u.start,
                "end": u.end,
                "text": u.text,
                "words": [
                    {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence}
                    for w in u.words
                ],
            }
            for u in transcript.sorted_by_time()
        ],
    }


def run(args: argparse.Namespace) -> None:
    if args.paths:
        audio_files = expand_paths(args.paths)
    else:
        audio_files = prompt_for_paths()

    if not audio_files:
        print("No audio files to process. Exiting.")
        return

    language_raw = args.language
    if language_raw is None and not args.paths:
        # only prompt when we're already in interactive mode (no CLI paths given) --
        # a non-interactive invocation should never block on stdin
        try:
            language_raw = input(
                "What language is this audio in? (name or code, e.g. 'Hindi' or 'hi'; "
                "blank = auto-detect, less accurate on short/noisy audio) > "
            ).strip()
        except EOFError:
            language_raw = None
    language = resolve_language(language_raw)
    if language:
        print(f"Using language hint: {language}\n")

    backend = Backend.PRETRAINED if args.backend == "pretrained" else Backend.DUMMY
    hf_token = resolve_hf_token(args.hf_token)

    if backend == Backend.PRETRAINED and hf_token is None:
        print(
            "Note: no HF token found (HF_TOKEN env var / --hf-token). "
            "pyannote's gated VAD/OSD model will fail to load without one; "
            "everything else (embeddings, separation, ASR) will still work.\n"
        )

    cfg = build_variant(
        "PROPOSED", backend=backend, asr_model_size=args.asr_model_size, hf_token=hf_token
    )

    print(f"Loading models (backend={backend.value})... this can take a while on first run.")
    from pipeline.orchestrator import OverlapAwarePipeline

    pipeline = OverlapAwarePipeline(cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(audio_files)} file(s)...\n")
    for i, path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {path.name}")
        t0 = time.perf_counter()
        try:
            audio, sr = load_audio_file(path)
        except Exception as e:  # noqa: BLE001 -- keep processing the rest of the batch on a bad file
            print(f"  FAILED to load audio: {e}\n")
            continue

        if len(audio) == 0:
            print("  (empty/silent audio, skipping)\n")
            continue

        recording_id = path.stem
        transcript, stats, _overlap_regions = pipeline.run(audio, sr, recording_id=recording_id, language=language)
        elapsed = time.perf_counter() - t0

        print()
        print(format_transcript(transcript, stats.audio_duration))
        print(f"\n  (processed in {elapsed:.1f}s, RTF={stats.rtf:.2f})\n")

        txt_path = out_dir / f"{recording_id}.transcript.txt"
        json_path = out_dir / f"{recording_id}.transcript.json"
        txt_path.write_text(format_transcript(transcript, stats.audio_duration), encoding="utf-8")
        json_path.write_text(json.dumps(transcript_to_dict(transcript), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved: {txt_path}")
        print(f"  Saved: {json_path}\n")
        print("-" * 60 + "\n")

    print("Done.")


def main():
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Audio file(s) or folder(s). If omitted, prompts interactively.")
    parser.add_argument("--backend", choices=["dummy", "pretrained"], default="pretrained",
                         help="pretrained = real models (default, needed for a real transcript); "
                              "dummy = no downloads, structural smoke test only.")
    parser.add_argument("--asr-model-size", type=str, default="small",
                         help="faster-whisper model size: tiny/base/small/medium/large-v3 (default: small).")
    parser.add_argument("--language", type=str, default=None,
                         help="Known language of the audio (name like 'Hindi' or code like 'hi'). "
                              "Applies to ALL files in this run. Omit to auto-detect (less reliable "
                              "on short/noisy audio, and prompted for interactively if not given).")
    parser.add_argument("--hf-token", type=str, default=None,
                         help="HuggingFace token for gated pyannote VAD/OSD. Defaults to HF_TOKEN env var.")
    parser.add_argument("--output-dir", type=str, default="outputs/transcripts",
                         help="Where to save .txt/.json transcripts (default: outputs/transcripts).")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
