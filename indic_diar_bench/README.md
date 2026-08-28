# Overlap-Aware Joint Diarization-ASR Framework -- Indic DiarBench

Implementation of the overlap-aware selective-routing diarization-ASR pipeline described
in the project plan (`Indic_DiarBench_Paper3_Overlap_Aware_Joint_Diarization_ASR.pdf`):
VAD -> Overlapped Speech Detection -> Selective Router -> {standard diarization | speech
separation + multi-speaker attribution} -> multilingual ASR -> transcript fusion -> eval.

## Status (MVP)

**Both backends are now confirmed working end-to-end on real data**, not just synthetic:

- All 6 system variants (B1, B2, B3, B4, B5, PROPOSED -- section 11.3) run cleanly with
  the `dummy` backend (zero network/model downloads) on synthetic overlap audio, producing
  the full section-13.1 metrics table.
- 55 unit/integration tests pass (`pytest tests/`).
- **The `pretrained` backend (real pyannote/speechbrain/faster-whisper models) was
  exercised end-to-end this session** on a real downloaded recording
  (`hindi_nf_001`, 75s, from `sarvamai/indic-diarbench`) across every variant --
  B1 through PROPOSED all completed without error. Results trended the direction the
  paper's hypothesis predicts (DER: B1=0.438 -> B4=0.410 -> PROPOSED=0.394 on this one
  clip), though this is one short recording with the smallest Whisper checkpoint on CPU --
  not a real benchmark result, just strong evidence the whole system is wired correctly.
  **Five real, previously-latent bugs were found and fixed getting here** (see "Bugs found
  and fixed this session" below) -- none of the pretrained code paths had been run against
  a real model or real audio before this session, so this was the first real test of
  those assumptions, and it caught genuine mistakes.
- The real dataset loader (`data/prep/manifest.py`) is confirmed working end-to-end.

**Still not done:**
- Running at any real scale (more than 1-3 recordings) -- SepFormer + Whisper on CPU is
  slow (RTF 1.3-5.6x on the one clip tested here), so a real multi-recording run belongs
  on a GPU (see "Next steps").
- Multilingual/acoustic-condition/speaker-count breakdown scripts (sections 14-16),
  ablation sweep scripts (section 17), significance-testing driver script tying
  `eval/significance.py` into a report (section 18) -- the metric functions exist and are
  tested, but no CLI wires them to a full dataset run yet.
- Case-study / error-analysis tooling (section 19).
- Per-word tokenization of the dataset's segment-level transcripts (see "Dataset" below) --
  needed for WDER to be meaningful, not just DER/cpWER.

### Bugs found and fixed this session (pretrained backend, all confirmed against real data)

1. **pyannote model call convention was wrong.** `PyannoteVAD`/`PyannoteOSD` called
   `model({"waveform": ..., "sample_rate": ...})`, which crashes
   (`AttributeError: 'dict' object has no attribute 'dim'`). The real API is
   `model(waveform_tensor)`. Fixed in `pipeline/_pyannote_powerset.py`.
2. **Overlap-class detection used a wrong hardcoded index.** The original OSD code summed
   `probs[..., 2:]` to mean "overlap", but segmentation-3.0's powerset classes are
   `[none, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3]` -- classes 2 and 3 are each a
   *single* speaker, not overlap. Fixed by reading `Powerset.mapping` to find classes with
   >=2 active speakers generically (indices 4,5,6 for this checkpoint), so it stays correct
   for any base-class/max-class configuration.
3. **speechbrain's default fetch strategy needs Windows Developer Mode.** `EncoderClassifier`/
   `SepformerSeparation.from_hparams()` default to `LocalStrategy.SYMLINK`, which raises
   `OSError: [WinError 1314] A required privilege is not held by the client` without admin/
   Developer Mode. Fixed by passing `local_strategy=LocalStrategy.COPY` explicitly.
4. **Very short real VAD segments crash convolutional models.** Real audio produced a 34ms
   VAD segment; ECAPA-TDNN's SincNet front-end and faster-whisper's word-alignment both
   raised errors on inputs that short (too few frames for their conv padding / DTW
   alignment). Fixed with `pipeline/audio_utils.py::pad_to_min_length()`, applied before
   every pretrained embedding/separation/ASR call.
5. **Soft routing never actually routed anything.** `SelectiveRouter`'s soft mode decided
   the overlap/single branch by the segment's *mean* OSD probability across its whole
   duration -- which dilutes a short real overlap interval inside a multi-second VAD
   segment to near-zero, so 0% of real audio ever reached the separation branch. Fixed so
   both hard and soft modes select the branch by whether the segment intersects any
   detected overlap *region* (matching hard mode); soft mode still carries the continuous
   probability as `overlap_weight` for downstream blending per the h_t formula (section 6.3).
   After the fix, real audio routed 27-31% of its duration to the separation branch, in
   line with the ~7.5% frame-level overlap rate the OSD found on that recording (VAD
   segments are multi-speaker-turn-length, so a higher segment-level routed fraction than
   the raw frame-level overlap rate is expected).
6. **Windows console can't print most real transcripts.** Printing an actual ASR result
   crashed with `UnicodeEncodeError: 'charmap' codec can't encode character` -- Windows
   defaults to a legacy codepage (cp1252 here) that can't encode Devanagari/Bengali/Tamil/
   etc. script, or in fact most non-Latin-1 text at all. Not an edge case for a
   22-Indian-language project. Fixed with `scripts/_stdio.py::force_utf8_stdio()`, called
   first thing in every CLI entry point.
7. **ASR never used the recording's known language.** Even though `ManifestEntry.language`
   carries the ground-truth language name, nothing passed it to Whisper -- every
   transcription ran with blind auto-detection. Confirmed this makes a real difference: a
   real Hindi recording transcribed with `tiny` + no hint produced fragments in Japanese,
   Korean, and Urdu script mixed with the actual Hindi (language ID flipping mid-recording
   on short/noisy segments); with the hint applied, output stayed in Devanagari throughout.
   Fixed by (a) `pipeline/orchestrator.py::OverlapAwarePipeline.run()` accepting a
   per-call `language` override so one pipeline instance can be reused across a
   multilingual batch, and (b) `data/prep/language_codes.py`, which maps each of the 22
   dataset languages to a Whisper code -- checked live against the installed
   faster-whisper build's actual supported-code set (`whisper_supported_language_codes()`)
   rather than assumed, so it can't silently claim support for a code that isn't real. 14
   of the 22 languages have a Whisper code; the other 8 (Bodo, Dogri, Kashmiri, Konkani,
   Maithili, Manipuri, Odia, Santali) correctly fall back to auto-detect.

### Bugs found and fixed during the dataset-wide stress test (`scripts/stress_test.py`)

Running `PROPOSED` across a round-robin sample of the real dataset (not just one clip)
surfaced three more real bugs, in roughly increasing order of severity:

8. **One long/overlap-heavy recording could starve coverage of every other language.**
   The first (untruncated) run spent >15 minutes on a single Assamese recording without
   finishing it -- with 22 languages to cover in a 3h budget, that pace would have left
   several languages (alphabetically later ones especially) completely untested. Fixed
   with `scripts/stress_test.py::truncate_recording()` (`--max-audio-seconds`, default 45),
   which clips both the audio and its reference annotations consistently before
   processing, so every language gets exercised within the time budget instead of a few
   languages getting deep coverage and the rest getting none.
9. **A network failure could crash the entire multi-hour run, not just one recording.**
   The per-recording `try/except` in the main loop only protects the *processing* of an
   already-yielded recording -- it does NOT protect the `for entry, audio, sr in
   round_robin_recordings(...)` loop's own iteration, which is where the actual HTTP
   streaming happens. Confirmed live: a HuggingFace CDN timeout while downloading a large
   Gujarati parquet shard escalated (through `requests`' redirect-retry path, which buffers
   the full response into memory) into a `MemoryError`, which propagated straight out of
   the generator and killed the whole script after only 5 recordings. Fixed by moving the
   `try/except` inside `round_robin_recordings()` itself, around each per-language `next()`
   call -- a broken stream now just drops that one language from the round-robin rotation
   (logged) instead of crashing the run.
10. **The pretrained model stack can crash at the OS level on this machine, which Python
    cannot catch at all.** This machine has 7.3GB total RAM (~2.5GB free with the model
    stack loaded) -- confirmed via `Get-CimInstance Win32_OperatingSystem`. A relaunch
    (after fixing #9) died with a bare `Segmentation fault` (exit code 139) shortly after
    model load, before processing any recording. A SIGSEGV kills the process outright; no
    Python-level exception handling, however broad, can catch it. **This is not a code bug
    to "fix" so much as a resource constraint to survive** -- so instead of patching the
    unfixable, `scripts/run_stress_test_supervised.sh` wraps `stress_test.py` in a
    supervisor loop that tracks a fixed wall-clock deadline (not a per-attempt budget) and
    relaunches with `--resume` (skips recordings already in `results.jsonl`/
    `failures.jsonl`) whenever the process dies for any reason short of a clean finish
    (detected via `summary.json`'s `"finished": true`, written unconditionally at the end
    of a normal `run()`). It aborts early only if 5 consecutive attempts each die within
    30s of starting, since that pattern means a persistent bug, not a transient crash, and
    continuing would just burn the deadline in a crash loop. Validated with a short dry run
    before committing to the full 3h attempt (see `outputs/stress_test/` for the real run's
    results -- run `python scripts/summarize_stress_test.py` for a live-updating summary).
11. **First hypothesis (revised by #12 below): memory pressure builds up gradually within
    one long process lifetime.** After the fix for #10, a continued run hit
    `RuntimeError: mkl_malloc: failed to allocate memory` (Intel MKL, PyTorch's CPU
    backend, failing to allocate) three times in a row across different recordings/
    languages -- the per-recording `try/except` correctly kept the run alive each time
    (it moved on to the next recording rather than crashing). Mitigated with
    `--recordings-per-attempt` (default 12): the process voluntarily exits after that many
    recordings -- written to `summary.json` as `"exit_reason": "periodic_restart"` with
    `"finished": false` (deliberately, so the supervisor's "relaunch unless finished=true"
    logic treats it like any other non-fatal exit and relaunches with `--resume`). Kept as
    a real, useful safety net even after #12's more targeted fix, since it bounds peak
    memory regardless of root cause.
12. **The actual root cause of #11 wasn't a slow leak -- it was thread count.** On the very
    next relaunch, `mkl_malloc` recurred on the FIRST recording of a completely fresh
    process (ruling out "many recordings accumulate memory" as the sole explanation).
    This machine has 12 logical cores; PyTorch/MKL default to intra-op parallelism across
    all of them, and MKL allocates per-thread scratch buffers for BLAS/attention
    operations -- with 4 models loaded simultaneously (pyannote + ECAPA + SepFormer +
    Whisper) on 7.3GB RAM, 12x the scratch buffer size for an ordinary transient
    allocation was enough to exceed the memory budget outright, independent of how many
    recordings had run before it. Fixed at the top of `scripts/stress_test.py` (before any
    `import torch` anywhere in the process, since MKL/OMP read these once at native
    library init) by setting `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
    `NUMEXPR_NUM_THREADS` to `"2"`, reinforced with an explicit `torch.set_num_threads(2)`
    call in `run()`. A deliberate speed/memory tradeoff tuned for *this* 7.3GB machine, not
    a universal constant -- raise it on a machine with more headroom.
13. **A background process survived two session restarts without the harness noticing,
    running a stale (unfixed) version of the code.** After the session itself restarted
    (unrelated to this project), `TaskStop` on the tracked task ID returned success but
    the actual OS process tree (`bash` -> `stress_test.py` -> its Python subprocess) kept
    running underneath, for over an hour, invisibly -- because the harness's task tracking
    doesn't survive a session restart, but the OS process it originally launched does. It
    kept hitting the exact same `mkl_malloc` failure repeatedly (confirmed via
    `Get-CimInstance Win32_Process` showing its live command line) because the thread-limit
    fix (#12) had been written to a file but the fix wasn't applied until AFTER that -- so
    the running process never had it. Resolved with `Stop-Process -Force` on the actual PIDs
    (found via `Get-CimInstance Win32_Process`, not just the harness's task list) before
    relaunching. **Practical lesson for anyone continuing this work**: after any session
    interruption, check for orphaned processes (`Get-Process python,bash` /
    `Get-CimInstance Win32_Process` to see real command lines) before assuming a
    `TaskStop` actually stopped anything, and before trusting `outputs/stress_test/`'s
    file contents to reflect only the code currently on disk.

## Environment notes (Windows, CPU-only)

This machine has no usable CUDA GPU. The originally-installed `torch==2.4.0+cu121` failed
to import at all (`OSError: ... fbgemm.dll ... libomp140.x86_64.dll`) because the CUDA
build's native deps aren't satisfiable without a GPU driver. Fixed by installing the
**CPU-only** wheel matching what `whisperx` pins:

```
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
```

(`requirements.txt` already encodes this.) If you ever add a CUDA GPU, reinstall the
matching `+cu*` wheels instead and pass `device="cuda"` to `FasterWhisperASR` /
`ECAPAEmbedder` / etc.

## Repo layout

```
schemas/      Core dataclasses (SpeechSegment, WordToken, ManifestEntry, ...) + overlap math (3.5)
eval/         DER, WDER, cpWER, WER, OSD precision/recall, RTF/efficiency, significance (12, 18)
pipeline/     Every pipeline stage (4, 5-9) as swappable dummy/pretrained backends,
              orchestrator that composes them, and the B1..B5/Proposed variant table (11.3)
data/prep/    Real-dataset loader (sarvamai/indic-diarbench), language-code mapping,
              synthetic overlap generator
scripts/      demo.py (interactive: your own audio -> transcript), run_variant.py
              (benchmark against the dataset), stress_test.py (long-running resilience
              test across the whole dataset), summarize_stress_test.py
tests/        pytest suite (55 tests, dummy backend, no network needed)
```

## Demo: transcribe your own audio

`scripts/demo.py` is the "just show me it working" entry point -- point it at audio
file(s) and get a speaker-attributed transcript back, no dataset/ground-truth needed.
(Live microphone recording is planned for a later frontend stage; this is upload/file-path
only for now.)

```bash
# interactive: prompts for file paths (one per line, blank line to run) and for the
# audio's language (optional but recommended -- see "Bugs found" #7 below for why):
python scripts/demo.py

# non-interactive: pass file(s)/folder(s) directly, plus a language hint:
python scripts/demo.py meeting.wav --language Hindi
python scripts/demo.py call1.mp3 call2.wav interview.m4a  # mixed formats OK
python scripts/demo.py ./my_recordings/                    # whole folder

# quick structural check, no model downloads:
python scripts/demo.py meeting.wav --backend dummy
```

Prints a `[start - end]  SPEAKER  text` transcript per file to the console and saves both
`.txt` and `.json` versions to `outputs/transcripts/` (`--output-dir` to change). Accepts
WAV/FLAC/OGG natively and falls back to PyAV (bundles its own decoder, no system FFmpeg
needed) for MP3/M4A/AAC/etc. -- confirmed working on a real encoded MP3 this session.

## Running it

```bash
# smoke test, no network/model downloads:
python scripts/run_variant.py --synthetic 5 --variants B1,PROPOSED

# all 6 variants:
python scripts/run_variant.py --synthetic 5

# real dataset (needs network; streams from HuggingFace, no full download by default):
python scripts/run_variant.py --dataset --languages Hindi --limit 10 --variants B1,B2,B3,B4,B5,PROPOSED

# real pretrained models instead of the dummy backend (slow on CPU -- start with 'tiny' Whisper):
python scripts/run_variant.py --dataset --languages Hindi --limit 3 --backend pretrained --asr-model-size tiny
```

Run tests: `python -m pytest tests/ -q`

## The dummy vs. pretrained backend split

Every ML-backed stage (VAD, OSD, speaker embedding, separation, ASR) has two
implementations behind the same `Protocol` interface (`pipeline/interfaces.py`):

| Stage      | Dummy (default, offline)         | Pretrained (`--backend pretrained`)        |
|------------|-----------------------------------|---------------------------------------------|
| VAD        | `EnergyVAD` (frame energy)        | `PyannoteVAD` (pyannote/segmentation-3.0)    |
| OSD        | `EnergyRatioOSD` (spectral flatness proxy) | `PyannoteOSD` (same model, powerset overlap classes) |
| Embedding  | `MFCCStatsEmbedder` (spectral-band stats) | `ECAPAEmbedder` (speechbrain ECAPA-TDNN)     |
| Separation | `NullSeparator` (passthrough)     | `SepFormerSeparator` (speechbrain SepFormer) |
| ASR        | `RegexEnergyASR` (placeholder tokens) | `FasterWhisperASR` (CTranslate2 Whisper)     |

The dummy backend exists purely so the *plumbing* (routing, separation-branch wiring,
attribution, timestamp restoration, transcript fusion, every metric) can be verified
without a model download or GPU -- it will NOT produce meaningful DER/WER numbers on
real speech. Once you have network access and (for the gated pyannote models) an HF
token, flip `--backend pretrained` to get real diarization/ASR quality.

`pyannote/segmentation-3.0` is a gated model on HuggingFace -- accept its license (and
`pyannote/speaker-diarization-3.1`'s, commonly bundled) on the HF model page, then either
set the `HF_TOKEN` env var, pass `--hf-token` to `scripts/run_variant.py`, or pass
`hf_token=` directly to `PipelineConfig`.

## Dataset: Indic DiarBench

Confirmed live schema (`sarvamai/indic-diarbench` on HuggingFace, validated this session):
- Published as **22 separate HF configs, one per language** (not a single split) --
  `load_dataset("sarvamai/indic-diarbench", "Hindi", split="test")`. All 22 language
  names are listed in `data/prep/manifest.py::ALL_LANGUAGES`.
- Each row: `sample_id`, `recording_id` (`<language>_<condition>_<nnn>`), `language`,
  `audio` (16kHz mono WAV), `dataset_type` (`"Near field"|"Far field"|"In the wild"`),
  `duration_seconds`, `annotated_transcript` (list of `{speaker_id, transcript,
  start_time, end_time}`), `num_speakers`, `num_segments`.
- `data/prep/manifest.py::load_indic_diarbench()` streams this (no full 108h download
  needed to iterate a subset) and converts each row into our `ManifestEntry` /
  `SpeechSegment` / `WordToken` schema.

**Audio decoding bug found + fixed this session:** `datasets`' default `Audio` feature
decodes via `torchcodec`, which requires a matching FFmpeg install (versions 4-7). This
machine has neither FFmpeg nor a compatible torchcodec/torch pairing, so the naive path
raises `RuntimeError: Could not load libtorchcodec`. Fixed by calling
`.cast_column("audio", Audio(decode=False))` and decoding the raw WAV bytes directly via
`soundfile` (already a dependency, no FFmpeg needed) in `_decode_audio()`. If you later
have a working FFmpeg+torchcodec setup this bypass is harmless to keep -- soundfile
decoding is just as correct, only lacks torchcodec's video/streaming-container support
which this WAV-only dataset doesn't need.

Note: `annotated_transcript.transcript` is a **segment-level** string, not pre-tokenized
into words. `_row_to_entry` currently treats each segment's whole transcript as a single
`WordToken` spanning the segment -- this makes DER and cpWER-at-the-segment-level
meaningful immediately, but a real word-level WDER against this dataset will need the
segment text tokenized into individual words with estimated (or forced-aligned)
per-word timestamps first. That tokenization step is the next thing to add before
running full evaluation against real data.

## Stress testing against the full dataset

`scripts/stress_test.py` runs the **PROPOSED** variant (pretrained backend -- the actual
proposed system, not a baseline) across a broad, round-robin sample of real recordings
spanning every language and acoustic condition in the dataset, catching and logging every
per-recording failure (instead of crashing the whole run) so it can run unattended for
hours. Nothing about scope is hardcoded -- languages/conditions/thresholds are all CLI
flags, defaulting to the full `ALL_LANGUAGES` list and all three conditions; the
per-recording ASR language hint is resolved dynamically via `data/prep/language_codes.py`,
not assumed.

**Recommended: use the supervisor wrapper, not `stress_test.py` directly**, for anything
longer than a few minutes -- see bug #10 below for why (this machine can crash the whole
Python process at the OS level under memory pressure, which no amount of `try/except`
inside `stress_test.py` can catch; the supervisor is what makes a multi-hour unattended
run actually survive that).

```bash
# recommended: 3h run that auto-restarts (with --resume) through crashes of any kind:
HF_TOKEN=... ./scripts/run_stress_test_supervised.sh 3.0 outputs/stress_test

# narrower/faster run for a quick check, direct (no supervisor needed for a short run):
python scripts/stress_test.py --languages Hindi Tamil --time-budget-hours 0.5

# resume a prior run manually, skipping recordings already logged as processed:
python scripts/stress_test.py --resume

# see how it's doing (safe to run against an in-progress run):
python scripts/summarize_stress_test.py
```

Output goes to `outputs/stress_test/`: `results.jsonl` (one line per success, with full
metrics), `failures.jsonl` (one line per Python-catchable failure, with the full
traceback), and `summary.json` (running aggregate, rewritten every 5 recordings, with
`"finished": true` written once a `stress_test.py` invocation ends normally -- the signal
the supervisor watches for to know whether to relaunch).

## Next steps (suggested order)

1. Add per-word tokenization/alignment for `annotated_transcript.transcript` (see Dataset
   note above) so WDER is meaningful on real data, not just DER/cpWER.
2. Build the section 14-17 analysis scripts (language/condition/speaker-count breakdown,
   ablation sweep) as thin wrappers over `scripts/run_variant.py`'s per-recording result
   list + `eval/significance.py`.
3. Move real-dataset runs to a GPU environment (Colab/cloud) once validated on CPU at
   small scale -- SepFormer + Whisper-medium/large over 108h on CPU is not practical (this
   session's CPU run had RTF 1.3-5.6x on a single 75s clip with the smallest Whisper size).
4. Bump `--asr-model-size` up from `tiny` once on faster hardware, and try a few different
   OSD thresholds/routing modes (section 17.4-17.5 ablations) now that both are confirmed
   working -- `tau` and `RoutingMode` are already exposed on `PipelineConfig`.
