"""Word Diarization Error Rate (WDER), section 12.2.

WDER = (# words assigned to the wrong speaker) / (# reference words),
computed only over words that had a correct lexical match against the
reference (i.e. this isolates speaker-attribution error from ASR lexical
error -- see cpWER in wder.py's sibling module for the combined metric).

We align hypothesis words to reference words by timestamp overlap first
(each hyp word is matched to the reference word with the largest time
overlap), then check whether the matched pair's speaker labels agree
under the DER-style optimal speaker mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.types import WordToken


@dataclass
class WDERResult:
    wder: float
    total_words: int
    misattributed_words: int


def _best_speaker_mapping_by_words(
    reference: list[WordToken], hypothesis: list[WordToken]
) -> dict[str, str]:
    ref_speakers = sorted({w.speaker for w in reference if w.speaker})
    hyp_speakers = sorted({w.speaker for w in hypothesis if w.speaker})
    if not ref_speakers or not hyp_speakers:
        return {}

    counts: dict[tuple[str, str], int] = {}
    for hw in hypothesis:
        if hw.speaker is None:
            continue
        rw = _closest_by_time(hw, reference)
        if rw is None or rw.speaker is None:
            continue
        key = (hw.speaker, rw.speaker)
        counts[key] = counts.get(key, 0) + 1

    # greedy max-count assignment (small vocab -- typically <= ~9 speakers per section 3.4)
    mapping: dict[str, str] = {}
    used_ref: set[str] = set()
    for (hspk, rspk), _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        if hspk in mapping or rspk in used_ref:
            continue
        mapping[hspk] = rspk
        used_ref.add(rspk)
    return mapping


def _closest_by_time(word: WordToken, candidates: list[WordToken]) -> WordToken | None:
    best, best_overlap = None, 0.0
    for c in candidates:
        overlap = max(0.0, min(word.end, c.end) - max(word.start, c.start))
        if overlap > best_overlap:
            best, best_overlap = c, overlap
    if best is not None:
        return best
    # fall back to nearest midpoint if no direct time overlap
    mid = (word.start + word.end) / 2
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs((c.start + c.end) / 2 - mid))


def compute_wder(reference: list[WordToken], hypothesis: list[WordToken]) -> WDERResult:
    if not reference:
        return WDERResult(0.0, 0, 0)

    mapping = _best_speaker_mapping_by_words(reference, hypothesis)

    misattributed = 0
    scored = 0
    for hw in hypothesis:
        rw = _closest_by_time(hw, reference)
        if rw is None or rw.speaker is None or hw.speaker is None:
            continue
        scored += 1
        mapped_speaker = mapping.get(hw.speaker, hw.speaker)
        if mapped_speaker != rw.speaker:
            misattributed += 1

    total = len(reference)
    wder = misattributed / total if total > 0 else 0.0
    return WDERResult(wder=wder, total_words=total, misattributed_words=misattributed)
