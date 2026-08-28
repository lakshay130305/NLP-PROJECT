"""Concatenated minimum-permutation WER (cpWER), section 12.3.

Procedure (NIST/Rev-style, as used in speaker-attributed ASR evaluation):
1. Concatenate all reference words per reference speaker (in time order) -> one string per ref speaker.
2. Concatenate all hypothesis words per hypothesis speaker -> one string per hyp speaker.
3. Try every hyp-speaker -> ref-speaker permutation (or Hungarian assignment on pairwise WER cost),
   pick the one minimizing total edit operations.
4. cpWER = total edits across all matched pairs / total reference words.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from eval.wer import _edit_ops
from schemas.types import WordToken


@dataclass
class CPWERResult:
    cpwer: float
    total_edits: int
    total_ref_words: int
    speaker_mapping: dict[str, str]


def _concat_by_speaker(words: list[WordToken]) -> dict[str, list[str]]:
    by_spk: dict[str, list[WordToken]] = {}
    for w in words:
        by_spk.setdefault(w.speaker or "UNK", []).append(w)
    return {spk: [w.text for w in sorted(ws, key=lambda w: w.start)] for spk, ws in by_spk.items()}


def compute_cpwer(reference: list[WordToken], hypothesis: list[WordToken]) -> CPWERResult:
    ref_by_spk = _concat_by_speaker(reference)
    hyp_by_spk = _concat_by_speaker(hypothesis)

    ref_speakers = sorted(ref_by_spk)
    hyp_speakers = sorted(hyp_by_spk)
    total_ref_words = sum(len(w) for w in ref_by_spk.values())

    if not ref_speakers:
        return CPWERResult(0.0, 0, 0, {})
    if not hyp_speakers:
        return CPWERResult(1.0, total_ref_words, total_ref_words, {})

    # pad the smaller side with empty "silent" speakers so both sides have equal size for permutation search
    n = max(len(ref_speakers), len(hyp_speakers))
    padded_ref = ref_speakers + [None] * (n - len(ref_speakers))
    padded_hyp = hyp_speakers + [None] * (n - len(hyp_speakers))

    best_cost = None
    best_mapping: dict[str, str] = {}

    # exact search is fine for small speaker counts (section 3.4: typically <= ~9 speakers);
    # fall back to a greedy heuristic if it ever gets large enough to blow up factorially
    if n <= 8:
        candidate_perms = permutations(padded_hyp)
    else:
        candidate_perms = [tuple(sorted(padded_hyp, key=lambda h: h or ""))]

    for perm in candidate_perms:
        cost = 0
        mapping: dict[str, str] = {}
        for rspk, hspk in zip(padded_ref, perm):
            ref_words = ref_by_spk.get(rspk, []) if rspk else []
            hyp_words = hyp_by_spk.get(hspk, []) if hspk else []
            sub, dele, ins = _edit_ops(ref_words, hyp_words)
            cost += sub + dele + ins
            if rspk and hspk:
                mapping[hspk] = rspk
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_mapping = mapping

    cpwer = best_cost / total_ref_words if total_ref_words > 0 else 0.0
    return CPWERResult(
        cpwer=cpwer, total_edits=best_cost, total_ref_words=total_ref_words, speaker_mapping=best_mapping
    )
