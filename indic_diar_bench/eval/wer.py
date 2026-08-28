"""Plain word error rate (WER), section 12.4. Pure-python Levenshtein over word sequences
so this module has no dependency on jiwer/torch and can run in any environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WERResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int


def _edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Standard DP word-level edit distance, returning (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, sub, del, ins) minimal-cost path to align ref[:i], hyp[:j]
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub_cost = dp[i - 1][j - 1][0] + 1
            del_cost = dp[i - 1][j][0] + 1
            ins_cost = dp[i][j - 1][0] + 1
            best = min(sub_cost, del_cost, ins_cost)
            if best == sub_cost:
                _c, s, d, ins = dp[i - 1][j - 1]
                dp[i][j] = (best, s + 1, d, ins)
            elif best == del_cost:
                _c, s, d, ins = dp[i - 1][j]
                dp[i][j] = (best, s, d + 1, ins)
            else:
                _c, s, d, ins = dp[i][j - 1]
                dp[i][j] = (best, s, d, ins + 1)

    _, sub, dele, ins = dp[n][m]
    return sub, dele, ins


def compute_wer(reference: str, hypothesis: str) -> WERResult:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    sub, dele, ins = _edit_ops(ref_words, hyp_words)
    n = len(ref_words)
    wer = (sub + dele + ins) / n if n > 0 else (0.0 if not hyp_words else 1.0)
    return WERResult(wer=wer, substitutions=sub, deletions=dele, insertions=ins, ref_words=n)
