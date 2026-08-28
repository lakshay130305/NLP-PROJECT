from eval.cpwer import compute_cpwer
from eval.wder import compute_wder
from eval.wer import compute_wer
from schemas.types import WordToken


def test_wer_identical():
    result = compute_wer("hello world", "hello world")
    assert result.wer == 0.0


def test_wer_substitution():
    result = compute_wer("hello world", "hello there")
    assert result.substitutions == 1
    assert result.wer == 0.5


def test_wer_deletion_insertion():
    result = compute_wer("a b c", "a c")
    assert result.deletions == 1
    result2 = compute_wer("a b", "a b c")
    assert result2.insertions == 1


def _wt(text, start, end, speaker):
    return WordToken(text=text, start=start, end=end, speaker=speaker)


def test_wder_perfect_attribution():
    ref = [_wt("hi", 0, 0.5, "A"), _wt("there", 1, 1.5, "B")]
    hyp = [_wt("hi", 0, 0.5, "X"), _wt("there", 1, 1.5, "Y")]  # labels differ but mapping should resolve
    result = compute_wder(ref, hyp)
    assert result.wder == 0.0


def test_wder_misattribution():
    ref = [_wt("hi", 0, 0.5, "A"), _wt("there", 1, 1.5, "B"), _wt("friend", 2, 2.5, "A")]
    hyp = [_wt("hi", 0, 0.5, "A"), _wt("there", 1, 1.5, "A"), _wt("friend", 2, 2.5, "A")]
    result = compute_wder(ref, hyp)
    assert result.misattributed_words >= 1


def test_cpwer_perfect():
    ref = [_wt("hi", 0, 0.5, "A"), _wt("there", 1, 1.5, "B")]
    hyp = [_wt("hi", 0, 0.5, "X"), _wt("there", 1, 1.5, "Y")]
    result = compute_cpwer(ref, hyp)
    assert result.cpwer == 0.0


def test_cpwer_with_errors():
    ref = [_wt("hi", 0, 0.5, "A"), _wt("there", 1, 1.5, "B")]
    hyp = [_wt("hi", 0, 0.5, "A")]  # missing the second speaker entirely
    result = compute_cpwer(ref, hyp)
    assert result.cpwer > 0.0
