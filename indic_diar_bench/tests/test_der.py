from eval.der import compute_der
from schemas.types import SpeechSegment


def test_perfect_der_is_zero():
    ref = [SpeechSegment(0, 1, "A"), SpeechSegment(1, 2, "B")]
    hyp = [SpeechSegment(0, 1, "X"), SpeechSegment(1, 2, "Y")]  # different labels, but mapping should fix it
    result = compute_der(ref, hyp)
    assert result.der < 1e-9


def test_missed_speech():
    ref = [SpeechSegment(0, 2, "A")]
    hyp = []
    result = compute_der(ref, hyp)
    assert result.der == 1.0
    assert result.missed_speech == 2.0


def test_false_alarm():
    ref = [SpeechSegment(0, 1, "A")]
    hyp = [SpeechSegment(0, 1, "A"), SpeechSegment(1, 2, "B")]
    result = compute_der(ref, hyp)
    assert result.false_alarm == 1.0


def test_speaker_confusion():
    ref = [SpeechSegment(0, 1, "A"), SpeechSegment(1, 2, "B")]
    # hypothesis swaps who's active where, so after optimal mapping there should be confusion
    hyp = [SpeechSegment(0, 1, "X"), SpeechSegment(1, 2, "X")]
    result = compute_der(ref, hyp)
    assert result.der > 0
