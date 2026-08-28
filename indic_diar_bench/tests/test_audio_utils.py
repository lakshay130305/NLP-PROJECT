import numpy as np
import soundfile as sf

from pipeline.audio_utils import load_audio_file, pad_to_min_length


def test_short_audio_gets_padded():
    audio = np.ones(100, dtype=np.float32)  # far shorter than 0.5s at 16kHz
    padded = pad_to_min_length(audio, sample_rate=16000, min_seconds=0.5)
    assert len(padded) == 8000
    assert np.all(padded[:100] == 1.0)
    assert np.all(padded[100:] == 0.0)


def test_long_audio_unchanged():
    audio = np.ones(20000, dtype=np.float32)
    padded = pad_to_min_length(audio, sample_rate=16000, min_seconds=0.5)
    assert len(padded) == 20000
    assert np.array_equal(padded, audio)


def test_exact_min_length_unchanged():
    audio = np.ones(8000, dtype=np.float32)
    padded = pad_to_min_length(audio, sample_rate=16000, min_seconds=0.5)
    assert len(padded) == 8000


def test_load_audio_file_wav(tmp_path):
    t = np.arange(32000) / 16000
    sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), sig, 16000)

    audio, sr = load_audio_file(wav_path)
    assert sr == 16000
    assert audio.dtype == np.float32
    assert abs(len(audio) - 32000) <= 2


def test_load_audio_file_resamples_non_16k_source(tmp_path):
    t = np.arange(8000) / 8000
    sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "test_8k.wav"
    sf.write(str(wav_path), sig, 8000)

    audio, sr = load_audio_file(wav_path)
    assert sr == 16000
    assert abs(len(audio) - 16000) <= 4  # ~1s of audio at the target rate


def test_load_audio_file_mp3_via_pyav_fallback(tmp_path):
    av = __import__("av")

    t = np.arange(16000) / 16000
    sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mp3_path = tmp_path / "test.mp3"

    container = av.open(str(mp3_path), mode="w")
    stream = container.add_stream("mp3", rate=16000)
    frame = av.AudioFrame.from_ndarray(sig.reshape(1, -1), format="fltp", layout="mono")
    frame.sample_rate = 16000
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    audio, sr = load_audio_file(mp3_path)
    assert sr == 16000
    assert len(audio) > 0
    assert audio.dtype == np.float32
