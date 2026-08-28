import numpy as np

from pipeline.attribution import SpeakerPrototypeStore, cosine_similarity


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_first_embedding_creates_new_speaker():
    store = SpeakerPrototypeStore()
    spk, sim = store.assign(np.array([1.0, 0.0]))
    assert spk.startswith("SPK_NEW")
    assert sim == 1.0


def test_similar_embedding_reuses_speaker():
    store = SpeakerPrototypeStore(unknown_threshold=0.5)
    spk1, _ = store.assign(np.array([1.0, 0.0]))
    spk2, sim = store.assign(np.array([0.99, 0.01]))
    assert spk1 == spk2
    assert sim > 0.9


def test_dissimilar_embedding_creates_new_speaker():
    store = SpeakerPrototypeStore(unknown_threshold=0.5)
    spk1, _ = store.assign(np.array([1.0, 0.0]))
    spk2, sim = store.assign(np.array([-1.0, 0.0]))
    assert spk1 != spk2
    assert sim < 0.5


def test_seed_and_assign():
    store = SpeakerPrototypeStore(unknown_threshold=0.5)
    store.seed("SPKA", np.array([1.0, 0.0]))
    spk, _sim = store.assign(np.array([0.98, 0.02]))
    assert spk == "SPKA"
