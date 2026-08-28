"""Speaker representation and multi-speaker attribution, section 8.

Maintains a centroid/prototype per diarized speaker (8.2), assigns separated
streams to the most similar prototype via cosine similarity (8.3), allows
multi-speaker labeling of the same interval (8.4), creates new speaker
identities below a similarity threshold (8.5), and updates centroids over
time for consistency (8.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


@dataclass
class SpeakerPrototypeStore:
    """section 8.2 (prototype construction) + 8.6 (temporal consistency via centroid updating)."""

    unknown_threshold: float = 0.25
    centroid_momentum: float = 0.9  # weight kept on the old centroid when updating
    _prototypes: dict[str, np.ndarray] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _next_unknown_id: int = 0

    def seed(self, speaker: str, embedding: np.ndarray) -> None:
        self._prototypes[speaker] = embedding.copy()
        self._counts[speaker] = 1

    def assign(self, embedding: np.ndarray) -> tuple[str, float]:
        """section 8.3 (similarity-based assignment) + 8.5 (unknown speaker handling)."""
        if not self._prototypes:
            new_id = self._new_unknown_id()
            self.seed(new_id, embedding)
            return new_id, 1.0

        sims = {spk: cosine_similarity(embedding, proto) for spk, proto in self._prototypes.items()}
        best_speaker = max(sims, key=sims.get)
        best_sim = sims[best_speaker]

        if best_sim < self.unknown_threshold:
            new_id = self._new_unknown_id()
            self.seed(new_id, embedding)
            return new_id, best_sim

        self._update_centroid(best_speaker, embedding)
        return best_speaker, best_sim

    def _update_centroid(self, speaker: str, embedding: np.ndarray) -> None:
        old = self._prototypes[speaker]
        m = self.centroid_momentum
        self._prototypes[speaker] = m * old + (1 - m) * embedding
        self._counts[speaker] = self._counts.get(speaker, 0) + 1

    def _new_unknown_id(self) -> str:
        new_id = f"SPK_NEW{self._next_unknown_id:02d}"
        self._next_unknown_id += 1
        return new_id


@dataclass
class StreamAttribution:
    stream_index: int
    speaker: str
    similarity: float


def attribute_separated_streams(
    stream_embeddings: list[np.ndarray], prototypes: SpeakerPrototypeStore
) -> list[StreamAttribution]:
    """section 8.1 + 8.4: one embedding per separated stream -> multi-speaker labeling,
    i.e. the same time interval can end up attributed to Speaker A + Speaker B."""
    results = []
    for i, emb in enumerate(stream_embeddings):
        speaker, sim = prototypes.assign(emb)
        results.append(StreamAttribution(stream_index=i, speaker=speaker, similarity=sim))
    return results
