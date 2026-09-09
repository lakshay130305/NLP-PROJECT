"""Speaker clustering, section 4.4. Agglomerative hierarchical clustering (AHC) over
cosine distance between embeddings, with either a fixed speaker count or a distance
threshold (section 3.4 speaker-count assumptions vary 2 / 3-4 / 5+)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


class AHCClusterer:
    def __init__(self, num_speakers: int | None = None, distance_threshold: float = 0.7):
        """If `num_speakers` is given, cluster into exactly that many groups.
        Otherwise use `distance_threshold` on cosine distance to auto-select the speaker count."""
        self.num_speakers = num_speakers
        self.distance_threshold = distance_threshold

    def cluster(self, embeddings: np.ndarray) -> list[str]:
        if len(embeddings) == 0:
            return []
        if len(embeddings) == 1:
            return ["SPK00"]

        # Safety net for an overnight batch: one non-finite value anywhere in the matrix
        # makes AgglomerativeClustering raise and takes the whole run down with it. A
        # zeroed embedding just clusters badly for that one segment. Embedders are expected
        # to return finite vectors (see MFCCStatsEmbedder._log_band_means); this is the
        # backstop, not the fix.
        embeddings = np.nan_to_num(np.asarray(embeddings, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        if self.num_speakers is not None:
            model = AgglomerativeClustering(
                n_clusters=self.num_speakers, metric="cosine", linkage="average"
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="cosine",
                linkage="average",
            )
        labels = model.fit_predict(normed)
        return [f"SPK{label:02d}" for label in labels]
