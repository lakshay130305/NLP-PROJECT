"""The overlap-aware joint diarization-ASR framework, section 5 architecture:

Input Audio -> VAD -> Overlapped Speech Detection -> Selective Router
  non-overlap branch: standard diarization
  overlap branch: speech separation -> multi-stream speaker attribution
Both branches -> Multilingual ASR -> Transcript fusion -> Speaker-attributed output

Which of {OSD, separation, attribution, routing} are active is controlled by
PipelineConfig (pipeline/config.py, pipeline/variants.py), so this one class
implements every variant B1..B5/Proposed from section 11.3.
"""

from __future__ import annotations

import time

import numpy as np

from eval.efficiency import EfficiencyStats
from pipeline.attribution import SpeakerPrototypeStore, cosine_similarity
from pipeline.config import PipelineConfig
from pipeline.factory import (
    build_asr,
    build_clusterer,
    build_embedder,
    build_osd,
    build_separator,
    build_vad,
)
from pipeline.fuse import BranchOutput, fuse_transcript
from pipeline.osd import smooth_overlap_decisions, threshold_overlap
from pipeline.router import (
    RoutingDecision,
    RoutingMode,
    SelectiveRouter,
    routed_fraction,
)
from pipeline.separation import iterative_multi_speaker_separation
from schemas.types import OverlapRegion, SpeakerAttributedTranscript, SpeechSegment


class OverlapAwarePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vad = build_vad(config)
        self.osd = build_osd(config) if config.use_osd else None
        self.embedder = build_embedder(config)
        self.clusterer = build_clusterer(config)
        self.separator = build_separator(config) if config.use_separation else None
        self.asr = build_asr(config)

    def run(
        self,
        audio: np.ndarray,
        sample_rate: int,
        recording_id: str = "recording",
        language: str | None = None,
    ):
        """`language`, if given, overrides `config.asr_language` for this call only --
        lets one long-lived pipeline instance be reused across a multilingual batch
        (each recording passes its own known language) without rebuilding the ASR
        backend per language. Falls back to the config-level default (usually None,
        i.e. auto-detect) when not given."""
        stats = EfficiencyStats(audio_duration=len(audio) / sample_rate)
        asr_language = language if language is not None else self.config.asr_language

        vad_segments = self._timed(stats, "vad", self.vad.detect, audio, sample_rate)
        if not vad_segments:
            return SpeakerAttributedTranscript(recording_id=recording_id), stats, []

        predicted_overlap_regions: list[OverlapRegion] = []
        routing: list[RoutingDecision] | None = None

        if self.config.use_osd:
            times, probs = self._timed(
                stats, "osd", self.osd.frame_overlap_probs, audio, sample_rate
            )
            predicted_overlap_regions = smooth_overlap_decisions(
                times, threshold_overlap(times, probs, self.config.osd_tau), self.config.osd_min_duration
            )
            if self.config.use_separation or self.config.use_attribution:
                router = SelectiveRouter(
                    tau=self.config.osd_tau,
                    mode=self.config.routing_mode or RoutingMode.HARD,
                    min_overlap_duration=self.config.osd_min_duration,
                )
                routing = router.route_segments(vad_segments, times, probs)
                stats.stage_times["routed_fraction"] = routed_fraction(routing)

        overlap_segments, single_segments = self._split_segments(vad_segments, routing)

        # phase 1: non-overlap branch -- standard diarization (section 4)
        single_embeddings = np.array(
            [self._timed(stats, "embed", self.embedder.embed, self._slice(audio, sample_rate, s), sample_rate)
             for s in single_segments]
        ) if single_segments else np.empty((0, 1))

        labels = self._timed(stats, "cluster", self.clusterer.cluster, single_embeddings) if len(single_segments) else []

        prototypes = SpeakerPrototypeStore(unknown_threshold=self.config.unknown_speaker_threshold)
        for spk, emb in zip(labels, single_embeddings):
            if spk in prototypes._prototypes:
                prototypes._prototypes[spk] = (prototypes._prototypes[spk] + emb) / 2
            else:
                prototypes.seed(spk, emb)

        branch_outputs: list[BranchOutput] = []
        for seg, speaker in zip(single_segments, labels):
            words = self._timed(stats, "asr", self.asr.transcribe, self._slice(audio, sample_rate, seg),
                                 sample_rate, asr_language)
            branch_outputs.append(BranchOutput(chunk_start=seg.start, words=words, speaker=speaker))

        # phase 2: overlap branch -- separation + multi-speaker attribution (sections 7-8)
        for seg in overlap_segments:
            seg_audio = self._slice(audio, sample_rate, seg)

            if self.config.use_separation:
                if self.config.max_overlap_speakers <= 2:
                    streams = self._timed(stats, "separate", self.separator.separate, seg_audio, sample_rate, 2)
                else:
                    streams = self._timed(
                        stats, "separate", iterative_multi_speaker_separation,
                        seg_audio, sample_rate, self.separator, self.config.max_overlap_speakers,
                    )
                stats.record_separator_call(seg.duration)
            else:
                streams = [seg_audio]  # B3: no separation, attribute the mixed segment directly (section 8.4 note)

            stream_embeddings = [self.embedder.embed(s, sample_rate) for s in streams]

            for stream_audio, emb in zip(streams, stream_embeddings):
                if self.config.use_attribution:
                    speaker, _sim = prototypes.assign(emb)
                else:
                    speaker = self._nearest_fixed_centroid(emb, prototypes)
                words = self._timed(stats, "asr", self.asr.transcribe, stream_audio, sample_rate, asr_language)
                branch_outputs.append(BranchOutput(chunk_start=seg.start, words=words, speaker=speaker))

        transcript = fuse_transcript(recording_id, branch_outputs)
        return transcript, stats, predicted_overlap_regions

    @staticmethod
    def _split_segments(
        vad_segments: list[SpeechSegment], routing: list[RoutingDecision] | None
    ) -> tuple[list[SpeechSegment], list[SpeechSegment]]:
        if routing is None:
            return [], vad_segments
        overlap = [d.segment for d in routing if d.route == "overlap"]
        single = [d.segment for d in routing if d.route == "single"]
        return overlap, single

    @staticmethod
    def _slice(audio: np.ndarray, sample_rate: int, segment: SpeechSegment) -> np.ndarray:
        start_idx = max(0, int(segment.start * sample_rate))
        end_idx = min(len(audio), int(segment.end * sample_rate))
        return audio[start_idx:end_idx]

    @staticmethod
    def _nearest_fixed_centroid(embedding: np.ndarray, prototypes: SpeakerPrototypeStore) -> str:
        """B4 assignment rule: nearest existing cluster centroid, no new-speaker creation,
        no online centroid update -- isolates separation's contribution from attribution's."""
        if not prototypes._prototypes:
            return "SPK00"
        sims = {spk: cosine_similarity(embedding, proto) for spk, proto in prototypes._prototypes.items()}
        return max(sims, key=sims.get)

    @staticmethod
    def _timed(stats: EfficiencyStats, stage: str, fn, *args):
        t0 = time.perf_counter()
        result = fn(*args)
        stats.add_stage_time(stage, time.perf_counter() - t0)
        return result
