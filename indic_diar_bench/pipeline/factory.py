"""Builds concrete stage implementations from a PipelineConfig.Backend selection."""

from __future__ import annotations

from pipeline.asr import FasterWhisperASR, RegexEnergyASR
from pipeline.clustering import AHCClusterer
from pipeline.config import Backend, PipelineConfig
from pipeline.embeddings import ECAPAEmbedder, MFCCStatsEmbedder
from pipeline.interfaces import ASRModel, Clusterer, SeparationModel, SpeakerEmbedder
from pipeline.osd import EnergyRatioOSD, PyannoteOSD
from pipeline.separation import NullSeparator, SepFormerSeparator
from pipeline.vad import EnergyVAD, PyannoteVAD


def build_vad(cfg: PipelineConfig):
    if cfg.backend == Backend.PRETRAINED:
        return PyannoteVAD(hf_token=cfg.hf_token, device=cfg.device)
    return EnergyVAD()


def build_osd(cfg: PipelineConfig):
    if cfg.backend == Backend.PRETRAINED:
        return PyannoteOSD(hf_token=cfg.hf_token, device=cfg.device)
    return EnergyRatioOSD()


def build_embedder(cfg: PipelineConfig) -> SpeakerEmbedder:
    if cfg.backend == Backend.PRETRAINED:
        return ECAPAEmbedder(device=cfg.device)
    return MFCCStatsEmbedder()


def build_clusterer(cfg: PipelineConfig) -> Clusterer:
    return AHCClusterer(num_speakers=cfg.num_speakers, distance_threshold=cfg.clustering_distance_threshold)


def build_separator(cfg: PipelineConfig) -> SeparationModel:
    if cfg.backend == Backend.PRETRAINED:
        return SepFormerSeparator(device=cfg.device)
    return NullSeparator()


def build_asr(cfg: PipelineConfig) -> ASRModel:
    if cfg.backend == Backend.PRETRAINED:
        compute_type = "float16" if cfg.device == "cuda" else "int8"
        return FasterWhisperASR(model_size=cfg.asr_model_size, compute_type=compute_type, device=cfg.device)
    return RegexEnergyASR()
