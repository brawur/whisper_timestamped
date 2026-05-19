from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class MetadataResponse(BaseModel):
    service: str
    supports_word_timestamps: bool
    supports_speaker_diarization: bool
    mode: str
    supported_languages: list[str] = Field(default_factory=list)
    supported_parameters: list[str] = Field(default_factory=list)


class WhisperWord(BaseModel):
    text: str
    start: float
    end: float
    confidence: float | None = None


class WhisperSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    temperature: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    confidence: float | None = None
    words: list[WhisperWord] = Field(default_factory=list)


class FileTranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    model: str
    processing_ms: int
    segments: list[WhisperSegment] = Field(default_factory=list)


class LidCandidate(BaseModel):
    language: str
    confidence: float


class LidProbeResponse(BaseModel):
    language: str
    confidence: float
    candidates: list[LidCandidate] = Field(default_factory=list)
    model: str
    sample_offset_seconds: float
    sample_duration_seconds: float
    processing_ms: int


class HostCPUInfo(BaseModel):
    usage: float


class HostMemoryInfo(BaseModel):
    usage: float


class HostGPUInfo(BaseModel):
    available: bool
    name: str | None = None
    gpu_utilization: float | None = None
    memory_utilization: float | None = None
    used_memory: int | None = None
    total_memory: int | None = None


class RuntimeMetricsResponse(BaseModel):
    timestamp: int
    cpu: HostCPUInfo
    memory: HostMemoryInfo
    gpu: HostGPUInfo
