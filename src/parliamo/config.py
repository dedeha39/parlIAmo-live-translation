"""Typed configuration loading.

Layering, lowest precedence first:

    1. ``config/default.yaml``   - checked in, documents every option
    2. ``config/local.yaml``     - git-ignored machine-specific overrides
    3. explicit overrides passed to :func:`load_config` (e.g. from CLI flags)

Unknown keys are rejected. A typo in a config file should fail at startup with
a clear message, not silently change behaviour twenty minutes into a rehearsal.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT, resolve

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
LOCAL_CONFIG_PATH = REPO_ROOT / "config" / "local.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or contains unknown keys."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PathsConfig:
    models: str = "models"
    runs: str = "runs"
    data: str = "data"

    @property
    def models_dir(self) -> Path:
        return resolve(self.models)

    @property
    def runs_dir(self) -> Path:
        return resolve(self.runs)

    @property
    def data_dir(self) -> Path:
        return resolve(self.data)


@dataclass(slots=True)
class ProjectConfig:
    name: str = "parliamo"
    paths: PathsConfig = field(default_factory=PathsConfig)


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 32
    input_device: int | str | None = None
    output_device: int | str | None = None
    output_sample_rate: int = 24000
    output_gain: float = 1.0

    @property
    def block_frames(self) -> int:
        """Capture block size in frames."""
        return int(self.sample_rate * self.block_ms / 1000)


@dataclass(slots=True)
class VadConfig:
    backend: str = "silero"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 700
    speech_pad_ms: int = 120
    max_segment_ms: int = 15000


@dataclass(slots=True)
class AsrConfig:
    backend: str = "faster_whisper"
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = "tr"
    beam_size: int = 5
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    temperature: float = 0.0
    hotwords: str | None = None


@dataclass(slots=True)
class FriulianConfig:
    enabled: bool = False
    backend: str = "ctranslate2_nllb"
    model_path: str | None = None
    device: str = "cpu"
    src_token: str = "ita_Latn"
    tgt_token: str = "fur_Latn"


@dataclass(slots=True)
class MtConfig:
    backend: str = "ctranslate2_nllb"
    model_path: str | None = "models/ct2/nllb-200-distilled-600M"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 4
    max_decoding_length: int = 256
    split_sentences: bool = True
    source_lang: str = "tr"
    targets: list[str] = field(default_factory=lambda: ["it"])
    friulian: FriulianConfig = field(default_factory=FriulianConfig)


@dataclass(slots=True)
class ConversionConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    diffusion_steps: int = 4
    timeout_s: float = 10.0
    reference_voice: str | None = None
    consent: str = ""
    #: OmniVoice's speaking rate, sent with every sentence: below 1 is slower.
    #: None lets it copy the reference's pace - the presenter reads at 6.0
    #: syllables/s and the clones came out at 6.2-6.4, against Kokoro's 5.3;
    #: "a bit fast" for this audience. RVC and Seed-VC ignore it.
    speed: float | None = None
    #: The presenter's median pitch. When set, each sentence is sent to the
    #: voice service with the semitone shift from the synthesiser's voice to
    #: this - measured once at start - instead of one --pitch for every
    #: language. RVC keeps the pitch it is given; Seed-VC ignores the field.
    presenter_f0_hz: float | None = None


@dataclass(slots=True)
class TtsConfig:
    backend: str = "kokoro"
    device: str = "cuda"
    language: str = "it"
    voice: str = "if_sara"
    #: A voice per target language; ``voice`` is used where none is named.
    voices: dict[str, str] = field(default_factory=dict)
    speed: float = 1.0
    friulian_speaks_as: str = "it"
    conversion: ConversionConfig = field(default_factory=ConversionConfig)


@dataclass(slots=True)
class PipelineConfig:
    half_duplex: bool = True
    half_duplex_tail_ms: int = 250
    output_latency_ms: int = 0
    subtitles: bool = True
    emit_partial_transcripts: bool = True
    partial_interval_ms: int = 800
    stream_sentences: bool = False
    max_queue_depth: int = 4


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    console: bool = True
    jsonl: bool = True
    latency_trace: bool = True


@dataclass(slots=True)
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    mt: MtConfig = field(default_factory=MtConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into *base*, returning a new dict."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, data: dict[str, Any], path: str = "") -> Any:
    """Instantiate dataclass *cls* from *data*, rejecting unknown keys."""
    if not is_dataclass(cls):
        return data

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        where = path or "<root>"
        raise ConfigError(
            f"unknown configuration key(s) at {where}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        child_path = f"{path}.{name}" if path else name
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _build(f.type, value, child_path)
        elif isinstance(value, dict) and _nested_type(cls, name) is not None:
            kwargs[name] = _build(_nested_type(cls, name), value, child_path)
        else:
            kwargs[name] = value
    return cls(**kwargs)


# ``from __future__ import annotations`` turns field types into strings, so we
# resolve the handful of nested dataclasses explicitly instead of calling
# ``typing.get_type_hints`` (which would need every annotation to be importable).
_NESTED: dict[tuple[type, str], type] = {
    (ProjectConfig, "paths"): PathsConfig,
    (MtConfig, "friulian"): FriulianConfig,
    (TtsConfig, "conversion"): ConversionConfig,
    (Config, "project"): ProjectConfig,
    (Config, "audio"): AudioConfig,
    (Config, "vad"): VadConfig,
    (Config, "asr"): AsrConfig,
    (Config, "mt"): MtConfig,
    (Config, "tts"): TtsConfig,
    (Config, "pipeline"): PipelineConfig,
    (Config, "logging"): LoggingConfig,
}


def _nested_type(cls: type, field_name: str) -> type | None:
    return _NESTED.get((cls, field_name))


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict. A missing file yields an empty dict."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def load_config(
    default_path: Path | None = None,
    local_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load the layered configuration and validate it into a :class:`Config`."""
    merged = read_yaml(default_path or DEFAULT_CONFIG_PATH)
    merged = _deep_merge(merged, read_yaml(local_path or LOCAL_CONFIG_PATH))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return _build(Config, merged)
