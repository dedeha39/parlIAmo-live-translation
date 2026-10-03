"""Configuration loading, merging and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from parliamo.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    _deep_merge,
    load_config,
    read_yaml,
)


def test_default_config_file_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists(), "config/default.yaml is missing from the repo"


def test_default_config_loads_and_is_typed() -> None:
    cfg = load_config(local_path=Path("does-not-exist.yaml"))
    assert isinstance(cfg, Config)
    assert cfg.project.name == "parliamo"
    assert cfg.asr.language == "tr", "stage language must default to Turkish"
    assert cfg.tts.language == "it", "primary target must default to Italian"


def test_audio_block_frames_matches_sample_rate() -> None:
    cfg = load_config(local_path=Path("does-not-exist.yaml"))
    expected = int(cfg.audio.sample_rate * cfg.audio.block_ms / 1000)
    assert cfg.audio.block_frames == expected
    assert cfg.audio.block_frames == 512, "32 ms @ 16 kHz should be 512 frames"


def test_half_duplex_defaults_on() -> None:
    """Feedback protection must be opt-out, never opt-in.

    With speakers as the output device, a disabled gate turns the room into a
    loop within seconds. Defaulting this to False would be a stage-killing bug.
    """
    cfg = load_config(local_path=Path("does-not-exist.yaml"))
    assert cfg.pipeline.half_duplex is True


def test_friulian_disabled_until_finetuned() -> None:
    cfg = load_config(local_path=Path("does-not-exist.yaml"))
    assert cfg.mt.friulian.enabled is False
    assert cfg.mt.friulian.tgt_token == "fur_Latn"


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("nonsense_section:\n  foo: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(default_path=bad, local_path=Path("nope.yaml"))


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            asr:
              langauge: tr   # deliberate typo
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="asr"):
        load_config(default_path=bad, local_path=Path("nope.yaml"))


def test_local_overrides_default(tmp_path: Path) -> None:
    base = tmp_path / "default.yaml"
    base.write_text("asr:\n  language: tr\n  beam_size: 1\n", encoding="utf-8")
    local = tmp_path / "local.yaml"
    local.write_text("asr:\n  beam_size: 5\n", encoding="utf-8")

    cfg = load_config(default_path=base, local_path=local)
    assert cfg.asr.language == "tr", "unspecified keys must survive the merge"
    assert cfg.asr.beam_size == 5


def test_explicit_overrides_win(tmp_path: Path) -> None:
    base = tmp_path / "default.yaml"
    base.write_text("asr:\n  language: tr\n", encoding="utf-8")
    cfg = load_config(
        default_path=base,
        local_path=Path("nope.yaml"),
        overrides={"asr": {"language": "en"}},
    )
    assert cfg.asr.language == "en"


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1, "y": 2}}
    overlay = {"a": {"y": 3}}
    merged = _deep_merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 3}}
    assert base == {"a": {"x": 1, "y": 2}}, "merge must be pure"


def test_read_yaml_missing_file_is_empty() -> None:
    assert read_yaml(Path("definitely-not-here.yaml")) == {}


def test_read_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    f = tmp_path / "list.yaml"
    f.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        read_yaml(f)
