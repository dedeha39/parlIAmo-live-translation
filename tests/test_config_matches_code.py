"""The config must name things the code can actually build.

This file exists because of one bug. `config/default.yaml` carried
`mt.backend: llama_cpp` for weeks after ADR 0004 replaced the LLM with NLLB,
and nothing caught it: every benchmark constructed its backend directly, so no
test and no script ever read the value. It surfaced only when the live CLI was
written, by crashing at startup.

The class of failure is worth naming: **a decision recorded only in prose is not
a decision the system knows about.** Prose cannot be executed, so the config has
to be checked against the registries the same way code is checked against tests.

Everything here is cheap and hardware-free - it reads the shipped config and
asks the factories whether they recognise what it says.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parliamo.config import load_config

# Always test the checked-in defaults, never the developer's local overrides:
# a machine-specific local.yaml must not be able to make this pass or fail.
NO_LOCAL = Path("does-not-exist-local.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_config(local_path=NO_LOCAL)


# ---------------------------------------------------------------------------
# backend names resolve
# ---------------------------------------------------------------------------


def test_asr_backend_is_registered(cfg) -> None:
    from parliamo.asr import available_backends

    assert cfg.asr.backend in available_backends(), (
        f"config names ASR backend {cfg.asr.backend!r}, which no registry entry "
        f"can build. Available: {available_backends()}"
    )


def test_mt_backend_is_registered(cfg) -> None:
    from parliamo.mt import available_backends

    assert cfg.mt.backend in available_backends(), (
        f"config names MT backend {cfg.mt.backend!r}, which no registry entry "
        f"can build. Available: {available_backends()}. This is the exact bug "
        "that shipped as 'llama_cpp' after ADR 0004 chose NLLB."
    )


def test_friulian_backend_is_registered(cfg) -> None:
    from parliamo.mt import available_backends

    assert cfg.mt.friulian.backend in available_backends()


def test_tts_backend_is_registered(cfg) -> None:
    from parliamo.tts import available_backends

    names = available_backends()
    if not names:
        pytest.skip("no TTS backend importable in this environment")
    assert cfg.tts.backend in names, (
        f"config names TTS backend {cfg.tts.backend!r}; registered: {names}"
    )


def test_vad_backend_is_the_one_that_exists(cfg) -> None:
    # There is no VAD registry - the value is read by name in the pipeline, so
    # a typo would silently mean nothing rather than raising.
    assert cfg.vad.backend == "silero"


def test_every_registered_backend_is_constructible_by_name() -> None:
    """A registered name must map to a class, not to None or a stale import."""
    from parliamo.asr import _REGISTRY as asr_registry
    from parliamo.mt import _REGISTRY as mt_registry

    for registry in (asr_registry, mt_registry):
        for name, cls in registry.items():
            assert isinstance(cls, type), f"{name} is registered as {cls!r}, not a class"
            assert cls.name == name, (
                f"{cls.__name__} is registered under {name!r} but calls itself "
                f"{cls.name!r}; create_backend and describe() would disagree"
            )


# ---------------------------------------------------------------------------
# languages resolve
# ---------------------------------------------------------------------------


def test_translation_languages_have_nllb_tags(cfg) -> None:
    from parliamo.mt.ctranslate2_nllb import CTranslate2NLLBBackend as NLLB

    NLLB.tag(cfg.mt.source_lang)
    for target in cfg.mt.targets:
        NLLB.tag(target)
    NLLB.tag(cfg.mt.friulian.src_token)
    NLLB.tag(cfg.mt.friulian.tgt_token)


def test_output_language_can_be_spoken(cfg) -> None:
    """The synthesiser must have a voice for the language we translate into."""
    if cfg.tts.backend != "kokoro":
        pytest.skip(f"language mapping test is Kokoro-specific, backend is {cfg.tts.backend}")
    from parliamo.tts.kokoro_backend import KokoroBackend

    code, note = KokoroBackend().resolve_language(cfg.tts.language)
    assert code, f"no Kokoro voice for tts.language={cfg.tts.language!r}"
    assert not note, f"tts.language={cfg.tts.language!r} is only spoken by substitution: {note}"


def test_friulian_is_spoken_by_substitution_and_says_so(cfg) -> None:
    """Friulian has no voice anywhere. The fallback must be explicit, not silent."""
    if cfg.tts.backend != "kokoro":
        pytest.skip("Kokoro-specific")
    from parliamo.tts.kokoro_backend import KokoroBackend

    _, note = KokoroBackend().resolve_language("fur")
    assert note, "Friulian fell back to another language without reporting it"
    assert cfg.tts.friulian_speaks_as in note


def test_default_voicepack_belongs_to_the_output_language(cfg) -> None:
    if cfg.tts.backend != "kokoro":
        pytest.skip("Kokoro-specific")
    from parliamo.tts.kokoro_backend import DEFAULT_VOICES

    expected = DEFAULT_VOICES.get(cfg.tts.language)
    assert cfg.tts.voice == expected, (
        f"tts.voice={cfg.tts.voice!r} is not the default voicepack for "
        f"tts.language={cfg.tts.language!r} ({expected!r})"
    )


# ---------------------------------------------------------------------------
# values the pipeline depends on
# ---------------------------------------------------------------------------


def test_capture_block_is_exactly_one_silero_frame(cfg) -> None:
    """Silero needs 512 samples at 16 kHz, and the segmenter assumes it.

    The Reblocker absorbs a mismatch, but a config that does not line up here
    means every block is re-framed for no reason.
    """
    from parliamo.audio.vad import SILERO_BLOCK, SILERO_SAMPLE_RATE

    assert cfg.audio.sample_rate == SILERO_SAMPLE_RATE
    assert cfg.audio.block_frames == SILERO_BLOCK


def test_hotword_file_named_by_config_exists(cfg) -> None:
    from parliamo.paths import resolve

    if not cfg.asr.hotwords:
        pytest.skip("hotwords disabled")
    path = resolve(cfg.asr.hotwords)
    assert path.exists(), (
        f"asr.hotwords points at {path}, which does not exist. Recognition would "
        "silently run unbiased - measured at +2.21 WER points for turbo."
    )


def test_consent_is_empty_by_default(cfg) -> None:
    """Shipping a consent record in the defaults would authorise cloning for free."""
    assert cfg.tts.conversion.consent == ""
    assert cfg.tts.conversion.reference_voice is None


def test_half_duplex_is_on_by_default(cfg) -> None:
    """Off by default is a feedback loop in front of an audience."""
    assert cfg.pipeline.half_duplex is True
