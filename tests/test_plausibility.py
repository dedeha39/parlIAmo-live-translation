"""A sentence the audio could not have held is dropped; real speech never is.

The cases are real readings from the 68 s reference recording.
"""

from __future__ import annotations

from parliamo.asr.plausibility import MAX_SYLLABLES_PER_S, syllables, too_fast


def test_thank_you_for_watching_from_a_breath_is_dropped() -> None:
    """Whisper's subtitle-track ending, from the first 0.93 s of an utterance."""
    invented, rate = too_fast("İzlediğiniz için teşekkür ederim.", 0.93)
    assert invented and rate > 13


def test_a_misreading_at_the_same_position_is_dropped_too() -> None:
    """What a phrase list would have missed."""
    assert too_fast("Friulian Friulian Friuli Börgesine Kodur", 0.93)[0]


def test_the_fastest_real_readings_are_kept() -> None:
    """9.7 syllables/s: the fastest reading on the recording that was speech."""
    assert not too_fast("Bugün size üç şey gönderdim", 0.93)[0]
    assert not too_fast("Bu sorunumu hazırlarken", 0.93)[0]
    assert not too_fast("Friuli Bölgesi'ne konuşulan Friuli Anadolu", 1.73)[0]


def test_every_committed_sentence_is_far_below_the_limit() -> None:
    sentence = ("Bugün size üç şey göstermek istiyorum. Önce sistemin nasıl "
                "çalıştığını, sonra bir sesin ne kadar hızlı kopyalanabildiğini.")
    invented, rate = too_fast(sentence, 7.2)
    assert not invented and rate < MAX_SYLLABLES_PER_S / 1.5


def test_one_short_word_on_a_short_buffer_is_not_judged() -> None:
    assert too_fast("Evet, tamam.", 0.3) == (False, 0.0)


def test_turkish_counts_every_vowel_other_languages_count_diphthongs_once() -> None:
    assert syllables("İzlediğiniz için teşekkür ederim") == 13
    assert syllables("saat") == 2, "Turkish has no diphthongs: sa-at"
    assert syllables("Friulian") == 4, "counted as runs it was 2, and a misreading got through"
    assert syllables("vuoi", "it") == 1, "an Italian diphthong is one syllable"


def _backend_returning(text: str):
    import types

    from parliamo.asr.faster_whisper_backend import FasterWhisperBackend

    backend = FasterWhisperBackend(device="cpu")
    segment = types.SimpleNamespace(start=0.0, end=0.9, text=text, no_speech_prob=0.1,
                                    avg_logprob=-0.3)
    backend._model = types.SimpleNamespace(
        transcribe=lambda *a, **k: (iter([segment]),
                                    types.SimpleNamespace(language="tr", language_probability=0.9)))
    backend._loaded = True
    return backend


def test_the_recogniser_returns_nothing_for_an_invented_sentence_and_says_why() -> None:
    import numpy as np

    backend = _backend_returning("İzlediğiniz için teşekkür ederim.")
    transcript = backend.transcribe(np.zeros(int(0.93 * 16000), dtype=np.float32))
    assert transcript.text == ""
    assert transcript.dropped_segments[0]["syllables_per_s"] > 13


def test_the_recogniser_keeps_the_same_sentence_when_there_was_time_to_say_it() -> None:
    """At the end of a talk someone may really say it; 2.5 s is enough time."""
    import numpy as np

    backend = _backend_returning("İzlediğiniz için teşekkür ederim.")
    transcript = backend.transcribe(np.zeros(int(2.5 * 16000), dtype=np.float32))
    assert transcript.text == "İzlediğiniz için teşekkür ederim."
