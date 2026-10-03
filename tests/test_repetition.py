"""Repetition-loop detection.

The positive cases are real Whisper output captured during the first Turkish
benchmark run. The negative cases are real FLEURS Turkish references, which is
the harder half of the job: Turkish shares long suffixes across words, so it
compresses better than English and a threshold tuned on English flags perfectly
ordinary sentences.
"""

from __future__ import annotations

import pytest

from parliamo.asr.repetition import (
    DEFAULT_COMPRESSION_THRESHOLD,
    analyse,
    collapse_repeats,
    compression_ratio,
    max_phrase_repeats,
)

# Captured from faster-whisper "tiny" on a 4.2 s FLEURS utterance.
REAL_LOOP = (
    "bu videonun ve videonun ve videonun ve videonun ve videonun ve videonun ve "
    "videonun ve videonun ve videonun ve videonun ve videonun ve videonun"
)

# Real FLEURS Turkish references.
REAL_TURKISH = [
    "japonyanın nükleer ajansına göre tesiste radyoaktif sezyum ve iyodin tespit edildi",
    "avrupa nispeten küçük ama birçok bağımsız ülkenin bulunduğu bir kıtadır",
    "romantizm goethe fichte ve schlegel gibi yazarlardan geçen geniş bir kültürel harekettir",
    "apia samoanın başkentidir şehir upolu adasındadır ve kırk bin biraz altında nüfusa sahiptir",
    "güreşçi arkadaşları da lunaya saygılarını sundular",
]


# ---------------------------------------------------------------------------
# loops must be caught
# ---------------------------------------------------------------------------


def test_real_captured_loop_is_flagged() -> None:
    report = analyse(REAL_LOOP)
    assert report.is_repetitive
    assert report.max_phrase_repeats >= 4
    assert "videonun" in report.repeated_phrase


def test_single_word_loop_is_flagged() -> None:
    assert analyse("evet evet evet evet evet evet evet evet evet evet").is_repetitive


def test_short_loop_is_caught_by_phrase_counting_not_compression() -> None:
    """Too short to compress meaningfully, but obviously a loop."""
    text = "teşekkür ederim teşekkür ederim teşekkür ederim teşekkür ederim"
    report = analyse(text)
    assert report.is_repetitive
    assert report.max_phrase_repeats == 4


def test_subtitle_spam_is_flagged() -> None:
    """A classic Whisper hallucination on silence."""
    text = " ".join(["altyazı mehmet efe"] * 8)
    assert analyse(text).is_repetitive


# ---------------------------------------------------------------------------
# ordinary Turkish must not be
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", REAL_TURKISH)
def test_real_turkish_references_are_not_flagged(text: str) -> None:
    report = analyse(text)
    assert not report.is_repetitive, (
        f"false positive on real Turkish: {report.reason} "
        f"(ratio {report.compression_ratio:.2f})"
    )


def test_whole_turkish_paragraph_is_not_flagged() -> None:
    report = analyse(" ".join(REAL_TURKISH))
    assert not report.is_repetitive, report.reason


def test_threshold_leaves_headroom_over_natural_turkish() -> None:
    """The margin the threshold was chosen for, asserted rather than assumed."""
    ratios = [compression_ratio(t) for t in REAL_TURKISH if len(t) >= 32]
    assert max(ratios) < DEFAULT_COMPRESSION_THRESHOLD, (
        f"natural Turkish reaches {max(ratios):.2f}, threshold is "
        f"{DEFAULT_COMPRESSION_THRESHOLD}"
    )
    assert compression_ratio(REAL_LOOP) > DEFAULT_COMPRESSION_THRESHOLD


def test_legitimate_double_word_is_allowed() -> None:
    """Turkish reduplicates for emphasis: 'yavaş yavaş', 'tek tek'."""
    text = "yavaş yavaş ilerledik ve tek tek herkesle konuştuk sonra eve döndük"
    assert not analyse(text).is_repetitive


# ---------------------------------------------------------------------------
# the primitives
# ---------------------------------------------------------------------------


def test_empty_and_short_text_do_not_produce_a_ratio() -> None:
    assert compression_ratio("") == 0.0
    assert compression_ratio("kısa") == 0.0


def test_max_phrase_repeats_finds_multiword_phrases() -> None:
    count, phrase = max_phrase_repeats("bir iki bir iki bir iki bir iki")
    assert count == 4
    assert phrase == "bir iki"


def test_max_phrase_repeats_returns_one_when_nothing_repeats() -> None:
    count, phrase = max_phrase_repeats("bir iki üç dört beş altı yedi sekiz")
    assert count == 1
    assert phrase == ""


def test_max_phrase_repeats_ignores_very_short_input() -> None:
    assert max_phrase_repeats("bir iki") == (1, "")


def test_report_serialises() -> None:
    payload = analyse(REAL_LOOP).as_dict()
    assert payload["is_repetitive"] is True
    assert isinstance(payload["compression_ratio"], float)
    assert payload["reason"]


# -- a loop that started from a real phrase -------------------------------------

REHEARSAL_LOOP = " ".join(["Brad Pitt'in annesinden sonra da"] * 24)


def test_a_loop_is_said_once() -> None:
    assert collapse_repeats(REHEARSAL_LOOP) == "Brad Pitt'in annesinden sonra da"
    # Cut off mid-phrase by the token limit - as it came back in rehearsal.
    assert collapse_repeats(REHEARSAL_LOOP + " Brad Pitt'in annesinden") == (
        "Brad Pitt'in annesinden sonra da")
    assert collapse_repeats("Onlar yıllar " + "yıllar " * 67 + "yıllar") == "Onlar yıllar"


def test_a_short_natural_repetition_is_left_alone() -> None:
    assert collapse_repeats("evet evet haklısınız") == "evet evet haklısınız"


def _backend():
    from parliamo.asr.faster_whisper_backend import FasterWhisperBackend

    return FasterWhisperBackend(language="tr")


def test_the_rehearsal_sentence_is_kept_once_not_lost() -> None:
    from parliamo.asr.base import Segment

    kept, dropped = _backend()._filter_repetition(
        [Segment(start=0.0, end=2.66, text=REHEARSAL_LOOP)])
    assert [s.text for s in kept] == ["Brad Pitt'in annesinden sonra da"]
    assert dropped == []


@pytest.mark.parametrize("text,end", [
    (" ".join(["Friulian"] * 16), 4.12),          # a hotword out of a breath
    ("dünyanın " + " ".join(["en"] * 10), 0.92),  # too short a segment
    (" ".join(["sona"] * 101), 0.92),
])
def test_a_loop_that_said_nothing_is_still_dropped(text, end) -> None:
    from parliamo.asr.base import Segment

    kept, dropped = _backend()._filter_repetition([Segment(start=0.0, end=end, text=text)])
    assert kept == [] and len(dropped) == 1



@pytest.mark.parametrize("text", ["Altyazı M.K.", "altyazı: M.K.", "Sottotitoli a cura di QTSS",
                                  "Subtitles by", "Altyazı"])
def test_a_subtitle_credit_is_recognised(text) -> None:
    from parliamo.asr.plausibility import subtitle_credit

    assert subtitle_credit(text)


@pytest.mark.parametrize("text", ["Altyazılar da ekranda görünüyor.",
                                  "İzlediğiniz için teşekkür ederim.",
                                  "Çeviri bazen hata yapar.",
                                  "Bu altyazı sistemi internete bağlı değil."])
def test_real_sentences_about_subtitles_are_not_credits(text) -> None:
    from parliamo.asr.plausibility import subtitle_credit

    assert not subtitle_credit(text)
