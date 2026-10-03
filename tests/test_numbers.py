"""Digits spelled out for OmniVoice, which read "15 secondi" as "chi me è secondo"."""

from __future__ import annotations

import pytest

from parliamo.tts.numbers import spell_numbers

pytest.importorskip("num2words")


def test_italian_numbers_are_words() -> None:
    assert spell_numbers("Una registrazione di 15 secondi basta.", "it") == (
        "Una registrazione di quindici secondi basta.")
    assert spell_numbers("ha perso 25 milioni di dollari", "it") == (
        "ha perso venticinque milioni di dollari")


def test_thousands_are_grouped_the_way_the_language_writes_them() -> None:
    assert spell_numbers("circa 600.000 persone", "it") == "circa seicentomila persone"
    assert spell_numbers("about 600,000 people", "en") == "about six hundred thousand people"


def test_friulian_is_spelled_in_italian_like_its_voice() -> None:
    assert spell_numbers("3 robis", "fur") == "tre robis"


def test_text_without_digits_is_returned_as_is() -> None:
    text = "Prima riattaccate, poi richiamate."
    assert spell_numbers(text, "it") is text


def test_a_language_num2words_does_not_know_is_left_alone() -> None:
    assert spell_numbers("15", "xx") == "15"
