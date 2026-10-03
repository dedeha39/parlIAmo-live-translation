"""Text the audio could not have held: Whisper's other hallucination.

The repetition loop (``repetition.py``) is one failure. The other is a whole,
fluent sentence from a buffer with almost nothing in it. Whisper was trained on
subtitled video, and on a breath or the first fraction of a second of a new
utterance it produces what a subtitle track ends with. On the reference
recording, the first 0.93 s partial of one utterance came back as::

    İzlediğiniz için teşekkür ederim.      ("thank you for watching")

Thirteen syllables in 0.93 s - 14 per second. Turkish is spoken at about 5-7;
across 750 readings of the same recording the median was 6.7, every committed
sentence stayed under 7.5, and the fastest reading that was real speech was 9.7
(a partial whose last word Whisper completed early). Everything above 12 was
invented: that sentence, and "Friulian Friuli Börgesine Kodur" from the same
position in another utterance.

A phrase blacklist would catch the first and not the second, and would also
drop the sentence if the speaker actually says it at the end of the talk. A
rate cannot be said by anyone, so it is the rule.

One narrow class is caught by its words after all: **subtitle credits**. On a
0.54 s breath with a mean speech probability of 0.12, rehearsal produced
"Altyazı M.K." - a subtitler's signature from Whisper's training videos. Five
syllables are too few for the rate to judge, and it would have been spoken as
"Sottotitoli M.K.". Unlike "thank you for watching", no presenter says a
subtitle credit, so these are matched whole - and "Altyazılar da ekranda
görünüyor", a real line of this talk, is not.

Syllables: in Turkish every vowel is one - there are no diphthongs, "saat" is
sa-at - so the count is exact, and it is what the limit was measured with. In
other languages adjacent vowels are counted once, so an Italian or Spanish
diphthong is not counted twice; where that undercounts, it errs toward keeping
text. (Counting runs in Turkish too let "Friulian Friuli Börgesine Kodur"
through: "Friulian" became two syllables instead of four.)
"""

from __future__ import annotations

import re
import unicodedata

#: Faster than anyone speaks. Real readings topped out at 9.7 syllables/s on
#: the reference recording; hallucinations started at 14.
MAX_SYLLABLES_PER_S = 12.0

#: Below this many syllables the rate says nothing: one short word on a short
#: buffer can be briefly fast without being invented.
MIN_SYLLABLES_TO_JUDGE = 6

#: Whole transcripts that are a subtitle credit, not speech. Anchored at both
#: ends and kept short on purpose: a sentence that merely mentions subtitles
#: must never match.
_CREDIT_LINES = re.compile(
    r"^\s*(altyaz[ıi]|çeviri|sottotitoli(\s+a\s+cura\s+di)?|subtitles?(\s+by)?|untertitel)"
    r"\s*[:\-]?\s*[\w.]{0,12}(\s[\w.]{1,12})?\s*\.?\s*$",
    re.IGNORECASE,
)


#: The hotword prompt recited: at least this many of its words in their order...
HOTWORD_ECHO_RUN = 4
#: ...and this share of the transcript hotwords.
HOTWORD_ECHO_SHARE = 0.8
_LETTERS = re.compile(r"[^\W\d_]+")


def _words(text: str) -> list[str]:
    # "İ".casefold() is "i" plus a combining dot, which is not a letter: the
    # bare split read "İpek" as two words, "i" and "pek".
    plain = unicodedata.normalize("NFD", text.casefold())
    return _LETTERS.findall("".join(c for c in plain if not unicodedata.combining(c)))


def hotword_echo(text: str, hotwords: str | None) -> bool:
    """Is *text* Whisper reciting its own hotword prompt?

    On 2026-09-27 and 2026-10-02, with the microphone delivering speech at
    about -40 dBFS, Whisper answered dozens of segments with the start of the
    list it is biased toward - "İpek Nur Yıldız Acme" 13 times, "... Acme
    Valdastra Friuli" 33 times, sometimes with a stray word after it - slowly
    enough to pass the rate check, and they were spoken.

    A recital keeps the prompt's order; speech does not. So: four or more
    hotwords in the prompt's own order, and the transcript nearly all hotwords.
    "İpek Nur Yıldız." alone (three) is kept - it is how the talk opens - and
    so is "Ben İpek Nur Yıldız, Acme'de çalışıyorum" (four in order, but four
    of seven words). "Yapay zekâ ile ses klonlama" is hotwords out of order.
    """
    if not text or not hotwords:
        return False
    prompt = _words(hotwords)
    words = _words(text)
    if len(words) < HOTWORD_ECHO_RUN or not prompt:
        return False
    vocabulary = set(prompt)
    if sum(word in vocabulary for word in words) / len(words) < HOTWORD_ECHO_SHARE:
        return False
    # Longest run of the transcript that is also a run of the prompt.
    longest = 0
    previous = [0] * (len(prompt) + 1)
    for word in words:
        current = [0] * (len(prompt) + 1)
        for j, hot in enumerate(prompt, start=1):
            if word == hot:
                current[j] = previous[j - 1] + 1
                longest = max(longest, current[j])
        previous = current
    return longest >= HOTWORD_ECHO_RUN


def subtitle_credit(text: str) -> bool:
    """Is *text*, whole, a subtitle credit ("Altyazı M.K.") rather than speech?"""
    return bool(text) and bool(_CREDIT_LINES.match(text))


# Turkish, Italian, Spanish and German vowels. ä and á were missing, so German
# and Spanish readings counted fewer syllables than they had.
_VOWELS = "aeıioöuüâîûàáèéìíòóùúäëïÿAEIİOÖUÜÂÎÛÀÁÈÉÌÍÒÓÙÚÄËÏ"
_VOWEL_RUN = re.compile(f"[{_VOWELS}]+")


def syllables(text: str, language: str = "tr") -> int:
    """Syllables in *text*: every vowel in Turkish, runs of vowels elsewhere."""
    if language == "tr":
        return sum(c in _VOWELS for c in text)
    return len(_VOWEL_RUN.findall(text))


def too_fast(text: str, duration_s: float, language: str = "tr",
             max_rate: float = MAX_SYLLABLES_PER_S) -> tuple[bool, float]:
    """Could *text* have been spoken in *duration_s*? Returns ``(invented, rate)``."""
    count = syllables(text, language)
    if count < MIN_SYLLABLES_TO_JUDGE or duration_s <= 0:
        return False, 0.0
    rate = count / duration_s
    return rate > max_rate, rate
