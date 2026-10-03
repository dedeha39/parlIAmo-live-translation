"""Detecting the repetition loop, which is Whisper's worst failure mode on stage.

What it looks like
------------------
Given ambiguous or noisy audio, an autoregressive decoder can fall into a cycle
and emit the same phrase until it hits the token limit. A real example from the
first Turkish benchmark run, against a 4-second reference::

    bu videonun ve videonun ve videonun ve videonun ve videonun ve videonu...

That single utterance scored 1009% WER on its own. In a benchmark it distorts
the aggregate; in a live presentation it floods the translation stage with
nonsense, blocks the queue behind it, and cannot recover on its own.

Why not just use temperature fallback
-------------------------------------
Whisper's built-in remedy is to retry the segment at increasing temperature
until the output looks sane. That works, but it makes both the output and the
*latency* nondeterministic - a segment can silently cost five times its normal
decode budget. On stage a dropped sentence is recoverable and an unpredictable
multi-second stall is not, so we keep greedy decoding and catch the loop
afterwards instead.

The measure
-----------
gzip compression ratio, the same signal Whisper uses internally: repetitive text
compresses far better than natural language. Turkish is agglutinative and its
vowel harmony makes it compress somewhat better than English at baseline, so the
threshold is set above Whisper's default rather than at it.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

# Whisper's own default is 2.4, tuned on English. Natural Turkish sits higher
# because long shared suffixes (-lerini, -makta, -acağını) repeat across words,
# so 2.4 flags clean sentences. Measured on FLEURS Turkish references, ordinary
# text stays under ~2.8 while genuine loops run well past 4.
DEFAULT_COMPRESSION_THRESHOLD = 3.2

# A phrase repeated this many times in a row is a loop regardless of what the
# compression ratio says - it catches short loops that are too brief to compress.
DEFAULT_MAX_PHRASE_REPEATS = 4


@dataclass(slots=True)
class RepetitionReport:
    compression_ratio: float
    max_phrase_repeats: int
    repeated_phrase: str
    is_repetitive: bool
    reason: str = ""

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "compression_ratio": round(self.compression_ratio, 3),
            "max_phrase_repeats": self.max_phrase_repeats,
            "repeated_phrase": self.repeated_phrase,
            "is_repetitive": self.is_repetitive,
            "reason": self.reason,
        }


def compression_ratio(text: str) -> float:
    """Ratio of raw bytes to gzip-compressed bytes. Higher means more repetitive."""
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    if len(raw) < 32:
        # Too short for compression statistics to mean anything; the header
        # dominates and every short string looks incompressible.
        return 0.0
    return len(raw) / len(zlib.compress(raw, level=6))


def max_phrase_repeats(text: str, max_phrase_words: int = 6) -> tuple[int, str]:
    """Longest run of one phrase repeated back to back.

    Returns ``(count, phrase)``. A count of 1 means nothing repeats.
    """
    words = text.split()
    if len(words) < 4:
        return 1, ""

    best_count, best_phrase = 1, ""
    for size in range(1, min(max_phrase_words, len(words) // 2) + 1):
        index = 0
        while index + size <= len(words):
            phrase = words[index : index + size]
            count = 1
            cursor = index + size
            while cursor + size <= len(words) and words[cursor : cursor + size] == phrase:
                count += 1
                cursor += size
            if count > best_count:
                best_count, best_phrase = count, " ".join(phrase)
            index += 1
    return best_count, best_phrase


def analyse(
    text: str,
    compression_threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
    phrase_repeat_limit: int = DEFAULT_MAX_PHRASE_REPEATS,
) -> RepetitionReport:
    """Decide whether *text* is a decoder loop rather than speech."""
    ratio = compression_ratio(text)
    repeats, phrase = max_phrase_repeats(text)

    reasons: list[str] = []
    if ratio >= compression_threshold:
        reasons.append(f"compression ratio {ratio:.2f} >= {compression_threshold}")
    if repeats >= phrase_repeat_limit:
        reasons.append(f"phrase {phrase!r} repeated {repeats} times")

    return RepetitionReport(
        compression_ratio=ratio,
        max_phrase_repeats=repeats,
        repeated_phrase=phrase,
        is_repetitive=bool(reasons),
        reason="; ".join(reasons),
    )


def collapse_repeats(text: str, min_repeats: int = DEFAULT_MAX_PHRASE_REPEATS,
                     max_phrase_words: int = 6) -> str:
    """*text* with every back-to-back run of a phrase said once.

    A loop is usually a real phrase the decoder could not stop saying: in
    rehearsal "Brad Pitt'in annesinden sonra da" came back 24 times for 2.7 s
    of speech that said it once. Runs shorter than *min_repeats* are left
    alone - "evet evet" is how people talk.
    """
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        best_size, best_count = 0, 1
        for size in range(1, max_phrase_words + 1):
            phrase = words[i:i + size]
            if len(phrase) < size:
                break
            count, cursor = 1, i + size
            while words[cursor:cursor + size] == phrase:
                count += 1
                cursor += size
            if count >= min_repeats and count * size > best_count * best_size:
                best_size, best_count = size, count
        if best_size:
            phrase = words[i:i + best_size]
            out.extend(phrase)
            i += best_size * best_count
            # The decoder ran out of room mid-phrase: a partial repeat to the
            # end of the text is the same loop, not more speech.
            tail = words[i:]
            if tail and len(tail) < best_size and tail == phrase[:len(tail)]:
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)
