"""Speaking a sentence the moment it ends, instead of at the pause.

The presenter asked for this, twice, after hearing why it is off by default.
These tests hold what it promises and what it must never do.
"""

from __future__ import annotations

import numpy as np
import pytest

from parliamo.pipeline.translator import LiveTranslator

# ---------------------------------------------------------------------------
# a translator with fake backends, so every stage is observable
# ---------------------------------------------------------------------------


class _MT:
    source_lang = "tr"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, text, target_lang=None):
        self.calls.append(text)
        return type("R", (), {"text": f"IT[{text}]"})()


class _TTS:
    sample_rate = 24000

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text, language=None):
        self.spoken.append(text)
        return type("S", (), {"audio": np.zeros(240, dtype=np.float32),
                              "sample_rate": 24000, "duration_s": 0.01})()


class _Playback:
    def __init__(self) -> None:
        self.submitted = 0

    def submit(self, audio, sample_rate=None):
        self.submitted += 1


class _Segment:
    start_s = 0.0
    end_s = 1.0
    duration_s = 1.0


def _event(text: str, *, partial: bool, index: int = 1):
    from parliamo.pipeline.transcriber import TranscriptEvent

    return TranscriptEvent(index=index, text=text, segment=_Segment(), transcript=None,
                           commit_wait_s=0.5, queue_wait_s=0.0, asr_s=0.3, partial=partial)


def _hear(translator, text: str, *, times: int = 2) -> None:
    """Feed a partial the way it really arrives: repeated every 800 ms.

    Local agreement speaks a finished sentence on its second identical
    sighting, so a stable reading takes two calls; one call is a reading the
    recogniser has not yet confirmed.
    """
    for _ in range(times):
        translator._deliver(_event(text, partial=True))


@pytest.fixture
def rig():
    mt, tts, pb = _MT(), _TTS(), _Playback()
    transcriber = type("T", (), {"on_transcript": None})()
    translator = LiveTranslator(transcriber, mt, tts, playback=pb, target_lang="it")
    deliveries = []
    translator.on_delivery = deliveries.append
    return translator, mt, tts, pb, deliveries


# ---------------------------------------------------------------------------
# off: the documented behaviour is untouched
# ---------------------------------------------------------------------------


def test_off_a_partial_is_a_subtitle_and_never_spoken(rig) -> None:
    translator, _mt, tts, pb, deliveries = rig
    translator.stream_sentences = False
    translator._deliver(_event("Bugün hava güzel. Yarın da", partial=True))
    assert tts.spoken == []
    assert pb.submitted == 0
    assert deliveries and deliveries[-1].partial is True


# ---------------------------------------------------------------------------
# on: a finished sentence is spoken before the pause
# ---------------------------------------------------------------------------


def test_on_a_finished_sentence_in_a_partial_is_spoken_at_once(rig) -> None:
    """The point of the feature: the first sentence of a long run goes out early."""
    translator, _mt, tts, pb, deliveries = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel. Yarın da")
    assert tts.spoken == ["IT[Bugün hava güzel.]"]
    assert pb.submitted == 1
    spoken = [d for d in deliveries if d.spoken]
    assert len(spoken) == 1 and spoken[0].partial is False


def test_the_unfinished_tail_stays_a_subtitle(rig) -> None:
    """The tail has no verb yet. Turkish puts it last."""
    translator, _mt, tts, _pb, deliveries = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel. Yarın da")
    previews = [d for d in deliveries if d.partial]
    assert previews and previews[-1].source_text == "Yarın da"
    assert "Yarın" not in " ".join(tts.spoken)


def test_a_sentence_is_spoken_once_across_successive_partials(rig) -> None:
    """Partials arrive every 800 ms and repeat everything so far."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel.", times=1)
    _hear(translator, "Bugün hava güzel. Yarın", times=1)          # second sighting
    _hear(translator, "Bugün hava güzel. Yarın da güzel", times=1)
    assert tts.spoken == ["IT[Bugün hava güzel.]"]


def test_the_commit_does_not_repeat_what_streaming_said(rig) -> None:
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel. Yarın da")
    translator._deliver(_event("Bugün hava güzel. Yarın da güzel olacak.", partial=False))
    assert tts.spoken == ["IT[Bugün hava güzel.]", "IT[Yarın da güzel olacak.]"]
    assert translator._spoken_early == [], "the ledger is cleared at the commit"


def test_a_revised_reading_of_a_spoken_sentence_is_not_spoken_again(rig) -> None:
    """The recogniser revises: a sentence spoken early may commit with a word added.

    Saying both would say the thing twice. Eighty percent similarity, or the
    spoken text as a prefix, counts as the same sentence.
    """
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel.")
    translator._deliver(_event("Bugün hava çok güzel.", partial=False))
    assert tts.spoken == ["IT[Bugün hava güzel.]"]


def test_a_genuinely_different_sentence_is_spoken(rig) -> None:
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel.")
    translator._deliver(_event("Bugün hava güzel. Toplantı iptal edildi.", partial=False))
    assert tts.spoken == ["IT[Bugün hava güzel.]", "IT[Toplantı iptal edildi.]"]


def test_a_commit_with_nothing_new_delivers_nothing(rig) -> None:
    translator, _mt, tts, _pb, deliveries = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel.")
    before = len(deliveries)
    result = translator._deliver(_event("Bugün hava güzel.", partial=False))
    assert result is None
    assert len(deliveries) == before
    assert tts.spoken == ["IT[Bugün hava güzel.]"]


# ---------------------------------------------------------------------------
# the risk, stated as a test so nobody forgets it is real
# ---------------------------------------------------------------------------


def test_the_documented_risk_a_full_stop_one_word_early(rig) -> None:
    """This is what the presenter accepted, and the least bad way out of it.

    If the recogniser closes the sentence before the negation, the affirmative
    is spoken - streaming cannot prevent that. The committed sentence then
    *starts with* what was spoken and would be taken as the same sentence,
    leaving the negation unsaid forever. So a committed sentence that adds a
    clause-final negation the early reading lacked is spoken as a correction:
    the room hears the wrong half and then the right one, which is bad, and
    better than hearing only the wrong half.
    """
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Bu teknik bir mesele.")
    translator._deliver(_event("Bu teknik bir mesele değil.", partial=False))
    assert tts.spoken == ["IT[Bu teknik bir mesele.]", "IT[Bu teknik bir mesele değil.]"]


def test_a_revision_without_a_negation_is_still_not_repeated(rig) -> None:
    """The correction path is for negations only; ordinary revisions stay quiet."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Bu teknik bir mesele.")
    translator._deliver(_event("Bu teknik bir mesele aslında.", partial=False))
    assert tts.spoken == ["IT[Bu teknik bir mesele.]"]


def test_switching_off_clears_the_ledger(rig) -> None:
    from parliamo.ui.controller import PipelineController

    translator, _mt, _tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel.")
    assert translator._spoken_early

    controller = PipelineController(factory=lambda: (None, None))
    controller.translator = translator
    controller.set_streaming(False)
    assert translator.stream_sentences is False
    assert translator._spoken_early == []


# ---------------------------------------------------------------------------
# local agreement: the fix for what the first measurement found
# ---------------------------------------------------------------------------


def test_a_sentence_seen_once_is_not_spoken(rig) -> None:
    """Measured before this existed: 24 sentences spoken where the speaker
    said 11, the extras being confident early misreadings of the first words
    - "Ciao, che succede oggi?" for what became "Salve, oggi vorrei
    parlarvi...". A reading the next partial does not repeat is not trusted.
    """
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Ciao, che succede oggi?", times=1)
    _hear(translator, "Salve, oggi vorrei parlarvi di una cosa.", times=1)
    assert tts.spoken == [], "neither reading was confirmed by the next one"
    _hear(translator, "Salve, oggi vorrei parlarvi di una cosa.", times=1)
    assert tts.spoken == ["IT[Salve, oggi vorrei parlarvi di una cosa.]"]


def test_an_ellipsis_does_not_end_a_sentence(rig) -> None:
    """"..." is what the recogniser writes when the audio ran out mid-word."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Salve, oggi c'è una sensazione di...", times=3)
    assert tts.spoken == []


def test_sightings_reset_at_the_commit(rig) -> None:
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    _hear(translator, "Bugün hava güzel.", times=1)
    translator._deliver(_event("Bugün hava güzel.", partial=False))
    assert translator._seen_once == set()
    assert tts.spoken == ["IT[Bugün hava güzel.]"], "the commit spoke it, once"


def test_deliveries_carry_the_audio_position(rig) -> None:
    """The one honest clock for comparing policies on a file replay."""
    translator, _mt, _tts, _pb, deliveries = rig
    seg = _Segment()
    seg.end_s = 46.5
    from parliamo.pipeline.transcriber import TranscriptEvent

    ev = TranscriptEvent(index=1, text="Bugün hava güzel.", segment=seg, transcript=None,
                         commit_wait_s=0.5, queue_wait_s=0.0, asr_s=0.3, partial=False)
    translator._deliver(ev)
    assert deliveries[-1].audio_end_s == 46.5
    assert deliveries[-1].as_dict()["audio_end_s"] == 46.5


# ---------------------------------------------------------------------------
# chunk mode: the buffer
# ---------------------------------------------------------------------------


def test_chunks_speak_settled_words_before_the_sentence_ends(rig) -> None:
    """Four words agreed by two consecutive readings go out at once."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("Bugün hava çok güzel ve", partial=True))
    assert tts.spoken == [], "one reading agrees with nothing yet"
    translator._deliver(_event("Bugün hava çok güzel ve yarın", partial=True))
    assert tts.spoken == ["IT[Bugün hava çok güzel ve]"], (
        "everything that has settled, once there are at least four words")


def test_chunks_do_not_repeat_and_the_commit_finishes_the_rest(rig) -> None:
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("Bugün hava çok güzel ve", partial=True))
    translator._deliver(_event("Bugün hava çok güzel ve yarın da", partial=True))
    translator._deliver(_event("Bugün hava çok güzel ve yarın da güzel", partial=True))
    translator._deliver(_event("Bugün hava çok güzel ve yarın da güzel olacak.", partial=False))
    assert tts.spoken == ["IT[Bugün hava çok güzel ve]", "IT[yarın da güzel olacak.]"]
    assert translator._chunk_spoken == [] and translator._prev_words == []


def test_a_revision_of_unspoken_words_is_simply_waited_out(rig) -> None:
    """Words that change between readings are not settled and are not spoken."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("Bugün hava çok güzel", partial=True))
    translator._deliver(_event("Bugün hava çok kötü", partial=True))   # last word changed
    assert tts.spoken == [], "only three words agree, below the chunk size"


def test_punctuation_closes_a_short_chunk(rig) -> None:
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("Evet, doğru.", partial=True))
    translator._deliver(_event("Evet, doğru. Ama", partial=True))
    assert tts.spoken == ["IT[Evet, doğru.]"]


def test_the_chunk_cost_the_verb_arrives_after_the_room_heard_the_clause(rig) -> None:
    """What the presenter accepted by asking for a buffer.

    Turkish keeps the verb and the negation for the end. The first chunk is a
    clause with no verb; the translator does what it can with it, and the
    negation comes in the last piece, after the room has heard the rest.
    """
    translator, mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("Bu teknik bir mesele olarak", partial=True))
    translator._deliver(_event("Bu teknik bir mesele olarak görülmemeli", partial=True))
    translator._deliver(_event("Bu teknik bir mesele olarak görülmemeli değil mi.", partial=False))
    assert mt.calls[0] == "Bu teknik bir mesele olarak", "a clause with no verb"
    assert tts.spoken[-1] == "IT[görülmemeli değil mi.]", "the verb and negation, last"


# ---------------------------------------------------------------------------
# keeping up: a sentence in the generic voice beats a sentence not said
# ---------------------------------------------------------------------------


class _Converter:
    def __init__(self) -> None:
        self.calls = 0

    def convert(self, audio, rate, reference, **options):
        self.calls += 1
        return type("C", (), {"audio": audio, "sample_rate": rate, "duration_s": 0.01,
                              "round_trip_s": 0.9})()


def test_conversion_is_skipped_when_the_queue_is_backed_up(rig) -> None:
    """Seen live: three sentences dropped in eight seconds, six seconds on the screen.

    Conversion runs one sentence at a time and costs ~1 s. When the queue
    holds two or more waiting sentences, this one is spoken generic - the
    rule the pipeline already applies when conversion *fails*, applied before
    the queue overflows instead of after.
    """
    translator, _mt, _tts, _pb, deliveries = rig
    translator.converter = _Converter()
    translator.reference_voice = "ref.wav"

    translator._deliver(_event("Birinci cümle.", partial=False))
    assert deliveries[-1].voice == "cloned"
    assert translator.converter.calls == 1

    translator._queue.put_nowait(_event("İkinci.", partial=False))
    translator._queue.put_nowait(_event("Üçüncü.", partial=False))
    translator._deliver(_event("Dördüncü cümle.", partial=False))
    assert deliveries[-1].voice == "default"
    assert deliveries[-1].spoken is True, "spoken, not dropped"
    assert "delivery behind" in deliveries[-1].error
    assert translator.converter.calls == 1, "no conversion attempted while behind"
    assert translator.stats.conversion_skipped == 1


# ---------------------------------------------------------------------------
# a commit that never comes must not poison the next utterance
# ---------------------------------------------------------------------------


class _SegAt:
    def __init__(self, start: float) -> None:
        self.start_s, self.end_s, self.duration_s = start, start + 3.0, 3.0


def _ev_at(text: str, *, partial: bool, start: float):
    from parliamo.pipeline.transcriber import TranscriptEvent

    return TranscriptEvent(index=1, text=text, segment=_SegAt(start), transcript=None,
                           commit_wait_s=0.5, queue_wait_s=0.0, asr_s=0.3, partial=partial)


def test_chunk_state_does_not_leak_into_the_next_utterance(rig) -> None:
    """The regression: a whole sentence was never spoken.

    The recogniser drops a segment that decodes to nothing or to a loop and
    emits no commit. The ledger said five words were already spoken, and the
    next utterance - a different sentence - was sliced from word six.
    """
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    for text in ("bir iki üç dört beş", "bir iki üç dört beş altı"):
        translator._deliver(_ev_at(text, partial=True, start=10.0))
    assert tts.spoken == ["IT[bir iki üç dört beş]"]
    # no commit for that utterance; the next one starts at 20.0 s
    for text in ("Toplantı yarın saat üçte", "Toplantı yarın saat üçte başlayacak"):
        translator._deliver(_ev_at(text, partial=True, start=20.0))
    translator._deliver(_ev_at("Toplantı yarın saat üçte başlayacak.", partial=False, start=20.0))
    said = " ".join(tts.spoken[1:])
    assert "Toplantı" in said and "başlayacak" in said, tts.spoken


def test_a_sentence_said_again_in_a_new_utterance_is_spoken_again(rig) -> None:
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    for _ in range(2):
        translator._deliver(_ev_at("Hava güzel. Ve", partial=True, start=10.0))
    for _ in range(2):
        translator._deliver(_ev_at("Hava güzel.", partial=True, start=20.0))
    assert tts.spoken == ["IT[Hava güzel.]", "IT[Hava güzel.]"]


def test_partials_and_commit_of_one_utterance_share_the_ledger(rig) -> None:
    """The same start time is the same utterance: nothing is spoken twice."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_sentences = True
    for _ in range(2):
        translator._deliver(_ev_at("Hava güzel. Yarın", partial=True, start=10.0))
    translator._deliver(_ev_at("Hava güzel. Yarın gel.", partial=False, start=10.0))
    assert tts.spoken == ["IT[Hava güzel.]", "IT[Yarın gel.]"]


# ---------------------------------------------------------------------------
# chunk mode: the commit is a new reading, not the last partial plus words
# ---------------------------------------------------------------------------


def test_a_commit_that_joins_two_spoken_words_does_not_lose_the_next(rig) -> None:
    """The commit re-decodes the whole utterance and may segment it differently.

    The partials read "üç te"; the commit reads "üçte". The room has heard all
    of it. Sliced by count - five words spoken, so the rest starts at word six
    - the commit had only five words, and "başlayacak" was never spoken.
    """
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("Toplantı yarın saat üç te", partial=True))
    translator._deliver(_event("Toplantı yarın saat üç te başla", partial=True))
    assert tts.spoken == ["IT[Toplantı yarın saat üç te]"]
    translator._deliver(_event("Toplantı yarın saat üçte başlayacak.", partial=False))
    assert tts.spoken[1:] == ["IT[başlayacak.]"], tts.spoken


def test_a_commit_that_finds_a_word_the_partials_missed_does_not_repeat(rig) -> None:
    """The commit hears a word at the start the partials had not; nothing is said twice."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("hava çok güzel bugün", partial=True))
    translator._deliver(_event("hava çok güzel bugün ve", partial=True))
    assert tts.spoken == ["IT[hava çok güzel bugün]"]
    translator._deliver(_event("Evet hava çok güzel bugün ve yarın da.", partial=False))
    assert tts.spoken[1:] == ["IT[ve yarın da.]"], tts.spoken


def test_a_partial_that_rereads_spoken_words_does_not_shift_the_next_chunk(rig) -> None:
    """A later partial joins two words the room already heard; the next chunk starts right."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("Toplantı yarın saat üç te", partial=True))
    translator._deliver(_event("Toplantı yarın saat üç te başla", partial=True))
    translator._deliver(_event("Toplantı yarın saat üçte başlayacak ve herkes orada", partial=True))
    translator._deliver(_event("Toplantı yarın saat üçte başlayacak ve herkes orada olacak",
                               partial=True))
    assert tts.spoken == ["IT[Toplantı yarın saat üç te]",
                          "IT[başlayacak ve herkes orada]"], tts.spoken


def test_a_commit_that_rewrites_everything_falls_back_to_length(rig) -> None:
    """Nothing aligns: what was spoken is covered by its length, not dropped whole or repeated whole."""
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("aaa bbb ccc ddd", partial=True))
    translator._deliver(_event("aaa bbb ccc ddd eee", partial=True))
    translator._deliver(_event("xxx yyy zzz www son.", partial=False))
    assert tts.spoken[1:] == ["IT[son.]"], tts.spoken


def test_a_full_stop_that_comes_and_goes_does_not_hold_a_chunk_back(rig) -> None:
    """Punctuation flickers between readings; the words agree all the same.

    On the reference recording "istiyorum" and "istiyorum." alternated from
    one partial to the next, and each flip cost a chunk one reading - 800 ms.
    Ignoring punctuation moved spoken words 0.15-0.20 s earlier on average,
    with the same words missing and repeated as before.
    """
    translator, _mt, tts, _pb, _d = rig
    translator.stream_mode = "chunk"
    translator._deliver(_event("Bugün size üç şey göstermek istiyorum.", partial=True))
    translator._deliver(_event("Bugün size üç şey göstermek istiyorum Önce", partial=True))
    assert tts.spoken == ["IT[Bugün size üç şey göstermek istiyorum]"], tts.spoken
