# ADR 0008 — Show the subtitle before the sentence ends

**Status:** accepted · 2026-09-01
**Relates to:** ADR 0005 (streaming), ADR 0006 (the conversion cost this offsets)

## Context

The commit wait is 700 ms of the end-to-end delay, and it buys one thing:
certainty that the sentence has ended. Until then the pipeline produces nothing
at all, so the audience sees nothing while the speaker is talking.

That became more pressing when `diffusion_steps` rose from 4 to 15 (ADR 0006
amendment) and the total moved from 2.40 s to about 3.0 s. The obvious lever —
dropping the commit wait to 500 ms — buys 200 ms and costs mid-clause cuts.

There is a much larger lever available, and it does not trade anything away.

## The idle thread

While someone is speaking, the recogniser thread has **nothing to do**. It is
blocked waiting for the segmenter to close a segment, which by definition does
not happen until the speaker stops. On a 6-second sentence that is six seconds
of an idle recogniser, an idle GPU, and a blank screen.

A provisional recognition of the utterance-so-far runs in that time. It does not
compete with the committed sentence, because the committed sentence does not
exist yet.

## Decision

Emit a **partial** every `pipeline.partial_interval_ms` (800 ms) once an
utterance is at least `min_partial_ms` (900 ms) long. Recognise it, translate
it, display it. Replace it with the next partial, and finally with the committed
sentence.

Enabled by default: `pipeline.emit_partial_transcripts` was already in the
config and was read by nothing.

## Two invariants, both tested

**A partial is never spoken.** This is not a performance decision. Turkish is
SOV and marks negation as a suffix on the final verb:

> Bu teknik bir mesele **değil**, etik bir mesele.

Everything before the last word is consistent with the opposite meaning. A
subtitle that says the wrong thing for 800 ms and then corrects itself is
survivable, because the audience sees the correction. Speech is not: audio
cannot be un-said, and the room has already heard it. This is the same failure
ADR 0004 rejected an LLM for, and it would be self-inflicted.

**A partial never delays the sentence it previews.** `_enqueue_partial` drops
the partial whenever anything is already queued for the recogniser, and counts
the drop as `partials_skipped_busy`. A preview that costs the thing it previews
has inverted its own purpose.

## Measured

Time to first text on screen, from the segmenter's own constants:

| Sentence | With partials | Without | Earlier by |
|---|---|---|---|
| 3 s | 1.68 s | 4.48 s | **2.80 s** |
| 5 s | 1.68 s | 6.48 s | **4.80 s** |
| 7 s | 1.68 s | 8.48 s | **6.80 s** |

The first figure is constant because it depends on `min_partial_ms`, not on how
long the speaker goes on. The longer the sentence, the more this wins.

Replayed over five recorded sentences, the committed sentences' own latency was
unchanged — 1.20 s and 1.37 s against 1.21 s and 1.44 s without partials, which
is inside the run-to-run spread. The idle time was genuinely idle.

### What the audience actually sees

From the replay, one sentence's partials in order:

```
Merhaba, bugün sizlerle biraz tek...        -> Salve, oggi ho un po' di...
Merhaba bugün sizlerle biraz teknolojiden bahsettim
                                            -> Ciao, oggi vi ho parlato di tecnologia.
Merhaba, bugün sizlerle biraz teknolojiden bahsetmek istiyorum.
                                            -> Salve, oggi vorrei parlarvi di tecnologia.
```

It converges. The intermediate readings are wrong in the way a person guessing
ahead is wrong — tense and completion — not in the way that misleads.

## Costs, stated plainly

* **GPU time that would otherwise be idle.** Not free in absolute terms: each
  partial is an ASR pass plus an MT pass. It is free in *wall-clock* terms only
  as long as the recogniser really is idle, which the queue check enforces.
* **A subtitle that changes.** On a terminal each partial prints on its own
  line, which is noisy; on a real subtitle surface it would overwrite in place.
  The operator interface, when it exists, should render it that way.
* **Numbering.** A partial carries the number of the sentence it previews, not
  the one before it, or the subtitle appears to jump backwards.

## Alternatives not taken

**LocalAgreement-2**, as used by `whisper_streaming`: run recognition on a
growing buffer and emit only the prefix two consecutive runs agree on. Stabler
output, at the cost of at least two passes before anything is shown and a
noticeably later first subtitle. Worth revisiting if the changing text proves
distracting in rehearsal — the measurement to make then is how often a partial
is revised, not how it feels.

**Speaking partials.** Rejected above, and it stays rejected.

## Revisit if

- A rehearsal shows the changing subtitle is harder to read than a later stable
  one.
- The operator interface renders subtitles somewhere that cannot update in
  place.
- Partial recognition is observed delaying committed sentences, which
  `partials_skipped_busy` would show.
