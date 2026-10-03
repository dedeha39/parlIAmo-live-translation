# ADR 0009 — The commit wait is 500 ms, and 700 was making things worse

**Status:** accepted · 2026-09-01
**Corrects:** the assumption stated in ADR 0005, 0006 and 0008 that lowering
`vad.min_silence_ms` trades accuracy for latency.

## The assumption

`vad.min_silence_ms` is how long the segmenter waits in silence before deciding
a sentence has ended. It is the largest single term in the end-to-end delay, and
every document in this project has described the trade the same way:

> Lowering `vad.min_silence_ms` from 700 to 500 takes the total to 2.20 s at the
> cost of cutting speakers off mid-clause more often.

That sentence was never measured. It could not be: the 40 in-situ recordings are
one sentence each, and the pauses between them were inserted afterwards by hand.
Every threshold below the inserted gap segments identically, and every threshold
above it fails identically. The material could not distinguish 400 ms from
900 ms, so the number stayed at its initial guess for a fortnight.

## The measurement

68 seconds of continuous reading, phone recording, natural breath pauses.

| `min_silence_ms` | segments | mean | longest | **truncated** | fragments <1.5 s | WER | CER | ASR p95 | worst wait |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 14 | 4.3 s | 9.2 s | 0 | **2** | — | — | — | 9.5 s |
| 400 | 11 | 5.5 s | 10.1 s | 0 | 0 | **11.41** | **3.61** | 0.465 | 10.5 s |
| **500** | **11** | **5.5 s** | **10.1 s** | **0** | **0** | **11.41** | **3.61** | **0.456** | **10.6 s** |
| 600 | 8 | 7.6 s | 13.5 s | 0 | 0 | 12.08 | 3.71 | 0.500 | 14.1 s |
| 700 *(old default)* | 7 | 8.8 s | **15.0 s** | **1** | 0 | 12.08 | 3.90 | 0.545 | 15.7 s |
| 900 | 6 | 10.7 s | 15.0 s | **4** | 0 | 12.08 | 3.71 | 0.500 | 15.9 s |
| 1200 | 6 | 10.7 s | 15.0 s | **4** | 0 | — | — | — | 16.2 s |

**The assumption was backwards.** 500 ms is better than 700 on every axis at
once: 0.67 WER points lower, 0.29 CER points lower, 89 ms faster recognition,
no truncated segments, and **five seconds off the worst case**. There is no
trade to make.

## Why a longer wait was *worse*

At 700 ms the segmenter was not splitting at the speaker's natural pauses. It
ran on until `max_segment_ms` — a 15-second hard cut that exists so one endless
sentence cannot stall the pipeline — and that cut lands wherever the clock says,
which is mid-phrase:

```
... sonra bir sesin ne kadar hızlı
hızlı kopyalanabildiğini en sonunda da ...
```

The recogniser is then handed half a phrase with no ending, and the translator
is handed whatever it made of it. That is where the extra WER came from.

At 900 ms and above, four of six segments hit the hard cut. The threshold had
stopped being a sentence boundary detector at all.

## Decision

**`vad.min_silence_ms: 500`.**

400 ms measures identically — this speaker's pauses are either under 400 ms or
over 500, with nothing in between — so 500 is taken for the margin. 300 ms
begins producing fragments under 1.5 seconds, which is the failure the original
assumption feared, two steps further down than anyone had looked.

## The number this exposes, which is much larger

Read the last column again. Even at the best setting, the longest segment is
**10.1 seconds**, so the audience waits **10.6 s** for that sentence — against
an end-to-end budget of 2–3 s.

That budget was always computed for a *typical 4.4 s sentence*. Real continuous
speech does not produce a stream of 4.4-second sentences; it produces 5.5 on
average and 10.1 at the tail. **The dominant term in what the audience actually
waits is how long the speaker talks without pausing, not anything the pipeline
does.**

Two consequences:

1. **Partial subtitles (ADR 0008) matter more than they appeared to.** They put
   text on screen 1.68 s in, *regardless* of how long the segment runs on. On a
   10-second segment that is nine seconds earlier, not the 2.8–6.8 s estimated
   from typical sentences.
2. ~~Splitting long segments at clause boundaries is now a requirement.~~
   **Tried, and it does not work.** See below.

## What made this measurable

One 68-second phone recording. The tooling to answer the question had existed
for two weeks; what was missing was material with real pauses in it. A
measurement harness is only as good as what you feed it, and 40 sentences with
hand-inserted gaps were the wrong shape for this question in a way that was not
obvious until the right shape existed.

## Revisit if

- A different speaker presents, or this one changes pace under stage nerves.
  Re-run `scripts/live_translate.py --file` on a rehearsal recording; the table
  above is one speaker on one day.
- Clause splitting lands, which changes what `max_segment_ms` is protecting
  against.


---

## Addendum, same day — the tail cannot be segmented away

The consequence above said cutting long segments at a shorter pause was "a
requirement, not an optimisation". It was implemented within the hour and
measured, and the measurement says no.

`soft_cut_after_ms` closes an utterance at a brief pause once it has already run
long, instead of waiting for the full `min_silence_ms`. On the same recording:

| setting | segments | mean | longest | fragments <1.5 s | WER | worst wait |
|---|---|---|---|---|---|---|
| **off** | 11 | 5.5 s | 10.1 s | **0** | **11.41** | 10.6 s |
| 8000 / 240 ms | 12 | 5.0 s | 9.2 s | 1 | 12.08 | 9.7 s |
| 6000 / 240 ms | 12 | 5.0 s | 9.2 s | 1 | 12.08 | 9.7 s |
| 4000 / 240 ms | 12 | 5.0 s | 9.2 s | 1 | 12.08 | 9.7 s |

Nine tenths of a second off the worst case, bought with 0.67 WER points and a
fragment under a second and a half. Not a good trade.

**And the longest segment is 9.2 s at every setting, including the most
aggressive.** Firing after four seconds instead of eight changes nothing about
it. The reason is simple once seen: that stretch of speech *contains no pause
at all*, not even a 240 ms one. The speaker did not breathe.

So the tail is not the segmenter being too patient, and no segmentation policy
will fix it. A translation cannot be produced for a sentence that has not
finished, and this one takes nine seconds to finish.

That leaves two honest positions:

* **The audio tail is inherent.** Human simultaneous interpreters have exactly
  this problem, and solve it by starting before the sentence ends — which this
  project has rejected for Turkish, because the verb and its negation arrive
  last (ADR 0008).
* **The text tail is already solved.** Partial subtitles put text on screen
  1.68 s in whatever the segment does. On this 9.2 s segment that is seven and
  a half seconds before the audio.

`soft_cut_after_ms` is kept and defaulted to 0. A rehearsal under stage nerves
is not this recording, and a speaker who pauses differently may make it worth
switching on — but it goes on with a measurement, not on the reasoning that
produced it here, which was wrong.
