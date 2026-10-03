# ADR 0005 — Synthesis is the bottleneck, and streaming is not optional

**Status:** accepted · 2026-08-30
**Supersedes the latency reasoning in:** ADR 0002, ADR 0003, ADR 0004.

## The measurement

Chatterbox Multilingual v3, Italian, on the RTX 4070 Laptop. Four sentences
spanning the lengths that will actually occur, three repeats each.

| Characters | Audio produced | Generation | RTF |
|---|---|---|---|
| 11 | 2.68 s | 4.31 s | 1.61 |
| 65 | 4.28 s | 5.63 s | 1.32 |
| 85 | 4.64 s | 6.10 s | 1.31 |
| 154 | 8.16 s | 10.54 s | 1.29 |

An earlier run on a less loaded machine gave RTF 0.94–1.13 and a median of
4.49 s. The spread is machine load — 31 browser processes and the desktop still
composited on the NVIDIA GPU — not model variance. **Take RTF ≈ 1.0–1.6 and
4.5–6.1 s per sentence as the range.**

## What that does to the budget

```
commit wait    0.70 s   (policy: silence before a sentence is committed)
recognition    0.29 s   (large-v3-turbo p95)
translation    0.49 s   (NLLB-600M p95, single sentence)
synthesis      4.49–6.10 s
                ------
total          5.97–7.58 s
```

The target was 2–3 seconds. **The pipeline as designed is two to three times
over budget, and synthesis is the entire reason.**

## This reorders every previous decision

Three ADRs weighed recognition latency carefully:

* ADR 0002 chose turbo over large-v3 partly on 293 ms versus 724 ms.
* ADR 0003 revisited the same trade after correcting the VRAM arithmetic.
* ADR 0004 ruled out LLM translation at 2671 ms per sentence.

All three were arguing over hundreds of milliseconds while a 4500–6100 ms stage
sat downstream, unmeasured. **The 431 ms separating the two recognisers is
7–10% of the synthesis stage.** That question was never the important one, and
it looked important only because the expensive stage had not been measured.

The lesson is the one this project keeps relearning: an unmeasured component
does not merely leave a gap in the budget, it distorts the decisions made
around it.

The decisions themselves stand — turbo is still the right recogniser, NLLB is
still the right translator — but the *reasons* were disproportionate.

## Why RTF ≈ 1.0 is the key number, not the 6 seconds

RTF near 1.0 means synthesis produces audio at about the rate audio is
consumed. That is a bad property for batch generation and a **good** one for
streaming.

* **Not streaming** (current): nothing is audible until the whole utterance is
  generated. The audience waits the full 4.5–6.1 s. Unusable.
* **Streaming**: the first chunk arrives after roughly a second, and generation
  then keeps approximate pace with playback. The audience waits for the first
  chunk, not the whole sentence.

With streaming the budget becomes::

```
0.70 commit + 0.29 asr + 0.49 mt + ~0.8 first chunk  ≈  2.3 s
```

which is inside the target.

**Streaming synthesis is therefore not an optimisation. It is the difference
between a system that works and one that does not**, and it is now the single
highest-priority item in the project.

## Consequences

1. Streaming synthesis moves to the top of the plan, ahead of the Friulian
   branch and ahead of the operator interface.
2. `TTSBackend` already separates `first_audio_s` from `compute_s`; today they
   are equal because Chatterbox generates whole utterances. A streaming backend
   makes that distinction real, and the interface is ready for it.
3. At RTF up to 1.6 a long sentence still loses ground during playback. The
   pauses between sentences are what let it catch up, so the segmenter's
   `max_segment_ms` interacts with this directly and must be re-tuned once
   streaming exists.
4. If streaming cannot be made to work, the fallbacks are, in order of
   preference: a faster model (Kokoro is far quicker but has no voice cloning,
   which would cost the presentation its central demonstration), or accepting
   consecutive rather than simultaneous delivery.

## Risk on the way to fixing it

`chatterbox-streaming` is a third-party fork. Installing `chatterbox-tts`
already downgraded torch from a CUDA build to the CPU wheel and silently broke
every other stage; the fork can be expected to pin just as aggressively.
Install it in an isolated environment first, and verify `torch.cuda.is_available()`
immediately afterwards.

## Measurement hygiene note

The first attempt to re-measure was run while a model conversion was still
using the machine, and reported RTF 2.57–3.54 — nearly triple. Numbers taken
under competing load are not comparable with numbers taken without it. Check
for competing processes before measuring, and record the machine state
alongside the result.
