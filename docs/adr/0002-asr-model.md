# ADR 0002 — Use Whisper large-v3-turbo for Turkish recognition

**Status:** accepted · 2026-08-30
**Supersedes:** nothing. **Revisit after:** the in-situ recording test.

## Context

Four candidates were run over 200 FLEURS Turkish test utterances (41.9 minutes)
on the RTX 4070 Laptop, each loaded alone, warmed up, then unloaded, at
`int8_float16` with greedy decoding.

| Model | WER % | CER % | RTF | VRAM MiB | Load s |
|---|---|---|---|---|---|
| large-v3 | **5.59** | **1.31** | 0.075 | 1959 | 269.8 |
| **large-v3-turbo** | 6.15 | 2.02 | **0.024** | **950** | 167.6 |
| medium | 8.27 | 2.07 | 0.053 | 952 | 131.1 |
| small | 14.59 | 3.57 | 0.023 | 371 | 54.3 |

WER is after Turkish-aware normalisation. No model failed an utterance.

## Decision

**`large-v3-turbo`**, at `int8_float16` on CUDA.

## Reasoning

### VRAM decides it, not accuracy

large-v3 is 0.56 WER points better. It also costs **1009 MiB more**, and the
budget does not have that. Measured idle VRAM on this machine is ~1700 MiB with
a normal desktop session, leaving ~6490 MiB:

| | large-v3-turbo | large-v3 |
|---|---|---|
| ASR | 950 | 1959 |
| MT (4B, Q4_K_M + KV) | ~3100 | ~3100 |
| TTS (Chatterbox + vocoder) | ~2000 | ~2000 |
| **Total** | **6050** | **7059** |
| **Margin against 6490** | **+440** | **−569** |

large-v3 does not fit alongside translation and synthesis. The choice was made
by the hardware.

### Speed matters more than the last half point

RTF 0.024 against 0.075 — **3.1× faster**. All three stages share one GPU, so
ASR time is time the translation and synthesis stages do not get. Buying
0.56 WER points for triple the compute is a bad trade in a 2–3 second budget.

### medium is strictly dominated

Worse WER than turbo (8.27 vs 6.15), twice the RTF (0.053 vs 0.024), and the
same VRAM (952 vs 950). There is no configuration in which it is the right
choice, so it is dropped from consideration entirely rather than kept as a
fallback.

### turbo's errors are the more recoverable kind

Error composition over 3578 reference words:

| Model | Substitutions | Deletions | Insertions |
|---|---|---|---|
| large-v3 | 156 | 11 | 33 |
| large-v3-turbo | **146** | 23 | 51 |

turbo actually *mishears fewer words* than large-v3; its deficit is entirely in
boundary handling — dropped and added words around segment edges. That is the
better failure mode here, because a translation model with sentence context can
often absorb a missing function word, while a substituted content word silently
changes what the audience is told.

## Caveats that must not be forgotten

**FLEURS is clean read speech.** These numbers rank models; they do not predict
accuracy in a conference room, through a lavalier microphone, from a speaker
who is presenting rather than reading. The reference laptop's own capture path
turned out to be a 16 kHz DSP pipeline with aggressive noise gating, which is
exactly the sort of thing this test cannot see. An in-situ recording is still
required before the number is quoted anywhere.

**Normalisation is worth ~10.6 points.** Raw WER runs 10.0–10.8 points higher
than normalised WER across all four models — that difference is punctuation,
casing and digit formatting, not recognition. Comparing raw figures against
published benchmarks would be meaningless.

## Consequences

- `config/default.yaml` keeps `asr.model: large-v3-turbo`.
- The ASR line of the VRAM budget drops from the planned 1600 MiB to **950 MiB**,
  giving 650 MiB back to the rest of the pipeline.
- large-v3 stays available behind a config change for offline or batch work,
  where the VRAM constraint does not apply.

## Revisit if

- The in-situ WER for turbo is materially worse than for large-v3, in which case
  the extra VRAM has to come from somewhere else (CPU translation is the first
  lever).
- A streaming-native recogniser becomes usable on Windows without vLLM.

---

## Amendment, 2026-08-30 — the in-situ measurement

40 sentences read by the presenter into the laptop microphone, 4.28 minutes.
The script was read aloud, so the script is the reference and no transcription
was involved.

| Model | hotwords | WER % | CER % | RTF | VRAM MiB |
|---|---|---|---|---|---|
| large-v3 | no | **12.78** | **4.65** | 0.094 | 1894 |
| large-v3 | yes | 13.76 | 5.89 | 0.084 | 1903 |
| large-v3-turbo | **yes** | **14.50** | 4.94 | **0.038** | **1011** |
| large-v3-turbo | no | 16.71 | 5.45 | 0.039 | 1084 |

### FLEURS ranked the models; it did not predict the gap

| | FLEURS | In situ | Change |
|---|---|---|---|
| large-v3 | 5.59 | 12.78 | +7.2 |
| large-v3-turbo | 6.15 | 16.71 | +10.6 |
| **Gap between them** | **0.56** | **3.93** | **7× wider** |

On clean read speech the two are nearly equivalent. On a real voice through a
real microphone they are not. Choosing on public benchmarks alone would have
been choosing on a difference that does not hold.

### Hotwords help the weaker model and hurt the stronger one

Biasing decoding toward the names that fail hardest (`Friulian`, `ARLeF`,
`Gorizia`, `Chatterbox`) moved turbo by **−2.21 WER points** and large-v3 by
**+0.98** — it made the better model worse.

large-v3 already handles these words tolerably, so the bias buys nothing and
costs something: it pulls ordinary words toward the list. That dilution risk is
noted in `config/hotwords.tr.txt` itself, and this is it happening. The list
must be re-measured whenever it changes, never extended on the assumption that
more context is better.

### The decision stands, with the gap now much smaller

Hotwords narrow the distance from 3.93 to **1.72 points**, and the VRAM
arithmetic is unchanged:

| | turbo + hotwords | large-v3 |
|---|---|---|
| ASR | 1011 | 1894 |
| MT (4B Q4 + KV, estimated) | ~3100 | ~3100 |
| TTS (estimated) | ~2000 | ~2000 |
| **Total** | **6111** | **6994** |
| **Margin against ~6490** | **+379** | **−504** |

**`large-v3-turbo` with hotwords**, at 14.50% WER.

### One live option, deliberately not taken yet

Moving translation to CPU frees ~3.1 GB and makes large-v3 fit comfortably
(1894 + 2000 = 3894 MiB), buying 1.72 WER points for roughly 100–150 ms of
added translation latency. On the numbers so far that looks like a good trade.

It is not being taken now because the MT figures in the table above are
**estimates**. Deciding between two measured options using a third unmeasured
one is how the FLEURS-only choice would have gone wrong. Revisit once Phase 2
has measured translation VRAM and CPU throughput for real.

### Decoding sweep, same 40 in-situ sentences

| Variant | WER % | CER % | RTF | p95 s | VRAM MiB |
|---|---|---|---|---|---|
| large-v3, greedy, no hotwords | **12.78** | 4.65 | 0.089 | **0.724** | 1888 |
| turbo, beam 5, float16 | 13.76 | 4.76 | 0.046 | 0.322 | 1984 |
| large-v3, greedy, hotwords | 13.76 | 5.89 | 0.087 | 0.657 | 1882 |
| **turbo, beam 5, int8** | **14.00** | 4.87 | 0.042 | 0.293 | **1032** |
| turbo, greedy, int8 *(previous default)* | 14.50 | 4.94 | 0.037 | 0.272 | 1134 |
| turbo, temperature fallback | 14.50 | 4.94 | 0.041 | 0.399 | 1024 |
| turbo, greedy, float16 | 14.99 | 5.05 | 0.041 | 0.282 | 1992 |
| turbo, greedy, no hotwords | 16.71 | 5.45 | 0.035 | 0.248 | 1024 |

**Beam 5 adopted.** int8 14.50 → 14.00, float16 14.99 → 13.76. Either gap alone
sits near this sample's noise floor (407 reference words, so ~0.15 points per
word), but both quantisations move the same way, and the cost is 21 ms on p95
against a 700 ms commit wait. Beam search stays deterministic; temperature
fallback would not.

**float16 rejected.** At greedy it is *worse* than int8 (14.99 vs 14.50) and
costs 858 MiB. At beam 5 it is better by 0.24 points — about one word — for
952 MiB. Not a trade worth making in this budget.

**Temperature fallback rejected, and now on evidence rather than reasoning.**
Identical WER to greedy (14.50), but p95 latency rose 0.272 → 0.399 s, +47%.
It bought nothing and cost predictability, which is what it was originally
excluded for.

**Latency argues for turbo more strongly than VRAM does.** large-v3's p95 is
**724 ms**, against turbo's 293 ms. Added to a 700 ms commit wait, large-v3
roughly doubles the delay the audience experiences. That was not visible in the
FLEURS run, where mean RTF hid the tail.

### Turkish fine-tuned Whisper: measured, and clearly worse

`selimc/whisper-large-v3-turbo-turkish`, converted to CTranslate2 and run on the
same 40 sentences:

| Variant | WER % | CER % |
|---|---|---|
| stock large-v3 | **12.78** | 4.65 |
| stock turbo, beam 5, hotwords | 14.00 | 4.87 |
| tr-finetuned turbo, hotwords | 19.16 | 6.11 |
| tr-finetuned turbo, greedy | 19.66 | 6.30 |
| tr-finetuned turbo, no hotwords | 19.90 | 6.84 |

**+5.16 points worse than stock turbo** — 21 words out of 407, far outside this
sample's noise. Not close.

This is catastrophic forgetting behaving exactly as the literature describes.
The fine-tune was trained on Common Voice: short read sentences. Our audio is
5–8 second presentation sentences delivered at speaking pace. Gains on the
training distribution were bought with losses everywhere else, and everywhere
else is where we live.

The general lesson, which will apply again in Phase 2: **"fine-tuned for our
language" is a claim about a training set, not about our audio.** A published
WER (18.92 on Common Voice 17) says nothing about a different test set, and
here it happened to be roughly right by coincidence rather than by transfer.

### Caveat on the VRAM column

VRAM is measured as an `nvidia-smi` delta across model load, so it includes any
other process that happened to allocate during the same window. The same model
measured 950, 1011, 1084 and 1292 MiB across runs. Treat these as ±150 MiB, and
size the budget from the upper end.
