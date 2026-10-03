# ADR 0003 — The VRAM budget, measured

**Status:** accepted · 2026-08-30
**Corrects:** the estimated budget used in ADR 0002 and in the project plan.

## Context

The budget was assembled from per-model estimates and then used to eliminate
candidates. Both halves of that were mistakes: the estimates were wrong, and
summing separately-estimated models is not the same as measuring them together.

`scripts/measure_vram_budget.py` loads the stages into one process in sequence
and reports the real total after each addition, taking the peak of several
`nvidia-smi` samples rather than a single noisy reading.

## Measured

RTX 4070 Laptop, 8188 MiB total.

| Stage | Estimated | **Measured** | Error |
|---|---|---|---|
| ASR — large-v3-turbo int8 | ~1030 | **1223** | +19% |
| ASR — large-v3 int8 | ~1894 | **2113–2181** | +12% |
| MT — NLLB-600M on GPU | ~3100 | **844–929** | **−70%** |
| MT — NLLB-600M on CPU | — | **0** | |
| TTS — Chatterbox Multilingual v3 | ~2000 | **2824–3368** | **+68%** |
| TTS — generation activations | (not budgeted) | **42–284** | |
| Desktop session | ~1400 | **1476–1857** | varies |

Two estimates were wrong in **opposite directions** and partly cancelled, which
made the total look roughly right by luck. Translation was overestimated more
than three-fold; synthesis was underestimated by two-thirds and is in fact the
**largest single consumer in the pipeline**, not the smallest.

## Full-pipeline peaks

| Configuration | Peak MiB | Headroom |
|---|---|---|
| turbo + NLLB(GPU) + TTS | 7230 | 958 |
| large-v3 + NLLB(GPU) + TTS | 7751 | 437 |
| large-v3 + NLLB(CPU) + TTS | **7034** | **1154** |

## Consequence: ADR 0002's reasoning was wrong

large-v3 was eliminated on this arithmetic:

```
1894 (ASR) + 3100 (MT) + 2000 (TTS) = 6994  >  6490 available   ->  does not fit
```

Two of those three numbers were wrong. Measured, it fits with 437 MiB to spare,
and with translation on the CPU it fits with 1154 MiB.

**The conclusion survives; the reasoning does not.** turbo remains the choice,
but for the reason the sweep found rather than the reason originally given:
large-v3's p95 recognition latency is **724 ms against turbo's 293 ms**, and
that 431 ms lands on every single segment on top of a 700 ms commit wait. The
accuracy it buys is 1.22 WER points — about five words in 407, close to this
sample's noise floor.

So: **not "large-v3 does not fit" but "large-v3 is too slow for what it buys".**

## The headroom is thinner than one number suggests

The desktop baseline measured 1476, 1565 and 1857 MiB across three runs
minutes apart — a 381 MiB spread caused by nothing more than browser tabs. With
large-v3 and GPU translation the margin is 437 MiB, which is inside that spread.
That configuration would work on a quiet machine and fail on a busy one, which
is the worst kind of "works".

Moving translation to the CPU costs roughly 100–150 ms per sentence and buys
1154 MiB of margin — enough to be insensitive to what else is open. It is the
first lever to pull if anything else needs to grow.

## Rules this establishes

1. **No component enters the budget as an estimate.** Anything unmeasured is
   named as unmeasured, and a decision that depends on it waits.
2. **Measure co-resident, not summed.** CUDA context, allocator fragmentation
   and cuBLAS workspaces are paid once, not once per model.
3. **Budget against the *peak* desktop baseline**, not a lucky reading. Use
   ~1900 MiB, not 1476.
4. **An out-of-memory during measurement is a result, not a failure.** It is the
   answer to whether a configuration fits.

## Still open

Chatterbox generation latency has not been measured yet, only its memory. Until
it is, the end-to-end latency budget still contains an estimate, and the
turbo-versus-large-v3 question cannot be closed properly.
