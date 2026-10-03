# ADR 0007 — Keep NLLB-600M, and split sentences before translating

**Status:** accepted · 2026-08-31
**Closes:** the "NLLB-1.3B is converted but unmeasured" item open since ADR 0004.

## What prompted this

The pipeline was replayed end to end over five sentences of the presenter's own
recorded voice — the first time the whole chain had run on real speech rather
than one stage at a time. Four translations were fine. One was not:

```
tr   Kusurumu hazırlarken aklımdan getiren ilk soru şuydu.
     Acaba gerçekten çalışacak mı?
it   Mentre preparavo il mio errore, la prima domanda che mi è venuta in
     mente è stata:
```

The second sentence — *I wonder whether it will really work?* — is **gone**.

That is the worst shape a failure can take here. The Italian is fluent and
grammatical, `sanitise` finds nothing wrong with it, the length ratio is
unremarkable, and no error is logged. The audience simply never hears a
sentence that was said, and nobody on stage can tell.

## Diagnosis

Isolated with seven probe sentences against NLLB-600M:

| Input | Output |
|---|---|
| `…ilk soru şuydu. Acaba gerçekten çalışacak mı?` | `…è stata:` — **second sentence dropped** |
| `…ilk soru şuydu: acaba gerçekten çalışacak mı?` | `…è stata: funzionerà davvero?` — **complete** |
| `Acaba gerçekten çalışacak mı?` alone | `Mi chiedo se funzionerà davvero.` — fine |
| `Birinci cümle budur. İkinci cümle şudur. Üçüncü…` | all three translated |

So it is not "only translates the first sentence". The trigger is the
**cataphoric** `şuydu` / `şudur` construction — "was this:" — which points
*forward* at what follows. NLLB renders it as `è stata:` and stops, discarding
what the colon promised. Written with an actual colon instead of a full stop,
the same content survives.

`şuydu`, `şudur`, `şöyle` are ordinary presentation Turkish. This is not an
exotic input; it is the register the talk is written in.

## Measured: does NLLB-1.3B fix it?

Yes. Same probes, same settings:

| Case | 600M | 1.3B |
|---|---|---|
| cataphoric + question | drops the second sentence | `…è stata: funzionerà davvero?` |
| `Dışarıda hava kapalı` (it is overcast) | `L'aria è chiusa` (the air is closed) | `Fuori è buio` (it is dark outside) |
| `Takımadalarda` (in the archipelagos) | `nelle squadre` (in the teams) | `nei gruppi` (in the groups) |
| `değil` negation | correct | correct |

And on the same 200 FLEURS `tr→it` pairs the 600M figure was measured on:

| | NLLB-600M | NLLB-1.3B | Δ |
|---|---|---|---|
| chrF++ | 45.38 | **47.11** | +1.73 |
| BLEU | 18.91 | **21.34** | +2.43 |
| batch ms/sentence | **55.1** | 107.8 | ×1.96 |
| single-sentence p95 | **606 ms** | 846 ms | +240 ms |
| VRAM, loaded alone | **751 MiB** | 1713 MiB | +962 |

1.3B is better by every quality measure. It is still rejected for the live path,
on memory.

## Why 1.3B does not ship

Measured co-resident — ASR turbo, MT, Kokoro, and the Seed-VC service running in
its own process — on the 8188 MiB card:

| MT | Peak | Headroom |
|---|---|---|
| NLLB-600M | 7292 MiB | **896 MiB** |
| NLLB-1.3B | 7829 MiB | **359 MiB** |

The two runs had slightly different desktop baselines (4133 and 4485 MiB), so
compare the MT line itself: **+1118 MiB co-resident**, which puts the honest
1.3B peak near 7.5 GB and the headroom near 700 MiB.

ADR 0003 already established what to do with a margin that size. The desktop
baseline alone measured 1476, 1565 and 1857 MiB across three runs minutes apart
— a **381 MiB spread caused by nothing but browser tabs**. A 359–700 MiB margin
sits inside that spread. It would work on a quiet machine and fail on a busy
one, which ADR 0003 named as the worst kind of "works". That reasoning
eliminated large-v3; it eliminates 1.3B for the same reason.

### The CPU escape hatch does not open

ADR 0003 offered one lever: move translation to the CPU and the whole MT line
leaves the budget. Measured, 1.3B on the 13900HX:

```
p95 3023 ms per sentence
```

Three seconds for translation alone, against a 2–3 s budget for the *entire*
pipeline. That is worse than the 2671 ms that disqualified TranslateGemma in
ADR 0004. The lever exists for 600M (≈100–150 ms extra); it does not for 1.3B.

*(The CPU run scored chrF++ 45.86 on 60 pairs rather than 200 — a smaller
sample, not a quantisation effect. It is quoted for latency only.)*

## Decision

1. **NLLB-600M stays on the live path.**
2. **Sentences are split before translation** — `mt.split_sentences: true`.
3. **NLLB-1.3B is kept for offline work**, where seconds do not matter: the
   prepared Friulian demonstration sentences, pre-translated slide text, and
   checking synthetic training data. Same conclusion ADR 0004 reached about
   LLMs, and for the same reason.

## Why splitting is the right fix, and not a workaround

Each sentence translated separately, then joined:

| | whole segment | split |
|---|---|---|
| cataphoric case | second sentence dropped | **recovered** |
| median latency, probe set | 408 ms | **355 ms** |
| batch ms/sentence, 200 FLEURS | 55.1 | **47.4** |
| chrF++, 200 FLEURS | 45.38 | **45.38** |

It **recovers the lost content and is faster**, because a batch of short decodes
beats one long autoregressive decode. And chrF++ is unchanged to two decimal
places on FLEURS, because FLEURS sentences are single sentences — the splitter
returns them whole and does nothing. That is the ideal shape for a fix: it
costs nothing where the bug is absent and only acts where it is present.

`CTranslate2NLLBBackend` overrides `_translate_pieces` to send one batch rather
than one call per sentence, which is what keeps it free.

### Splitting Turkish is not `text.split(".")`

Two constructions break a naive splitter, and both appear in this talk:

* **Ordinals** are written with a full stop — `1. Dünya Savaşı`, `3. sınıf`.
* **Abbreviations** — `Dr.`, `Prof.`, `vb.`, `M.Ö.`

Both are vetoed explicitly in `parliamo/mt/sentences.py`, with tests. Thousands
separators (`40.000`) are safe without special handling because they carry no
following space — the same Turkish digit convention that once corrupted the ASR
reference text (BUILD_LOG, 2026-08-30).

A colon is deliberately **not** a boundary: measured, splitting there made the
cataphoric case slightly worse, because the colon carries the link between the
halves. Only `.`, `!`, `?` and `…` split.

## What this does not fix

600M's ordinary quality is untouched. `Dışarıda hava kapalı` still becomes
`L'aria è chiusa` — "the air is closed" — losing both the idiom and the word
*outside*. `Takımadalarda` still becomes `nelle squadre`. These are the same
class as the error recorded in ADR 0004 and they remain the strongest argument
for revisiting this decision.

## Revisit if

- A card with more than 8 GB is used, in which case 1.3B is simply better.
- Seed-VC is dropped from the live path (2.9 GB), which would free enough room.
- A distilled or quantised 1.3B lands within ~400 MiB of 600M.
- The in-situ replay shows content errors that splitting does not address.
