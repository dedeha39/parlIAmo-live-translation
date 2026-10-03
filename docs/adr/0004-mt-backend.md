# ADR 0004 — NLLB for translation, not an LLM

**Status:** accepted · 2026-08-30

## Context

The plan assumed an instruction-following LLM would translate better than a
dedicated NMT model, and budgeted ~3100 MiB for one. That assumption was never
tested. The question raised was simply: *why would we use an LLM at all?*

Answering it properly needed the eliminated models actually run rather than
estimated, so `mt/ollama_backend.py` was added to serve GGUF models over HTTP —
building `llama-cpp-python` with CUDA on Windows failed on a missing shared
library, and Ollama was already installed.

## Measured

`google/translategemma-4b-it`, Q8_0 GGUF, fully GPU-resident, greedy decoding,
against six Turkish sentences from the presentation script and the FLEURS set.

| | NLLB-200-600M | TranslateGemma-4B Q8 |
|---|---|---|
| VRAM | **844–929 MiB** | **5240 MiB** |
| Latency, batched | **42 ms/sentence** | not applicable |
| Latency, single | 487 ms p95 | **2671 ms median** |
| Throughput | — | ~45 tok/s |
| Target language | always correct | **wrong on 2 of 6** |
| chrF++ (tr→it, 200 FLEURS) | **45.38** | not scorable |

### Latency rules it out on its own

2671 ms median per sentence, against a total end-to-end budget of 2–3 seconds
that already contains a 700 ms commit wait, ~300 ms of recognition and
synthesis on top. Translation alone would consume the entire budget.

Q4_K_M would roughly halve it to ~1.3 s, which is still more than the whole
remaining budget. The arithmetic is structural, not a tuning problem:
autoregressive generation at 45 tok/s cannot produce a 30-token sentence in
under ~0.7 s even before prompt processing. NLLB's encoder-decoder does it in
42 ms batched because it is not generating token by token against a chat
template.

### Quality was not merely worse — it was unusable

Four distinct failures in six sentences:

| Failure | Example |
|---|---|
| Wrong target language | `Do not deface or graffiti on buildings.` — English, asked for Italian |
| Language leakage | `…basta quindici secondi.\nes: No se nec` — drifted into Spanish |
| Wrong domain vocabulary | `un lungo brano per clonare un suono` — "brano" is a piece of music, "suono" a sound; should be *registrazione* and *voce* |
| **Dropped negation** | source: "teknik bir mesele **değil**, etik bir mesele" → output: "è una questione tecnica, ma soprattutto etica" |

The last is the one that matters. The source says *not* a technical matter; the
translation says it *is* one. On stage that tells the audience the opposite of
what was said, fluently and with no sign anything went wrong.

No chrF++ score is reported because the outputs cannot be scored honestly — a
corpus where some sentences are in the wrong language is not measuring
translation quality.

## Decision

**NLLB-200-distilled-600M via CTranslate2 for the live path.**

## Fairness caveats, stated rather than buried

* **Correction, 2026-09-23: the latency above carries ~2 s that was not the
  model's.** `OllamaBackend` addressed the server as `localhost`, which Windows
  resolves to IPv6 first; Ollama listens on IPv4, and every request waited ~2 s
  for the failed attempt. Measured on HY-MT1.5: 2274 ms per sentence through
  `localhost`, 222 ms through `127.0.0.1`, for 230 ms of model time. Re-run
  through `127.0.0.1`, TranslateGemma with this ADR's prompt ran on past the
  sentence into other languages until the token limit (12 s), translated one
  sentence into English and turned "fifteen seconds" into *un minuto*. The
  decision stands on quality and memory; the 2671 ms does not describe the
  model.

* The prompt format may still be imperfect. Three were tried — chat with an
  instruction, chat with `tr:`/`it:` codes, and generate with stop tokens — and
  only the third produced single-line output at all. A better template might
  fix the language-selection failures. **It would not fix the latency**, which
  is what rules the approach out.
* Q8_0 was tested because that was the GGUF pulled. Q4_K_M would be about twice
  as fast and half the memory, and still too slow.
* This is a verdict on *LLM translation in the live path on this hardware*, not
  on TranslateGemma as a model. On a larger card, or offline, it may well beat
  NLLB.

## Where an LLM still earns its place

Nothing above applies without a latency constraint. An LLM remains the right
tool for **offline** work in this project:

* pre-translating the prepared Friulian demonstration sentences, where quality
  matters and seconds do not;
* generating or checking synthetic training data for the Friulian fine-tune;
* building the terminology glossary.

The `mt/ollama_backend.py` implementation stays for exactly those uses, and it
earned its keep in another way: it is the second implementation behind
`MTBackend`, which is what makes that interface a real abstraction rather than
a guess about what varies.

## What this vindicates

The `sanitise` layer in `mt/base.py` was written defensively, against failures
that had not been observed. Every one of them then happened: preambles,
continuation past the answer, drift into other languages. It is load-bearing.

## Still open

NLLB-600M's own quality is mediocre — chrF++ 45.38, with real errors including
`Takımadalarda` (in the archipelagos) rendered as `nelle scuole` (in the
schools). The next thing to measure is **NLLB-1.3B**, which stays inside the
NMT family and therefore inside the latency budget.
