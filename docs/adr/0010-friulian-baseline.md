# ADR 0010 — Friulian needs a speaker, not a fine-tune

**Status:** accepted · 2026-09-01
**Corrects:** the Friulian plan carried since the project outline.

## The plan that was never checked

Every document in this project has described Friulian as a data problem:

> **Friulian branch.** Nothing done. Data situation: ~1000 natural sentence
> pairs on OPUS, plus FLORES-200 `fur_Latn` (997 dev + 1012 devtest), La Bibie
> 2019 aligned to the Italian CEI Bible (~31k verses), and `fur.wikipedia` for
> back-translation.

That is a plan for **fine-tuning NLLB**: collect a corpus, train, evaluate.
Weeks of work, and the largest untouched item on the board.

Nobody had run the model that was already on disk.

## Measured

NLLB-200-distilled-600M, no fine-tuning, `ita_Latn` to `fur_Latn`:

| Italian | Friulian |
|---|---|
| Questo sistema funziona interamente su questo computer portatile. | **Chest sisteme al funzione dut su chest laptop.** |
| Il friulano è una lingua minoritaria... | **Il furlan al è une lenghe minoritarie...** |
| Non è un problema tecnico ma una scelta collettiva. | **Nol è un probleme tecnic ma une opzion coletive.** |
| Le lingue minoritarie restano indietro... | **Lis lenghis minoritariis a restin...** |

This is not Italian passed through. The markers are the right ones:

* **`al è` / `nol è`** — the Friulian verbal clitic, and its contraction with
  the negative. Italian has no such thing.
* **`Lis lenghis minoritariis`** — Friulian feminine plural in `-is`, against
  Italian `-e`.
* **`chest`, `dut`, `ducj`, `jessi`, `cj`, circumflexes** — Friulian
  vocabulary and orthography.

None of six outputs was identical to its Italian input; character similarity
ran 0.56–0.86. An `it → fur → it` round trip returned the meaning intact at
0.75–0.95.

### Errors are real but ordinary

`parlata` (spoken) came out as `parade`, which looks like a false friend for
`fevelade`. `scelta` (choice) became `opzion` rather than `sielte`. These are
the same class of error NLLB-600M makes in Italian — `Takımadalarda` rendered
as `nelle scuole` (ADR 0004) — not the collapse a barely-seen language usually
produces.

## The pivot was assumed too

`config/default.yaml` routed Friulian through Italian: `src_token: ita_Latn`,
with the comment *"pivot through Italian: tr -> it -> fur"*. Also never
measured. Direct against pivot, same six sentences:

| | direct `tr→fur` | pivot `tr→it→fur` |
|---|---|---|
| median latency | **331 ms** | 585 ms |
| model passes | 1 | 2 |
| routes agreeing exactly | 2 of 6 | |

Where they differ there is no consistent winner. Direct rendered *"thank you
for being here"* as `grazie di jessi li`, closer than the pivot's `grazie par
jessi vignûts` (for having come). The pivot rendered *"falling behind"* as
`a restin indaûr`, better than direct's `a restin dismenteadis` (remain
forgotten). And the pivot introduced a **tense error** — present `yetiyor`
became past `bastave` — by passing through an Italian rendering that had
already drifted.

That last one is the structural argument. A pivot cannot recover information
the first hop lost, and it adds a place to lose some.

## Decision

1. **`mt.friulian.src_token: tur_Latn`** — translate directly, not through
   Italian. Half the latency, one fewer failure point, no measured quality
   cost.
2. **No fine-tuning for now.** The baseline is good enough to build the
   demonstration on. Fine-tuning stays available if a speaker judges the
   prepared sentences inadequate, and the corpus notes stay in
   [04-adding-a-language.md](../04-adding-a-language.md).
3. **What the branch actually needs is a Friulian speaker** looking at about
   ten prepared sentences. ARLeF is the obvious contact.

## What this changes about the talk

The Friulian segment was scoped as *"we fine-tuned a model for a language with
almost no data"*. It can now be something more honest and more interesting:

*"Nobody trained anything for Friulian here. One general model has seen it, and
it produces real Friulian — with mistakes a native speaker catches immediately.
And no text-to-speech model for it exists anywhere, so you are about to hear it
read with Italian phonetics."*

That is the argument the talk is making, demonstrated rather than asserted.

The absence of a Friulian voice was also checked rather than assumed on the
same day: HuggingFace's `fur` language tag returns translation models only, and
Meta's MMS-TTS, which covers over 1100 languages, does not include it.

## The pattern, again

Three assumptions were carried for weeks and all three were wrong when
measured: Kokoro was eliminated in one line (ADR 0006), the commit-wait trade
was never tested (ADR 0009), and Friulian was scoped as a training problem
without anyone running the model.

Each was cheap to check and expensive to assume. The common shape is a
plausible sentence written early, repeated in every document afterwards, and
never executed. **Prose does not run.**

## Caveats

* **Six sentences, and no Friulian speaker.** This says the output is real
  Friulian with plausible morphology, not that it is good Friulian. The
  judgement that matters has not been made yet.
* Round-trip similarity measures whether meaning survived two passes, which is
  weaker than a reference translation. FLORES-200 has 997 `fur_Latn` dev
  sentences and is the proper test; it is gated behind a licence acceptance.

## Revisit if

- A Friulian speaker reads the prepared sentences and finds them wrong in kind
  rather than in detail.
- The demonstration grows beyond prepared sentences into live Friulian.

---

## Appendix, 2026-09-01 — the search for a Friulian voice, and what it found

The user asked whether a Friulian voice could be obtained anywhere: a model to
run locally, a dataset, a commercial API, a recording from broadcast. Five
routes were checked rather than assumed.

| Where | Coverage | Friulian |
|---|---|---|
| Meta **MMS-TTS** | 1100+ languages | **no** |
| **eSpeak NG**, installed on this machine | 222 voices | **no** |
| **ElevenLabs** multilingual v2 / Flash v2.5 | 29 / 32 languages | **no** |
| HuggingFace models, `fur` language tag | — | translation only, no TTS, no ASR |
| HuggingFace datasets, `fur` language tag | 30 results | **all text** — Wikipedia, FLORES, OPUS, Tatoeba, UDHR. Not one speech corpus |

eSpeak NG is worth singling out. It is a formant synthesiser from the 1990s
lineage, it sounds robotic, it ships **222 voices** including Catalan and
Romanian — and it does not have Friulian, or Ladin, or Sardinian, or Romansh.
A synthesiser that will read Chuvash cannot read the language of the region
this talk is being given in.

### This is not a dead end. It is the finding.

The segment was going to *assert* that minority languages are left behind.
It can now *show* it, with receipts:

> A language with 600,000 speakers, official status in an Italian region, in
> one of the wealthiest parts of the European Union. Meta's model speaks 1100
> languages and not this one. eSpeak speaks 222 and not this one. ElevenLabs
> sells 32 and not this one. There is no published recording corpus at all.
>
> The translation you just read was produced by a general model that happened
> to see some Friulian text. The voice you just heard was Italian, because
> nothing else exists.

That is a stronger slide than any demonstration would have been.

### What ElevenLabs would and would not buy

It has no Friulian model, so it would read Friulian text with **Italian**
phonetics — the same substitution Kokoro already makes locally and for nothing.
The gain would be naturalness, not correctness.

Against that it costs the project's central claim. The talk's argument is that
this runs on one laptop with nothing leaving the room, demonstrable by
unplugging the network. A Friulian sample generated by an API contradicts the
slide it appears on. Not taken.

### What is still worth getting, and what it is for

Two different things, often confused:

**A Friulian reader** — someone who checks the ten prepared sentences. This is
the only thing that unblocks the branch, and it needs a person, not audio.
**ARLeF** is the obvious approach: a public agency of the Friuli Venezia Giulia
region whose statutory job is promoting the language, which explicitly offers
language consultancy to public and private bodies. Via della Prefettura 13,
Udine; +39 0432 555812; arlef@regione.fvg.it; open to the public 10:00–12:00.
A student-project request to check ten sentences for a talk about minority
languages in AI is squarely their mandate.

**A recording of spoken Friulian** — for the audience's ear, played beside the
system's output so the room hears the gap. A short broadcast excerpt used for
commentary is ordinary practice. It proves nothing about whether *our*
sentences are right.

**And a third option that is better theatre than either.** The talk is given in
Italy. Ask the room: *"is anyone here a Friulian speaker? Tell me whether this
is right."* A live correction from the audience makes the point better than a
pre-checked translation, and costs nothing if nobody answers.
