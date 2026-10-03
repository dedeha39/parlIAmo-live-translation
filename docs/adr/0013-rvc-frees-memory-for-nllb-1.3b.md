# 13. RVC frees the memory NLLB-1.3B was rejected for

Date: 2026-09-23

Status: accepted — offered on the Setup tab, guarded by pre-flight; the default
stays 600M because the default voice service is Seed-VC

## Context

[ADR 0007](0007-nllb-size-and-sentence-splitting.md) found NLLB-1.3B better
than 600M on every quality measure (chrF++ 47.11 against 45.38) and rejected it
on memory alone: +1118 MiB co-resident left ~350–700 MiB of headroom. That
budget had Seed-VC in it, holding ~2.9 GB in its own process.

[ADR 0012](0012-rvc-beside-seedvc.md) then added RVC for the presenter's voice.
Its memory was never measured separately; the rehearsal guide said both
services "hold ~3 GB". The presenter's complaint in the meantime was not only
speed — *"dediğimi düzgün anlamıyor gibi"*, it does not seem to understand what
I say.

## Measured

**The RVC service on its own**, `nvidia-smi` sampled every 50 ms:

| | MiB |
|---|---|
| loaded, idle | +660 |
| peak during a conversion | **+1204** |
| Seed-VC, for comparison (ADR 0003) | ~2900 held, +358 per conversion |

RVC holds ~2 GB less than Seed-VC. Applio empties the CUDA cache after every
conversion, which is why the idle and peak figures differ.

**End to end with RVC**, the 68 s reference recording, 11 sentences, report
files `runs/live/translate-20260923-205957.json` and `-210130.json`:

| translator | mean lag | p95 | mt | peak VRAM (whole machine) |
|---|---|---|---|---|
| NLLB-600M | 1.72 s | 2.07 | 0.23 s | 6038 MiB |
| NLLB-1.3B | **1.92 s** | 2.39 | 0.41 s | **6942 MiB** of 8188 |

1.3B fits beside RVC with ~1.2 GB to spare. It costs 0.2 s per sentence.

**What the 0.2 s buys**, read sentence by sentence on the same run (Turkish
source, both Italian outputs):

| | better | example |
|---|---|---|
| 1.3B | meaning | "konuşmacı sayısı azalan diller" → *lingue con un numero di parlanti in diminuzione*; 600M: *lingue in diminuzione*, and the wrong subject |
| 1.3B | meaning | "bir sesin ne kadar hızlı kopyalanabildiği" → *una voce può essere copiata*; 600M: *un suono* — "a sound", in a talk about voices |
| 1.3B | meaning | "rahatsız edici ... temiz bir ses kaydı" → *inquietante ... pulita*; 600M: *fastidiosa*, and "clean" dropped |
| 600M | meaning | "şu masanın üstündeki" → *sul tavolo*; 1.3B: *sul vostro tavolo*, "on *your* table" |
| 600M | register | *mi sono fatto* against 1.3B's literary *mi feci* |
| — | equal | the other six |

Three meaning errors fixed against one introduced, which is what chrF++ +1.7
predicted. Neither model can fix the first sentence: the recogniser heard
*takdir* (appreciate) where the speaker said *taklit* (imitate), and both
translate the wrong word faithfully. That one is the recogniser's; see
[11-model-survey.md](../11-model-survey.md).

## Decision

- **NLLB-1.3B is recommended whenever the voice service is RVC.** The picker
  on the Setup tab says so: *fits beside RVC, NOT beside Seed-VC*.
- **The default stays 600M**, because the configured service can be Seed-VC
  and the volunteer demonstration needs it.
- **Pre-flight warns** when 1.3B is chosen and the service that answers is
  Seed-VC.

## Consequences

- One choice to make before the talk, not a code change: with RVC, choose
  NLLB 1.3B on the Setup tab and press Start.
- The 0.2 s lands on every sentence. Against 1.72 s it is 12%.
- Switching to Seed-VC for the volunteer segment means switching the
  translator back — both apply at the next Start, so it is one stop/start
  either way.

## Correction recorded here because it was found here

Re-measuring for this ADR would not match ADR 0012's table, and the reason was
the table: the summary script added the 0.5 s commit wait to a total that
already contained it. Every end-to-end figure in the documents was 0.50 s too
high — RVC was 1.58 s, not 2.08; Seed-VC 2.29, not 2.79. The differences between
configurations were right, so no decision changes. The documents are
corrected. Today's rerun of the same RVC configuration measured 1.72 s against
1.58 on 2026-09-15; the stage times account for it (recognition +0.04 s,
synthesis +0.05 s, translation +0.04 s — the same code on a busier desktop).
