# 14. Spanish, German and Turkish spoken; Calabrese not translatable

Date: 2026-09-25

Status: accepted for Spanish, German and Turkish. Calabrese: no machine
translation exists that can be offered honestly; options below.

## Context

The presenter asked for Calabrese, Spanish and German as well as Italian.
Before this, Spanish was wired (NLLB `spa_Latn`, Kokoro voices) but untested;
German translated but could only be subtitles (Kokoro has no German); Turkish as
a *target* - an Italian question heard in Turkish - was subtitles only for the
same reason.

## Measured: translation quality from Turkish

FLORES-200 devtest, first 200 sentences, sentence-split as live, chrF++:

| target | NLLB-600M | NLLB-1.3B |
|---|---|---|
| Italian | 47.5 | 49.2 |
| Spanish | 46.7 | 48.6 |
| German | 47.5 | 49.9 |
| Friulian | 42.7 | 46.3 |
| Sicilian | 29.2 | 30.5 |

Spanish and German are as good as Italian. Sicilian is not, and the reason is
worse than the score: **148-174 of the 200 "Sicilian" outputs carry Corsican
forms** (*hè statu*, *cù u*, *chjaru*), none of the other targets do, and
translating from Italian instead of Turkish does not help (148/200). NLLB's
Sicilian training data is contaminated with Corsican - a known problem, which
the `Napizia/Good-Sicilian-in-NLLB` filtering dataset exists to address.

## Calabrese

- Central-southern Calabrian belongs with Sicilian in the Extreme Southern
  Italian group; northern Calabrian with Neapolitan. Neither Calabrese nor
  Neapolitan has an ISO code of its own in NLLB, in Google's open MADLAD-400
  (checked: no `nap`, no `scn`), or in Google Translate's 2024 expansion, which
  added Sicilian, Friulian, Venetian, Lombard and Ligurian but not these.
- The one proxy, NLLB's Sicilian, answers in Corsican most of the time. An
  audience of elderly Calabrians would hear a language from another island.
- Real Calabrese that exists openly: a Wikitongues speaker filed as
  "Napoletano-Calabrese" (Foffo, CC BY-SA 4.0, variety not stated - to be
  checked by listening); the VIVALDI dialect atlas (non-commercial licence);
  printed dictionaries.

**Decision: no live Calabrese translation.** What can be done honestly, in
the Friulian manner ([ADR 0010](0010-friulian-baseline.md)): a native speaker
translates the talk's key sentences once, and those are shown and spoken
(Italian phonetics, disclosed); or a real Calabrian speaker's recording is
played as recorded. Both need a person the presenter knows.

## Measured: voices

| language | synthesiser | voice | where | median F0 |
|---|---|---|---|---|
| Italian | Kokoro | if_sara | GPU | 223 Hz |
| Spanish | Kokoro | **em_alex** (was ef_dora) | GPU | 139 Hz |
| German | **Piper** | de_DE-thorsten-high | CPU | 125 Hz |
| Turkish | **Piper** | tr_TR-dfki-medium | CPU | 105 Hz |

Piper runs on the CPU and takes no GPU memory. On this hybrid CPU its default
of one thread per core was the slowest setting: a German sentence took 2.0-3.1 s,
1.14-1.38 s with 8 threads (`PiperBackend.threads`). On the GPU
(onnxruntime-gpu 1.22, CUDA 12 like torch) the same sentence took 0.51-0.59 s -
not adopted, because it means replacing the environment's onnxruntime; the
option is recorded.

**One pitch cannot fit four voices.** RVC keeps the pitch it is given, and the
server took one `--pitch` for every language: -8 fits if_sara and would have put
the Spanish and German men near 80 Hz. Now the translator measures the
synthesiser's voice at warm-up (pYIN, `audio/pitch.py`), and every sentence
carries its own shift to `tts.conversion.presenter_f0_hz` (139). Converted
output, measured: Italian 123 Hz, Spanish 128, German 137.

The measurement phrase matters: on a short greeting ("Buongiorno a tutti,
cominciamo.", 2.5 s) if_sara read -10.2 semitones from the presenter; on two
plain sentences (6.1 s), -7.5 - three runs within 0.1. The warm-up phrases are
now those sentences in each language.

## Measured end to end

The 68 s reference recording, RVC, NLLB-600M, 11 sentences each:

| direction | mean lag | p95 | synthesis |
|---|---|---|---|
| Turkish -> Italian | 2.17 s | 2.86 | 0.27 |
| Turkish -> Spanish | 1.98 s | 2.49 | 0.27 |
| Turkish -> German | 3.60 s | 5.35 | 1.78 |
| Italian -> Turkish (no cloning) | 1.25 s | - | 0.20 |

The Italian figure is higher than 2026-09-23's 1.72 s on the same code path -
a busier machine that afternoon (German before the thread fix measured 5.23 s).
German is slower by what Piper costs on the CPU.

## Also fixed on the way

- `tts.voice` and `tts.speed` never reached the synthesiser: `build()` did not
  pass them. Invisible while the default voice was the configured one.
- The Turkish hotword list was applied whatever language was spoken. A
  `.tr.` list is now used for Turkish only, or a `.<lang>.` list if one exists.
- The invented-sentence filter did not count `ä` or `á` as vowels.
- `live_translate.py --source` sets the recogniser and the translator together.

## Consequences

- On the page: Spanish and German are spoken targets; Turkish is spoken too,
  for questions from the room. German says it is slower.
- German costs ~1.5 s more a sentence than Italian until Piper moves to the GPU.
- Calabrese waits on a person, not on a model.
