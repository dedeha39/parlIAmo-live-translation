# parlIAmo — offline live speech translation with OmniVoice voice cloning

**Speak one language, the room hears another, in your own voice — with no
cloud, no API keys and no network at run time.** Real-time speech-to-speech
translation built on Whisper (faster-whisper), NLLB-200, Kokoro TTS and
OmniVoice, with RVC and Seed-VC as alternative voice cloners. Seven languages in any
direction out of the box (Turkish, Italian, English, Spanish, French, German,
Friulian); the models underneath reach far more. Microphone to
loudspeaker, everything runs on one consumer laptop with an 8 GB GPU.

Built for a live talk about AI voice cloning and phone scams, given in Turkish
to an Italian-speaking audience of seniors in Friuli. The system was the talk's
live demo: the room heard the speaker in Italian, in a voice cloned from them
with their consent — which is exactly the point the talk was making about how
accessible voice cloning has become.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![Licence: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-green)](LICENSE)
[![Tests: 819](https://img.shields.io/badge/tests-819%20passing-brightgreen)](tests/)
![Platform: Windows 11](https://img.shields.io/badge/platform-Windows%2011-lightgrey)

```
 microphone ─▶ VAD ─▶ speech recognition ─▶ translation ─▶ speech synthesis ─▶ voice ─▶ speakers
   (Turkish)  Silero   faster-whisper        NLLB-200       Kokoro / Piper     cloned    (Italian)
                       large-v3-turbo        (CTranslate2)                     voice
                                     └─▶ subtitles on a second screen, Friulian branch
```

---

## What it does

- **Simultaneous translation between many languages** — see
  [Languages](#languages). Turkish → Italian is the default because that was
  the talk; the pair is chosen on the Setup page, and Italian → Turkish works
  too, for questions from the room.
- **Speaks in a cloned voice** — three interchangeable voice services, all
  local: OmniVoice (speaks the sentence in the reference voice), Seed-VC
  (zero-shot conversion) and RVC (a voice trained once). Cloning is refused
  without a consent record.
- **Friulian** — a low-resource minority language, translated directly and
  shown beside a native speaker's recording.
- **An operator page** (Live, Setup, Voices, Studio, Friulian, Ethics) and an
  **audience screen** with live subtitles, served locally from the laptop.
- **A voice Studio** — every OmniVoice feature with sliders: clone from a
  recording or an upload, design an invented voice from a description, derive a
  new voice from a consented one, non-verbal tags, 646 languages.
- **A pre-flight checklist** that says what would go wrong if you started now,
  in plain words.

## Languages

Ready to use, in **any direction — 42 pairs, all of them spoken aloud**:

| Language | Recognised | Translated | Spoken by |
|---|:-:|:-:|---|
| Turkish | ✓ | ✓ | Piper (CPU) |
| Italian | ✓ | ✓ | Kokoro (GPU) |
| English | ✓ | ✓ | Kokoro |
| Spanish | ✓ | ✓ | Kokoro |
| French | ✓ | ✓ | Kokoro |
| German | ✓ | ✓ | Piper |
| Friulian | partly — Whisper has no Friulian and hears it as Italian | ✓ | Kokoro's Italian voice — no Friulian voice exists anywhere; disclosed on stage |

Underneath, the reach is far wider: Whisper was trained on **~99 languages**
(quality varies widely by language — measure before relying on one), NLLB-200
translates between **~200**, Kokoro also speaks Portuguese, Hindi,
Japanese and Chinese, and OmniVoice — the cloned-voice service and the
Studio — speaks **646**. A language without a synthesiser voice still works as
live subtitles. Adding one is a tag in a table plus a voice:
[docs/04-adding-a-language.md](docs/04-adding-a-language.md). Calabrese is the
one asked for that no model translates
([ADR 0014](docs/adr/0014-spanish-german-turkish-and-calabrese.md)).

## Measured, not assumed

Every number here was measured on the reference laptop (RTX 4070 Laptop 8 GB,
i9-13900HX, 32 GB, Windows 11) or in the hall itself.

| | |
|---|---|
| End-to-end lag, cloned voice (RVC) | **1.58 s** mean, 1.80 s p95, from the moment the speaker stops |
| End-to-end lag, generic voice | **1.29 s** mean |
| Speech recognition, the speaker's own voice | **14.00 % WER**, p95 293 ms |
| Translation tr→it | chrF++ **45.38**, 355 ms median |
| Every model loaded at once | **7292 of 8188 MiB** VRAM, measured co-resident, never summed |
| Speakers → microphone round trip in the hall | **306 ms** (the laptop's own speakers: ~430 ms of vendor DSP) |
| Whisper hallucinations caught by the filters | loops, impossible speech rates, subtitle credits, and **76** recitals of its own hotword list — with no real sentence lost across 7,152 logged transcripts |

The full latency table, the model bake-offs and the decisions that were made
wrongly first and corrected by measurement are in [docs/adr/](docs/adr/) and
[docs/11-model-survey.md](docs/11-model-survey.md).

## Built for a stage

A live demo fails in ways a benchmark never sees. Each of these was found in a
rehearsal and is now handled — with a test that fails on the old code:

- **Feedback.** The room's speakers reach the microphone, and the system would
  translate its own Italian. A half-duplex gate deafens the microphone while it
  speaks and for a tail sized from a measurement *in the room*
  (`measure_audio_device.py`, `measure_room_echo.py`) — not from what the driver
  reports.
- **Everything else the speakers play.** Videos on the slides, applause,
  questions: **P** pauses listening; whatever was said before the press is
  still translated.
- **Choppy output.** On Windows, MME and DirectSound starve while Python is busy
  (39 gaps in 8 s in the hall, WASAPI none). The device picker ranks WASAPI
  first, warns on the others, and gaps are logged.
- **Devices that move.** Windows renumbers devices whenever one connects; the
  choice is kept by name and host API, with a fallback that is reported, never
  silent.
- **Whisper inventing text** from breaths, silence or a quiet microphone —
  filtered by speech rate, repetition, known subtitle credits and hotword
  recitals, and kept on disk for listening.

## Voice cloning and ethics

This project clones voices. That capability is the subject of the talk, not a
side effect of it.

- A voice is cloned **only with a consent record**; the software refuses
  otherwise, at start and on every mid-talk swap.
- Volunteer recordings are deleted after the event — the application has the
  button and the consent form promises it.
- Every demonstration clip announces itself (*"questa voce è sintetica"*) at
  unpredictable points, so it cannot be trimmed into something usable.
- No public figures, no fictional characters, no voice without its owner's
  consent — regardless of what is technically possible.
- This pipeline **does not watermark**, and the documentation says so rather
  than implying otherwise.

The reasoning, the consent form and the relevant law (EU AI Act Art. 50) are in
[docs/07-ethics-and-consent.md](docs/07-ethics-and-consent.md).

---

## Quick start

Requirements: Windows 11, an NVIDIA GPU with 8 GB, Python 3.12 (conda), and
the models downloaded once. Full setup, environment by environment:
[docs/01-setup.md](docs/01-setup.md).

```bash
# 1. a voice service, in its own environment (pick one)
<omnivoice-venv>\Scripts\python.exe scripts\omnivoice_server.py
<seedvc-env>\python.exe scripts\voice_conversion_server.py
<applio-env>\python.exe scripts\rvc_server.py --model <your-voice> --port 8766

# 2. the application
<parliamo-env>\python.exe scripts\live_translate.py --ui
```

Open the address it prints, read **Setup**, pick the microphone and the
speakers (WASAPI), press **Start**. The audience screen is the same address
plus `/screen`.

First time in a room — the gate is sized from these:

```bash
python scripts/measure_audio_device.py --skip-bandwidth --skip-noise --write-config --label venue
python scripts/measure_room_echo.py --input "<mic>@Windows WASAPI" --output "<speakers>@Windows WASAPI" --write-config
```

No microphone to hand? Replay a recording through the identical pipeline:

```bash
python scripts/live_translate.py --file recording.wav --save-audio runs/replay
```

## Documentation

| | |
|---|---|
| [docs/00-state.md](docs/00-state.md) | Everything in one place: what is decided and why, what is measured, what is open |
| [docs/09-reference.md](docs/09-reference.md) | Every module, public function, the HTTP API, the threading model, the sharp edges |
| [docs/10-rehearsal.md](docs/10-rehearsal.md) | What to run before a talk, in what order, and what each step should show |
| [docs/02-architecture.md](docs/02-architecture.md) | The shape of the system |
| [docs/03-hardware-budget.md](docs/03-hardware-budget.md) | The VRAM contract every model choice was made under |
| [docs/04-adding-a-language.md](docs/04-adding-a-language.md) | How the Friulian branch was approached |
| [docs/adr/](docs/adr/) | Fifteen architecture decision records |

```
src/parliamo/    audio · asr · mt · tts · pipeline · ui · ethics · eval
scripts/         the application, voice services, measurement tools
tests/           819 tests — pytest, no GPU or sound card needed for most
config/          default.yaml; your machine's values go in config/local.yaml (git-ignored)
```

```bash
python -m pytest        # the suite
python -m ruff check .  # lint
```

## Known issues

- The translator (NLLB-200) can drop the last clause of a long sentence when
  the recogniser writes it without punctuation; with punctuation it is
  complete.
- After unplugging a headset while the application runs, Stop/Start is not
  enough — restart the application so the device list is read again.
- OmniVoice ran ~3 s per sentence at one venue against ~1.2 s at home; not yet
  explained.

## Licence

The code is **Apache-2.0** ([LICENSE](LICENSE)). The models it loads carry their
own licences, and two on the default path are **non-commercial**: NLLB-200 and
OmniVoice's weights are CC-BY-NC-4.0. Read
[docs/08-model-licences.md](docs/08-model-licences.md) before building anything
commercial on top.

## Acknowledgements

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and OpenAI Whisper ·
[NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) and
[CTranslate2](https://github.com/OpenNMT/CTranslate2) ·
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) ·
[Piper](https://github.com/rhasspy/piper) ·
[Silero VAD](https://github.com/snakers4/silero-vad) ·
[OmniVoice](https://github.com/k2-fsa/OmniVoice) ·
[Seed-VC](https://github.com/Plachtaa/seed-vc) ·
[Applio / RVC](https://github.com/IAHispano/Applio) ·
the Friulian recordings of [Wikitongues](https://wikitongues.org) (CC BY-SA).
