# parlIAmo — current state

**Read this first.** One document, everything needed to pick the project up
cold: what it is, what is decided and why, what is measured, what is open, and
how to run it. Updated 2026-10-02.

---

## 1. What this is

A simultaneous speech translation system that runs entirely on one laptop, with
no network calls at run time. The speaker talks Turkish; the room hears Italian,
optionally in the speaker's own cloned voice.

It exists for a one-hour presentation titled **parlIAmo** (*parliamo* + *IA*),
about how AI is used in speech and what that means for security, given to an
elderly, Friulian audience: hence the Friulian branch.
The talk is delivered in Turkish and translated live. Slides are Italian;
code and documentation are English. The deck and how to run everything:
[10-rehearsal.md](10-rehearsal.md).

### Why local

The usual argument — "APIs add latency" — is only partly true. In simultaneous
translation the dominant delay is *policy* delay: you must wait for enough
context before you can translate at all. The real reasons are that a venue's
internet cannot be trusted, that a volunteer's voice recording is biometric data
that should never leave the machine, and that "this runs on a laptop" is itself
the argument the talk is making about how accessible voice cloning has become.

---

## 2. Hardware and environments

| | |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop, 8188 MiB, compute 8.9, driver 610.47 |
| CPU / RAM | Intel i9-13900HX, 32 logical cores / 32 GB |
| Machine | Acer Predator Helios Neo 16 (PHN16-71), Windows 11 |

**Four Python environments.** This is not accidental; see §6.

| Env | Python | torch | Holds |
|---|---|---|---|
| `parliamo` | 3.12 | 2.11.0+cu128 | the pipeline: audio, ASR, MT, Kokoro |
| `seedvc` | 3.10 | 2.6.0+cu124 | voice conversion only, behind a socket |
| `translator` | 3.11 | 2.4.0 | the user's earlier prototype, untouched |
| Applio | — | 2.7.1+cu128 | RVC: the presenter's voice, trained once, served by `rvc_server.py` (ADR 0012) |

Interpreters are addressed by full path; conda is not on PATH:

```
C:\Users\you\miniconda3\envs\parliamo\python.exe
C:\Users\you\miniconda3\envs\seedvc\python.exe
```

---

## 3. The pipeline

```
microphone ─▶ VAD ─▶ Whisper ─▶ NLLB ─▶ Kokoro ─▶ Seed-VC | RVC ─▶ speakers
   (tr)      Silero   turbo    600M              (socket, by port)  (it, cloned)
                                   └─▶ Friulian branch ─▶ on-screen text + native clips
```

Four threads, because the stages have incompatible timing:

1. **PortAudio callback** — delivers 32 ms blocks, must never block.
2. **Segmenter** — VAD on every block (<5 ms each), must keep exact pace.
3. **Recogniser** — 200–800 ms bursts.
4. **Delivery** — translate, synthesise, convert, play; spends most of its time
   waiting for audio.

### Measured latency, end to end

Same 68 s of the presenter's continuous speech through the whole pipeline -
`live_translate.py --file` - eleven sentences, 500 ms commit wait, no
failures. Lag from the moment the speaker stops, commit wait included:

> **Corrected 2026-09-23.** Until then every figure in this table was 0.50 s
> too high: the script that summarised the run reports added the commit wait
> to `total_lag_s`, which already contains it. The stage columns were right;
> only the totals were wrong. Recomputed from the same report files
> (`runs/live/translate-20260915-*.json`).

| configuration | mean | p95 | asr | mt | tts | vc |
|---|---|---|---|---|---|---|
| subtitles only | 1.10 s | 1.33 | 0.37 | 0.23 | — | — |
| generic voice (Kokoro) | 1.29 s | 1.55 | 0.37 | 0.23 | 0.19 | — |
| cloned, RVC (trained once) | **1.58 s** | **1.80** | 0.32 | 0.19 | 0.14 | 0.43 |
| cloned, Seed-VC (zero-shot) | 2.29 s | 2.96 | 0.39 | 0.25 | 0.22 | 0.93 |

Voice conversion is the largest term and the one with a choice in it: Seed-VC
clones any consented reference with no training and costs 0.93 s at every
sentence; RVC trains the presenter's voice once and applies it in 0.43 s. Both
run behind the same socket protocol; `tts.conversion.port` picks one
([ADR 0012](adr/0012-rvc-beside-seedvc.md)).

**The tail is driven by sentence length, not by the pipeline.** Synthesis and
conversion both scale with how much audio there is to produce, and continuous
speech averages 5.5 s per segment with a tail at 10.1 s. Cutting long segments
at clause boundaries was tried, measured and rejected - the longest segment
contains no pause at all; see [ADR 0009](adr/0009-commit-wait.md).

The earlier prototype, reproduced as closely as offline allows (Whisper
medium, 0.3 s wait, generic voice): 1.19 s. This pipeline generic at the same
wait: 1.04 s. It felt faster because it cloned nothing and streamed a cloud
synthesiser; its recogniser was slower than ours.

### What the audience *reads*, which is much sooner

Provisional subtitles appear **1.68 s** into an utterance regardless of how long
it runs on. On the 10.1 s segment above that is nine seconds before the audio.

Verified free on the same recording: with partials on, mean lag 2.779 s over 73
partials; with them off, 2.798 s. Identical within noise, because the recogniser
is idle while someone is speaking. See [ADR 0008](adr/0008-partial-subtitles.md).

### Measured VRAM, co-resident

Measured with everything loaded at once and the conversion service running, not
by adding per-model figures together - ADR 0003 forbids the latter, and an
earlier version of this table did it anyway.

| Stage | Added | Running total |
|---|---|---|
| Desktop, apps closed | — | ~1200 |
| Seed-VC service (separate process) | ~2900 | ~4130 |
| ASR large-v3-turbo | 1213 | 5346 |
| MT NLLB-600M | 844 | 6190 |
| Kokoro | 348 | 6538 |
| Kokoro generation activations | 396 | 6934 |
| One voice conversion | 358 | **7292** |

**Peak 7292 MiB of 8188, headroom 896.** Reproduce with
`scripts/measure_vram_budget.py --tts kokoro --conversion`. Full detail in
[03-hardware-budget.md](03-hardware-budget.md).

---

## 4. Decisions, and the evidence for them

Every decision has an ADR in `docs/adr/`. The short version:

| Choice | Why | ADR |
|---|---|---|
| Windows native, not WSL2 | WSL2 has no direct microphone access; audio reliability outranks model choice | [0001](adr/0001-windows-native.md) |
| `large-v3-turbo`, beam 5, int8, hotwords | 14.00% WER in situ; p95 293 ms against large-v3's 724 ms | [0002](adr/0002-asr-model.md) |
| VRAM measured co-resident, never summed | estimates were wrong by 3.3× and 1.8× in opposite directions | [0003](adr/0003-vram-budget.md) |
| NLLB-600M, not an LLM | LLM took 2671 ms/sentence, output the wrong language, dropped a negation | [0004](adr/0004-mt-backend.md) |
| Streaming matters more than model size | Chatterbox at RTF 1.3 made the whole pipeline unusable | [0005](adr/0005-tts-latency.md) |
| Kokoro + Seed-VC, split | 42× faster generation; zero-shot cloning kept | [0006](adr/0006-synthesis-architecture.md) |
| Keep NLLB-600M; split sentences first | 1.3B is better but needs +1118 MiB co-resident; splitting recovers a dropped clause **and** is faster | [0007](adr/0007-nllb-size-and-sentence-splitting.md) |
| Show the subtitle before the sentence ends | the recogniser is idle while someone speaks; text arrives 2.8–6.8 s earlier for nothing | [0008](adr/0008-partial-subtitles.md) |
| Commit wait 500 ms, not 700 | measured on continuous speech: lower WER, lower CER, faster ASR, no mid-phrase cuts — the trade everyone assumed does not exist | [0009](adr/0009-commit-wait.md) |
| Friulian needs a speaker, not a fine-tune | NLLB already produces real Friulian (`nol è`, `lis lenghis minoritariis`); direct beats the assumed pivot at half the latency | [0010](adr/0010-friulian-baseline.md) |
| RVC beside Seed-VC | Seed-VC clones at every sentence (0.93 s, 41% of the total); RVC trains the presenter's voice once and applies it in 0.43 s — 1.58 s mean against 2.29 end to end. Volunteers still need zero-shot | [0012](adr/0012-rvc-beside-seedvc.md) |
| OmniVoice as a third voice service | speaks the sentence from its text in any consented voice; 16 steps ~1.0 s, 2.5 s end to end, WER 1.7% once digits are spelled; the reference must be cut on sentence boundaries | [0015](adr/0015-omnivoice.md) |
| Spanish, German and Turkish spoken; no live Calabrese | Spanish and German translate as well as Italian (chrF++ 46.7-49.9); Piper voices German and Turkish on the CPU; each sentence carries its own RVC pitch; NLLB's Sicilian - the only Calabrese proxy - answers in Corsican | [0014](adr/0014-spanish-german-turkish-and-calabrese.md) |
| NLLB-1.3B when the voice is RVC | RVC holds 0.7 GB, not ~3; 1.3B then fits (6942 MiB peak) and fixed 3 meaning errors in 11 sentences for 0.2 s each | [0013](adr/0013-rvc-frees-memory-for-nllb-1.3b.md) |
| The audio "hardware fault" was a threading fault | WASAPI is COM and COM is per-thread: all three outputs opened from the main thread, none from the worker that builds the pipeline. Not Bluetooth, not torch — both ruled out by probing before and after | [0011](adr/0011-audio-device-opening.md) |

### The four measurements that changed a decision

**Public benchmarks ranked the recognisers but not the gap.** On FLEURS,
large-v3 and turbo differed by 0.56 WER points. On the presenter's own voice
through this microphone: **3.93 points**. Choosing on public data alone would
have been choosing on a difference that does not hold.

**A Turkish fine-tuned Whisper was 5 points worse than stock.** Trained on
Common Voice short read sentences, tested on 5–8 second presentation speech.
"Fine-tuned for our language" is a claim about a training set, not about our
audio.

**Hotwords helped the weak model and hurt the strong one.** −2.21 WER points for
turbo, **+0.98** for large-v3. Biasing pulls ordinary words toward the list.

**Synthesis was 4.5–6.1 s and nobody had measured it.** Three ADRs argued over
hundreds of milliseconds of recognition latency while an unmeasured stage twenty
times larger sat downstream.

---

## 5. Bugs found by measurement

Recorded because each was invisible without a number, and several would have
failed silently on stage.

| Bug | Effect | Found by |
|---|---|---|
| Default mic was the MME clone | +88 ms on every segment | listing devices |
| Round-trip measured on two streams | meaningless numbers, 521 ms spread | results not reproducing |
| Gate sized from reported latency | mic reopens mid-sentence → **feedback loop** | measuring real output latency |
| `_fit()` trimmed resampled chunks | **53% of the microphone discarded** | block count not adding up |
| Pre-roll shorter than confirmation | 130 ms cut from every utterance's start | testing against real speech |
| `str.lower()` on Turkish | invented errors in the reference | knowing the language |
| `40.000` → "kırk sıfır" | corrupted the reference every model was scored against | reading the output |
| Whisper repetition loop | one utterance scored 1009% WER | reading the worst cases |
| Zero-padded dB smoothing | detector returned the top of the axis | the tool contradicting itself |
| WAV saved as PCM_16 | re-analysis disagreed with the live run | same |
| `chatterbox-tts` pinned torch | **silently replaced CUDA torch with the CPU wheel** | checking CUDA after installing |
| `mt.backend: llama_cpp` in config | live CLI would not start; ADR 0004 never reached the config | writing the first script that read it |
| NLLB drops a trailing sentence | **a whole clause vanishes, fluently and silently** | replaying real speech end to end |
| Turkish `print` on a cp1252 console | **subtitles crash and are swallowed** — operator sees nothing | printing Turkish through a pipe |
| Benchmarks labelled every run "NLLB-600M" | a report attributing 1.3B's numbers to 600M | running `--nllb` against another model |
| VRAM budget script measured Chatterbox | budget printed for an architecture that no longer ships | reading it against ADR 0006 |
| Reference voice chosen by hand | 49.5% digital silence — the worst of 40 files | measuring them all |
| Two conversion servers on one port | 5.8 GB held, conversions 4.8–9.8× slower | a 6× "regression" that was not real |
| Streaming ledger never reset without a commit | **chunk mode went silent for the rest of the talk** after one dropped utterance | replaying two utterances with the commit of the first removed |
| Blocking calls inside the page's async handlers | pre-flight or a voice swap **froze the subtitles for 1.3 s** | timing `/state` while a slow action ran |
| Latency table counted the commit wait twice | **every end-to-end figure 0.50 s too high** — RVC read 2.08 s, was 1.58 | a re-measurement that would not match the table |
| **Cloned sentences played at the wrong rate** | RVC answers at 48 kHz, Seed-VC at 22.05; playback opened at the synthesiser's 24 kHz and was handed the samples without their rate. RVC played at half speed an octave down - "slowed down, robotic" - and kept the microphone gated twice as long; Seed-VC played 9% fast. Every saved file was right, so no measurement saw it | the presenter's first rehearsal with RVC through the speakers (2026-09-26); on the real output, 2 s of RVC audio took 4.17 s, 2.12 after the fix |
| Start pinned to one voice-service port | the talk swaps RVC for Seed-VC for the volunteer; the pipeline would have knocked on the closed port and spoken the volunteer generic | writing the volunteer step of the deck's notes |
| Pitch measured on a short greeting | if_sara read at -10.2 semitones on "Buongiorno a tutti, cominciamo.", -7.5 on two plain sentences | repeating the measurement |
| Playback callback between two lines of `submit()` | **the gate released just before a sentence played** - a sentence to an open microphone, the feedback case | forcing the interleaving in a test |
| Callback put the rest of a sentence back on the queue | a sentence submitted in that instant **played in the middle of the current one** | same |
| A failed Start left everything loaded | ~2.5 GB of models and a running delivery thread stayed resident; the next Start loaded a second copy | reading what happens after "could not open any audio output" |
| `tts.voice` and `tts.speed` never passed to the synthesiser | the configured voice was inert; invisible while it equalled the default | wiring a second voice per language |
| Turkish hotwords applied to every spoken language | Italian or German recognition pulled toward Turkish words | adding Spanish and German |
| One RVC `--pitch` for every language | Spanish and German men would have come out near 80 Hz | measuring each synthesiser voice |
| Voice service timeout 60 s, no pause after one | a hung service = a minute of silence per sentence | reading the client with a hung service in mind |
| `vad.threshold` never reached the segmenter | adjusting it for a noisy hall would have changed nothing | following the value through the code |
| Ollama addressed as `localhost` | **~2 s added to every request** on Windows (IPv6 first) - 2274 ms against 222 | a model that worked in 230 ms taking 2.3 s |
| RVC's memory assumed equal to Seed-VC's | NLLB-1.3B rejected for a budget RVC does not use: 0.7 GB held, not ~3 | measuring the RVC process on its own |
| Speakers on MME | **the translation reached the room choppy** - MME starves while Python is busy: 45 underflows in 8 s, WASAPI none; logged at debug level, so the run log showed nothing | the 2026-10-02 rehearsal, then a silent stream under load |
| Whisper recited its hotword list | "İpek Nur Yıldız Acme Valdastra Friuli" **spoken aloud** 33 times across rehearsals - a quiet microphone, and a recital slow enough to pass the rate check | reading what the room heard against the run logs |
| A second copy could start beside the first | two processes fighting over the microphone and the GPU; the page showed the wrong one | the rehearsal where "everything was empty" |

---

## 6. Why four environments

`chatterbox-tts` pins `torch==2.6.0` exactly. Installing it replaced the CUDA
build with the CPU wheel and broke every stage — caught only because CUDA was
checked immediately afterwards. The pin turned out to be over-strict.

Since then, every risky install is isolated and verified. Seed-VC pins
`torch==2.4.0` and `transformers==4.46.3`; those turned out to be conservative
too, but it has a **genuine** `huggingface_hub` incompatibility:

```
BigVGAN._from_pretrained() missing 2 required keyword-only
arguments: 'proxies' and 'resume_download'
```

Forcing an older hub into `parliamo` would endanger `datasets` and
`transformers`. So conversion runs in its own process behind a socket —
~0.57 s of compute against ~1 ms of loopback, so the split is free.

### Windows obstacles, recorded so nobody rediscovers them

- torch 2.4.0 cannot load `fbgemm.dll`. Use 2.6.0.
- `webrtcvad` needs MSVC build tools. Use `webrtcvad-wheels`, install
  `resemblyzer` with `--no-deps`.
- `SeedVCWrapper(device="cuda")` fails — the library calls `.type` on it. Pass
  `torch.device("cuda")`.
- `datasets` ≥5 decodes audio through `torchcodec`, which needs FFmpeg's
  *shared* libraries; common Windows builds ship only a static exe. FLEURS audio
  is WAV, so it is decoded with `soundfile` instead.
- The Intel UHD iGPU was disabled in MUX mode. After switching to Optimus and
  installing Intel's driver, the NVIDIA baseline fell from ~1400 to ~620 MiB,
  but it still drives both displays.

---

## 7. How to run it

The rehearsal script — what to run, in what order, and what each step should
show — is [10-rehearsal.md](10-rehearsal.md). The short version:

### Check the machine, then the room

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\check_env.py
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\measure_audio_device.py --write-config --label venue
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\verify_feedback_loop.py
```

### The voice service, in its own terminal — pick one

```bash
C:\Users\you\miniconda3\envs\seedvc\python.exe scripts\voice_conversion_server.py
C:\Users\you\Applio\env\python.exe scripts\rvc_server.py --model presenter --port 8766 --f0-method fcpe --index-rate 0 --pitch -8
```

Seed-VC on 8765 for volunteers; RVC on 8766 for the presenter. Only one fits
on the GPU; Start uses whichever is running (the configured port first), and
the pre-flight list says which one answered and on which port.

### The application

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\live_translate.py --ui
```

Open the address it prints. Setup, then Start. `/screen` for the room.
**M** mute · **G** generic voice · **S** Turkish under each line ·
**L** cycles wait-for-pause → sentence ends → chunks.

The consent for the presenter's own voice is in `config/local.yaml`; the
Voices tab carries it. For any other recording the tab asks for the signed
form reference, and `VoiceProfile.validate()` refuses without one.

### Replay a recording (no microphone, reproducible)

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\live_translate.py --file data\insitu\continuous\reading-16k.wav --save-audio runs\replay
```

`--no-clone` for the generic voice · `--converter-port 8766` for RVC ·
`--stream sentence|chunk` for the streaming policies · `--target fur` for
Friulian.

### Measure something

```bash
scripts\measure_audio_device.py      # microphone bandwidth, noise, round-trip; --write-config
scripts\verify_feedback_loop.py      # the gate, on this hardware, without a person
scripts\measure_vram_budget.py       # co-resident VRAM
scripts\bench_asr.py                 # WER, with --manifest for in-situ audio
scripts\bench_mt.py                  # chrF++ and BLEU
scripts\bench_tts.py                 # synthesis latency
scripts\prepare_native_clips.py      # cut a native speaker's recording for the Friulian tab
```

---

## 8. Open items, in priority order

1. **`pipeline.output_latency_ms` must be measured in the venue.** The gate is
   sized from PortAudio's reported latency, which excludes ~430 ms of vendor
   DSP; at 0 the microphone hears the speakers and the system translates its own
   output. This is the only remaining item that can ruin the demonstration
   outright.

   Measuring it is now one command, and it writes the result itself — the manual
   copy step was exactly what got skipped:

   ```bash
   python scripts/measure_audio_device.py --skip-bandwidth --skip-noise --write-config --label venue
   ```

   Measured three times on this laptop's speakers: medians 432, 366, 432 ms with
   individual chirps ranging 229–597. **The path is jittery**, so the tool also
   checks the value it wrote against the worst chirp and says how much margin is
   left — 85 ms on the last run, which holds but is thin. It edits
   `config/local.yaml` as text rather than re-serialising it, because that file
   carries the consent record and a YAML round-trip would delete every comment.

   The number is a property of the room and the speakers. **This laptop's figure
   is not the venue's figure.**
2. **A rehearsal with a live microphone.** The whole chain runs end to end on
   *recorded* speech — `live_translate.py --file`, 5/5 sentences delivered,
   1.21–1.44 s each without cloning — but nobody has spoken into it live.

   **The gate and the acoustic loop no longer need a person**, which was the
   part of this that could not be checked before:

   ```bash
   python scripts/verify_feedback_loop.py
   ```

   It plays recorded speech through the speakers twice, once with the gate off
   and once with it on, and compares what the *pipeline* received. Measured on
   this laptop:

   | | room floor | while playing | after reopen | delivered |
   |---|---|---|---|---|
   | gate OFF | −93.6 dB | −57.6 dB | −111.8 dB | 4.0 s |
   | gate ON | −99.7 dB | −120.0 dB | −126.4 dB | **0.0 s** |

   So the loop is real on this hardware — 36 dB above the room floor, four
   seconds of audio the recogniser would have transcribed — and the gate stops
   **all** of it. The room is back to its floor once the gate reopens, so
   432 + 250 ms is long enough here.

   It runs twice on purpose. "The microphone delivered nothing" is also what
   muted speakers, headphones and a dead microphone look like, so the gate is
   credited for a *difference*, never for a silence. If the gate-off pass shows
   no loop, the run reports that it proved nothing.

   What still needs a person: a voice into the microphone, and the whole thing
   under stage nerves.
3. **Ten minutes read into a phone, for RVC.** The phone recording came
   (2026-09-01) and fixed Seed-VC's quality: 0% gating against 16–50% from the
   laptop, WER 14.00 → 11.41, diffusion steps back down to 4. RVC then trained
   on everything on this machine - 7.3 minutes, some of it the gated laptop
   audio - and that is now the ceiling on the *trained* voice. Ten clean
   minutes into a phone would retrain in an hour. Not blocking; the samples
   delivered on 2026-09-15 are for the presenter's ear.
4. **Friulian: get a speaker to read ten sentences.** Much smaller than it
   looked. NLLB produces real Friulian untouched - `nol è un probleme`,
   `lis lenghis minoritariis a restin` - so no fine-tuning is needed to build
   the segment ([ADR 0010](adr/0010-friulian-baseline.md)). What is missing is
   a native speaker's judgement on about ten prepared sentences. ARLeF is the
   obvious contact. The corpus notes stay in
   [04-adding-a-language.md](04-adding-a-language.md) if fine-tuning is ever
   wanted.
5. **Operator application: done, no rehearsal behind it.**
   `live_translate.py --ui` serves two windows. **`/`** is the operator's, read
   at laptop distance: Live (subtitles, counts, mute, voice), Voices, Friulian,
   Ethics, and Setup with a pre-flight list that names what would go wrong if
   you started now. **`/screen`** is the room's - dark, huge, projected - a
   different problem at a different distance, so deliberately not the same
   page. A late-joining browser is replayed the last four sentences, because a
   blank subtitle area on stage looks exactly like the system having died.

   It starts and stops the pipeline from the page — `PipelineController` builds
   on a worker thread and publishes each stage, because a page that says nothing
   for forty seconds is indistinguishable from one that has crashed. It picks
   input and output devices, and shows which device is *actually* open beside
   the one that was chosen: a device that will not open is replaced by one that
   will, which is right thirty seconds before a talk and wrong to do quietly. It
   records a volunteer from the browser and refuses to without a consent record.

   Keyboard: **M** mutes, **F** hides the operator strip, **S** shows the
   Turkish. Entirely self-contained — no external asset is fetched, because the
   venue has no internet and the talk's claim is that the network can be
   unplugged.

   The Friulian tab plays the pipeline's own renderings and 39 sentences from
   a native speaker - Marco Moroldo, Wikitongues, CC BY-SA 4.0 - as recorded,
   with attribution. The licence covers the recording, not his voice, so the
   folder is never offered as a reference; the room hears real Friulian beside
   what the pipeline makes of it, which is the honest half of that segment.
   **Nothing has been rehearsed with a person in front of it.** The
   rehearsal script is [10-rehearsal.md](10-rehearsal.md).
6. **Ethics segment: built.** `scripts/ethics_demo.py` produces the whole
   thing. The presenter's cloned voice reads the actual fraud script - voice
   fraud aimed at older people is a relative in trouble, not a politician, so a
   deepfaked public figure would teach the wrong lesson. Every clip speaks
   "questa voce è sintetica" at the start, the end, and **a random point
   inside**, in the generic voice rather than the cloned one; the interior
   placement is what stops the clip being made usable by trimming the ends.
   The watermark table is the uncomfortable part and is measured, not asserted:
   Perth recovers Chatterbox's mark perfectly and finds **nothing** in Kokoro +
   Seed-VC, which is what ships here.

   The volunteer flow is built too. `scripts/clone_volunteer.py` records
   fifteen seconds, builds a proper reference, and POSTs it into the running
   pipeline through the operator page - a restart at that moment would end the
   demonstration, whose point is that it takes fifteen seconds. Consent is
   required before recording and checked again on the swap, which is the moment
   most likely to skip it. **`--forget-all` deletes every volunteer recording**,
   which the consent form has promised since it was written and nothing did
   until now. Dropping back to the generic voice needs no consent and is one
   keypress (**G**), because removing a clone is the safe direction.
   Still open: rehearsing any of this with a real volunteer.

### Audit, started 2026-09-23

A full pass over the code, looking for what the tests could not see because
nobody had thought to write them. Each fix below has a test that fails on the
old code.

Fixed:

- **Streaming state leaked across utterances.** The sentence and chunk ledgers
  were cleared only by a commit. When a commit never came — the recogniser
  returned nothing for the final segment, or it was dropped — the next
  utterance was compared against the last one's words, and chunk mode stayed
  silent until the end of the talk. The ledger now resets whenever the
  utterance's start time changes; partials and the commit of one utterance
  carry the same start (11/11 on the reference recording), so the reset never
  fires inside one.
- **The page's handlers blocked its own event loop.** Nineteen calls — start,
  stop, pre-flight, device listing, voice swap and others — ran synchronously
  inside `async` routes. While one ran, `/state` and the subtitle stream
  waited; measured at 1.32 s behind a slow action. They now run on a worker
  thread.
- **A second copy started beside the first.** It took the next free port, and
  both then opened the microphone and loaded models. A second
  `live_translate.py --ui` now finds the first (by asking ports 8770–8773 for
  `/state`) and exits with its address instead.

Fixed in the second pass:

- **Chunk mode's ledger was a word count.** Every reading re-decodes the
  whole utterance, and the commit may segment it differently from the
  partials - "üç te" then "üçte". Sliced by count, that loses the next word
  or repeats one. Spoken words are now aligned to each new reading, with
  length in characters covering what does not align. Honest result: on the
  68 s reference recording this changed nothing - the same 4 words missing
  and 4 extra at 500 ms, all of them the recogniser revising a word the room
  had already heard ("3" then "üç", "sitemin" then "sistemin"), which no
  policy can take back. The fix is for the case the tests construct, which
  that recording happens not to contain.
- **Chunk agreement was punctuation-sensitive**; sentence mode never was.
  "istiyorum" and "istiyorum." flicker between readings, and each flip held a
  chunk back one partial. Comparing without punctuation moved spoken words
  **0.15-0.20 s earlier** on average on the same recording, with the same
  words missing and repeated.
- **The RVC server's checkpoint choice was dead code.** Its filter matched
  every file, so the newest by file time loaded - and the two 300-epoch files
  were written 0.09 s apart. It now picks Applio's `_best_epoch` if there is
  one, else the highest epoch by number; `--weights` overrides. For this
  model both candidates are the same epoch, step and loss, so the voice does
  not change. The `--pitch` help also said the opposite of what was measured
  (RVC keeps the pitch it is given) and is corrected.
- **The conversion service is warmed on speech, not silence.** A fresh Seed-VC
  took 1.12/0.94 s on the first real sentence after a silent warm-up and
  0.88/0.87 s after a spoken one. On RVC the difference was 0.05 s - noise.

Checked and not a defect: the mutable class-level tables in `server.py` are
never mutated, and the clip-quality cache is keyed by path and file time, so
sharing it between instances is correct.

Third pass - research and measurement, written up in
[11-model-survey.md](11-model-survey.md):

- **The latency tables were 0.50 s too high everywhere.** The script that
  summarised the run reports added the commit wait to a total that already
  contained it. Recomputed from the same files: RVC **1.58 s** (not 2.08),
  Seed-VC 2.29 (not 2.79), generic 1.29, subtitles 1.10. Differences between
  configurations were right, so no decision changes. Corrected in every
  document that carried them.
- **RVC holds 0.7 GB, not ~3** (+1.2 GB at the peak of a conversion). That is
  the memory NLLB-1.3B was rejected for. With RVC, 1.3B peaks the machine at
  6942 MiB and costs 0.2 s per sentence (1.92 s against 1.72, same-day rerun);
  read sentence by sentence it fixed three meaning errors and introduced one.
  Recommended with RVC; the Setup picker and pre-flight say where it does not
  fit ([ADR 0013](adr/0013-rvc-frees-memory-for-nllb-1.3b.md)).
- **Whisper's "İzlediğiniz için teşekkür ederim" from a breath is dropped.**
  A reading faster than 12 syllables per second cannot have been spoken
  (`asr/plausibility.py`). WER unchanged; both inventions on the reference
  recording gone.
- **The opening sentence's key word is misheard** - *takdir* for *taklit*.
  A hotword made WER worse (11.41% to 12.08%) and did not fix it; neither did
  a topic prompt. *Kopyalanabildiğini* is recognised every time: a wording
  change for the talk, not a code change.
- **Candidates measured** after the presenter approved the downloads
  ([11-model-survey.md](11-model-survey.md)):
  - **HY-MT1.5-1.8B** - chrF++ 45.30, level with NLLB-600M, behind 1.3B's
    47.11; more fluent in places, ungrammatical in others. Not adopted.
  - **Qwen3-TTS 0.6B** - excellent intelligibility, RTF 2.2-2.4 on this GPU
    (14.5 s for a sentence Kokoro + RVC renders in 0.54 s). Not for the live
    path; possibly for clips prepared before the talk.
  - **WebRTC echo cancellation** - at low volume it removed everything the
    pipeline would hear in the echo, and raised WER from 5% to 15% when
    talking over it. Performance volume left for the rehearsal
    (`scripts/measure_echo_cancellation.py`, step 9b).
- **Two measurement faults found on the way:** the Ollama backend addressed
  `localhost`, which on Windows costs ~2 s per request (ADR 0004's
  TranslateGemma figure carried it; the rejection stands on quality and
  memory), and Ollama's derived template for HY-MT dropped the prompt.
  Both fixed and tested.

### Audit, second round (2026-09-24): the audio path and the controller

Each has a test that fails on the old code, and the playback fixes were run on
the real Realtek output as well.

- **The gate could release just before a sentence played.** `submit()` closed
  the gate, marked playback busy, and only then counted the frames. A sound
  card callback landing in between saw nothing pending, decided the speakers
  had drained and released the gate. Now all four happen under the callback's
  lock.
- **A sentence could play in the middle of another.** The callback put the
  unplayed rest of the current sentence back on the queue every buffer; a
  sentence submitted in that instant went in front of it. The callback now
  keeps the sentence it is playing and never re-queues.
- **A delivery thread outliving `stop()` reopened the speakers.** A stopped
  playback now refuses late sentences.
- **The gate tail grew every time playback opened** - measured latency added
  to a tail that already contained it (680 ms, then 1110). Computed from the
  gate's own base now.
- **A failed Start left everything resident.** The controller now stops the
  half-started pipeline and unloads its models before reporting the failure.
- **A hung voice service would have cost 60 s per sentence.** The timeout
  is now 10 s (slowest conversion ever measured: 4.7 s), and one timeout
  pauses conversion for 30 s - the following sentences go out in the generic
  voice at once. Not connecting (absent service) is kept apart from connecting
  and not answering (hung).
- **A garbled reply dropped the sentence.** Only `OSError` became a
  `ConversionError`; unparseable JSON escaped the fallback and the sentence
  was not spoken at all. Now it is spoken in the generic voice.
- **`vad.threshold` reached nothing.** The segmenter used a fixed 0.5. It now
  takes the configured value (still 0.5 by default, so nothing changes today).

### Languages (2026-09-25)

[ADR 0014](adr/0014-spanish-german-turkish-and-calabrese.md). Spanish and
German are spoken targets, Turkish too (questions from the room heard in
Turkish, 1.25 s). Piper speaks German and Turkish on the CPU (8 threads: a
German sentence in ~1.3 s instead of 2-3 s with the default); the GPU would be
0.5 s but means replacing onnxruntime - left for the presenter to decide. Each
sentence is converted with the pitch shift for the voice that spoke it
(output 123-137 Hz across three languages against the presenter's 139).
Calabrese: no model translates it; NLLB's Sicilian, the only proxy, is Corsican
in 148-174 of 200 sentences.

### 2026-10-02 (after the rehearsal): P, and the echo measured

- **P pauses listening.** The gate covers only the system's own voice; the
  deck's videos, applause and questions reach the microphone through the
  room's speakers and were subtitled and spoken over. M stops only the voice.
  P drops the microphone's blocks after its meter (the operator sees what is
  ignored) and before the gate (its counters stay about the system's voice),
  commits the sentence said before the press at once instead of holding it
  until resume, and survives Stop/Start. Red banner, header button, mic pill
  "paused - not listening".
- **"held — speaking" on the gate means the system is speaking**, not the
  presenter: the microphone is deliberately deaf and the green meter is the
  raw level before the gate. The presenter read it the other way round.
- **`measure_room_echo.py`** measures what the existing chirp tool could
  not: how long the hall keeps the last word audible to the voice detector,
  against the moment the gate reopens. `--write-config` raises
  `half_duplex_tail_ms` when the margin is under 150 ms; Start now re-reads
  both gate values, so the page need not be restarted. Not yet run in the
  hall. Run end to end on this laptop with the speakers turned down (-72 dBFS
  at the microphone, never speech) - proof that it runs, not a measurement.
- **Measured in the hall, 13:15, empty** (headset microphone on the Realtek
  jack, Dante speakers on WASAPI, PA at talk volume):
  - **Round trip 306 ms** (304-309 over five chirps). The config said 38, so
    the gate reopened 288 ms after the last sample - *before* the last sound
    reached the microphone. Now `output_latency_ms: 306`; the gate holds 556 ms.
  - **Echo:** three Italian sentences reached the headset microphone at -20,
    -23 and -28 dBFS peak - as loud as the presenter's own speech (-19). With
    the 556 ms gate the detector's last speech frame came 512, 2816 and 1696
    ms before reopening; nothing after. Covered; `half_duplex_tail_ms` stays
    250.
  - **Speakers:** a silent stream under load on the Dante output, MME 39
    underflows in 8 s, WASAPI none - the rehearsal's choppy sound, reproduced.
  - **Headset microphone gave pure digital zeros** until it was unplugged and
    pushed back in; Windows showed it active and unmuted throughout. After:
    speech peaks -19 dBFS, silence gated to digital zero by the driver (not
    within speech), and Whisper read "1-2-3-4-5" right. Check the meter after
    plugging it in.
  - **"Something happens with long sentences"** (the presenter's words, the
    symptom not pinned down). The presenter disabled the headset's own
    speaker endpoint ("Headphones (Realtek)") in Windows' sound settings and
    it looked right afterwards; the microphone still delivered speech at -22
    dBFS peak with no zeros inside speech. The app never plays to that
    endpoint (output verified on Dante WASAPI), so the mechanism is unknown -
    possibly the Realtek driver's headset processing. One real long-sentence
    fault was in the log: Whisper wrote a long sentence without punctuation,
    the splitter had nothing to split on, and NLLB dropped the last clause
    ("umarım keyif alırsınız ve beğenirsiniz"). With punctuation, twice, it
    was complete. Not fixed yet.

### 2026-10-02 (rehearsal, at the venue's Dante system)

- **The translation reached the room choppy.** The speakers were chosen on
  MME ("Speakers (Dante USB I/O Module) - MME - 90 ms"). Measured on a silent
  stream with the interpreter kept busy, as the pipeline keeps it: MME 45
  underflows in 8 s, its callback run 150 times instead of 610; WASAPI, with a
  22 ms buffer, none. Not an encoding problem - the audio is right, it is
  delivered late. Choose the speakers' WASAPI entry. **Fixed the same day:**
  the speaker picker lists WASAPI first and marks MME/DirectSound "may stutter
  while translating"; pre-flight has a Speakers row that warns on them; and
  underflows while a sentence plays are counted and logged as a warning
  ("the speakers ran dry N time(s) on ..."), at most one line per 10 s -
  before, they went to debug level and the run log could not show the gaps.
- **The microphone was too quiet, not too sensitive.** Speech in the saved
  segments ran -31 to -57 dBFS (mostly about -40), with a -70 to -90 dBFS
  floor; the voice detector was sure it was speech (0.78-0.92) and Whisper
  could not read it. It then produced the hotword list itself ("İpek Nur
  Yıldız Acme Valdastra Friuli Hong Kong") on several longer segments, too
  slowly to trip the rate filter - and those were spoken aloud. **Fixed the
  same day:** a transcript with four or more hotwords in the list's own order,
  and at least 80% hotwords, is dropped as a recital. Over every transcript in
  `runs/` (7,152) it drops 76 - all recitals ("İpek Nur Yıldız Acme" 13
  times, "... Valdastra Friuli" 33) - and no real sentence: "Ben İpek Nur
  Yıldız, Acme'de çalışıyorum" and "İpek Nur Yıldız." alone are kept. About
  five recitals padded with stray words ("... Valdastra Farklı Farklı
  Farklı") still pass; the cure for all of them is a louder microphone.
- **Pre-flight described Windows' default microphone** whatever was picked.
  It now checks the chosen input and output, and says so when the chosen one
  is not connected and Windows' default would open instead.
- **OmniVoice took 3.3-3.7 s a sentence**, against 1.0-1.2 s measured at
  home, with the invented "bambino" voice and speed 0.8. Not explained yet;
  the laptop on battery (GPU power limit) is the first thing to rule out.

### 2026-09-27 (rehearsal)

- **The microphone chosen was not the microphone used** (2026-10-02). The
  jack microphone was picked on its WDM-KS entry ("10 ms"), which never opens
  on this laptop (-9996), and the fallback went to the laptop's own array -
  because the same jack is "Realtek HD Audio Mic input" on WDM-KS and
  "Microphone (Realtek(R) Audio)" elsewhere, so it was not recognised as the
  same device. Now: WDM-KS entries are listed last and cannot be picked; a
  choice that will not open falls back to Windows' own default device before
  anything else; and the page sends "name@host API" instead of an index -
  Windows renumbered the devices within minutes, index 26 going from the jack
  microphone to a "PC Speaker" loopback.
- **A real sentence lost to a decoder loop.** Whisper said "Brad Pitt'in
  annesinden sonra da" 24 times for 2.7 s of speech that said it once, and the
  repetition filter dropped the whole segment. Now the loop is cut to one and
  kept if what remains looks like speech (3+ words, 1 s+, 2-12 syllables/s);
  "Friulian" x16 out of a breath still goes. Replayed on the saved segment:
  the sentence comes back.
- **"Altyazı M.K."** - a subtitler's signature from Whisper's training data,
  from a 0.54 s breath - was too short for the rate check and would have been
  spoken as "Sottotitoli M.K.". Whole-transcript subtitle credits are now
  dropped; a sentence that mentions subtitles is not.
- **The cloned voice spoke fast.** OmniVoice copies the reference's pace: 6.2
  syllables/s against Kokoro's 5.3. `tts.conversion.speed` (0.85 in local.yaml,
  and a Setup-tab choice applied at once) brought it to 5.3 - and, measured on
  40 renders, garbled 4 short sentences instead of 9.

### 2026-09-26

- **Studio: derive, delete, speak from a description.** A design instruction
  added to a clone changed nothing measurable (pitch within 4 Hz for "very
  high" and "very low") and once inserted a negation; moving the reference's
  pitch did move the clone (-6/+4/+8 in, -5.1/+4.0/+8.3 out). So "invent from
  a recording" derives a new voice by pitch and tempo. Voices other than the
  presenter's can be deleted with their clips. The advanced settings open by
  default - closed, they looked empty to the presenter.
- **The Studio has every OmniVoice setting**: sliders for all ten generation
  settings and two switches, from one table the service also checks; seeds
  (the same seed gave a byte-identical clip), the model's own voice, 646
  languages, non-verbal tags, one clip per line, accents, upload of any
  recording the browser decodes (iPhone memos included), and the reference's
  transcript prepared and corrected on the page.
- **The Studio tab**: typed sentences spoken through OmniVoice in a consented
  or an invented voice, kept as clips, played through the pipeline's gated
  playback when it runs ([ADR 0015](adr/0015-omnivoice.md)). Invented voices
  come from OmniVoice's attribute list only; public figures and cartoon
  characters are not offered. Building it found that **volunteers' folders
  were not git-ignored** - names, consent records, recordings; now they are.
- **OmniVoice added as a third voice service** (port 8767,
  [ADR 0015](adr/0015-omnivoice.md)). It speaks the translated sentence in the
  reference voice instead of converting Kokoro's, so the rhythm is not the
  synthesiser's; a volunteer is a new reference, not a new service. Measured:
  2.50-2.55 s mean lag end to end (RVC 1.6-1.7, Seed-VC 2.29), zero failures,
  6.6 GB machine peak with NLLB 600M. Three things had to be found first: a
  reference cut mid-phrase lost the first word of every sentence (cut on
  sentence boundaries: WER 25.6% → 7-9%); encoding a reference on the GPU
  peaked at 5.6-12.5 GB (now on the CPU); digits were misread (spelled out:
  5.6% → 1.7%). Weights are CC-BY-NC, not Apache-2.0 as resellers say.
- **The hotword list is the talk's, not the old script's.** The list still
  named the earlier talk's tools ("Chatterbox", "Silero", "Interreg") and
  Whisper produced them from breaths ("Friulian Friulian ..." in rehearsal).
  Rewritten for this talk's names and measured: the presenter's 35 in-situ
  sentences 12.12% WER (old list 12.95, none 13.50); 16 sentences with the
  talk's names 8.80% (old 10.40, none 12.00) - "Crosetto" and "OmniVoice"
  right only with the new list.
- **The cloned voice played at half speed an octave down.** The voice
  services answer at their own rate (RVC 48 kHz, Seed-VC 22.05) and playback,
  opened at Kokoro's 24 kHz, was never told. RVC therefore sounded slowed and
  robotic through the speakers - while every saved file, and every measurement
  made from one, was right. `AudioPlayback.submit()` now takes the rate and
  converts; the translator passes it. On the real output: 2 s of RVC audio
  played for 4.17 s before, 2.12 s after. It also halves how long the
  microphone stays gated after a cloned sentence.
- **Start finds the voice service that is running.** It was pinned to
  `tts.conversion.port`; the volunteer swap (RVC off, Seed-VC on) would have
  left it on the closed port. Now the configured port, then the other known one
  (`find_voice_service`); pre-flight shows the one Start will use.
- **The pitch is measured on two plain sentences**, not a greeting: stable to
  0.1 semitone for Kokoro, within ~1 for Piper, and -7.5 for if_sara against
  the -8 found by ear.
- **The deck is the association's**: their flyer's colours and type,
  their logo, an introduction slide and a Friulian demo slide.

### Closed since the last update

- **The operator interface**, from the Claude Design handoff: five tabs,
  start/stop, device, language, model and tempo pickers, voice list with
  consent, volunteer recording, the audience screen. Verified in a browser
  against real data, which found three bugs the markup hid.
- **The audio "hardware fault"** was WASAPI needing a COM apartment on the
  worker thread ([ADR 0011](adr/0011-audio-device-opening.md)).
- **Output latency measured and written by one command**; the gate proven on
  hardware by `verify_feedback_loop.py` - 36 dB loop with it off, nothing with
  it on.
- **The presenter's voice was refused by the page** - its consent sat in the
  config and the list never attached it. Fixed. Pre-flight now names the
  voice service that answered, or says none did.
- **Streaming**, three positions - wait for pause, sentence ends, chunks -
  measured; local agreement added after the first measurement spoke 24
  sentences where the speaker said 11.
- **RVC** trained on the presenter's recordings and served beside Seed-VC:
  1.58 s mean against 2.29 ([ADR 0012](adr/0012-rvc-beside-seedvc.md)).
- **A native Friulian speaker can be heard**: 39 sentences from a CC BY-SA
  Wikitongues recording, played as recorded, never used as a reference.
- **Two silent bugs found by measuring pitch and reading logs**: `tts.voice`
  was ignored for every language; the sanitiser cut "La traduzione" off the
  front of real sentences.

- **NLLB-1.3B measured.** Better on every quality axis (chrF++ 47.11 against
  45.38) and rejected on memory: +1118 MiB co-resident leaves 359–700 MiB of
  headroom against a desktop baseline that varies by 381 MiB on its own. On CPU
  it costs 3023 ms p95, more than the entire budget. It stays for offline work.
  [ADR 0007](adr/0007-nllb-size-and-sentence-splitting.md).
- **The clause-dropping bug is fixed** by translating sentence by sentence,
  which also turned out to be *faster* than translating the whole segment.
- **The cloned voice was fixed and re-costed.** The reference recommended for
  the first test was 49.5% digital silence; `scripts/build_voice_reference.py`
  now assembles one properly. Four diffusion steps was then rejected by ear and
  raised to 15, which puts conversion at 1.64 s instead of 0.92 s.
- **Two conversion servers can bind the same port on Windows**, holding 5.8 GB
  between them and making every conversion 4.8–9.8× slower. The server now
  refuses to start when the port is taken.
- **Five missing documents written** — architecture, hardware budget, adding a
  language, ethics and consent, model licences. All five were linked from the
  README and none of them existed.

---

## 9. Working agreement

Recorded because it shaped every decision above.

- **Measure, do not estimate.** Every number in this document came from running
  something. Where a number is still an estimate, it says so.
- **Test before moving on.** 819 tests. A stage is not done until its test
  passes.
- **Be blunt.** The user asked explicitly for wrong things to be called wrong.
  Several decisions here reverse an earlier one because a measurement disagreed.
- **Isolate risky installs**, and verify `torch.cuda.is_available()` afterwards.
- **Never clone a voice without a consent record.** Enforced in code, not
  policy.

### The lesson that cost the most

*Measurement inside a chosen architecture cannot detect that the architecture is
wrong.* Kokoro was in the first model survey and was dismissed in one line —
"very fast but no voice cloning" — on an unexamined assumption that one
component must both generate speech and carry speaker identity. Weeks of careful
measurement followed, all of it inside that assumption. The correction came from
finding the user's own earlier prototype, which had already split the two.
