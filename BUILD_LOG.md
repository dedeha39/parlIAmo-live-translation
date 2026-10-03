# Build log

> For the current state rather than the history, read
> **[docs/00-state.md](docs/00-state.md)**. This file is the chronological
> record of how it got there and what went wrong on the way.


Chronological record of what was built, what was tested, and what the tests
revealed. Findings that changed the design are marked **FINDING**; bugs caught
before they reached the pipeline are marked **BUG**.

Nothing is marked done here unless a test was run and passed.

---

## 2026-08-29 — Phase 0: environment

### Machine survey

| Item | Result |
|---|---|
| GPU | RTX 4070 Laptop, 8188 MiB, cc 8.9, driver 610.47 |
| CPU / RAM | i9-13900HX, 32 logical cores / 31.7 GB |
| Python | miniconda 25.3.1, base 3.13.2 |
| ffmpeg / git | 8.0.1 / 2.53.0 present |
| Disk | initially 19.5 GB free on C: — **blocker**, cleared by the user to 174 GB |

**FINDING — idle VRAM is higher than planned.** With a normal desktop session
(browser, Spotify, WhatsApp, Notion, Claude) the GPU already held **1465–1712
MiB** before any model loaded. The plan assumed 1000 MiB. Usable budget is
therefore ~6.3 GB, not 7.2 GB. `check_env.py` now warns above 1200 MiB, and the
stage-day checklist has to include closing these applications.

### Environment

Created conda env `parliamo` on **Python 3.12.14**, deliberately not 3.13:
CTranslate2, faster-whisper and Chatterbox wheels are reliably available for
3.12, and a missing wheel two weeks before a deadline is not a risk worth taking.

### Built

- `src/parliamo/paths.py` — repository-root discovery
- `src/parliamo/config.py` — layered YAML config (default → local → CLI),
  validated into dataclasses, unknown keys rejected at startup
- `src/parliamo/logging_setup.py` — console + JSONL logging, `LatencyTracer`
  for per-stage timings
- `scripts/check_env.py` — the Phase 0 gate

### Tested

```
pytest -q   →  21 passed
python scripts/check_env.py  →  exit 0, all hard requirements met
```

Config rejects typos (`asr.langauge` fails loudly), the Turkish/Italian defaults
are asserted, and `pipeline.half_duplex` is asserted to default to `true` —
that flag defaulting to `false` would be a stage-killing bug.

---

## 2026-08-29 — Phase 0: audio I/O

### Built

- `src/parliamo/audio/devices.py` — enumeration, host-API-aware resolution,
  Bluetooth heuristic
- `src/parliamo/audio/gate.py` — the half-duplex feedback gate
- `src/parliamo/audio/capture.py` — 16 kHz mono float32 capture with soxr
  resampling and drop-oldest backpressure
- `src/parliamo/audio/playback.py` — queued playback that owns the gate
- `scripts/list_audio_devices.py`, `scripts/measure_audio_device.py`

### FINDING — the system default microphone is 45× slower than the same hardware on WASAPI

`list_audio_devices.py` on this machine:

| Device | Host API | Reported latency |
|---|---|---|
| Microphone Array (Intel Smart Sound) | MME **(system default)** | 90.0 ms |
| Microphone Array (Intel Smart Sound) | DirectSound | 120.0 ms |
| Microphone Array (Intel Smart Sound) | **WASAPI** | **2.0 ms** |
| Speakers (Realtek) | MME (system default) | 90.0 ms |
| Speakers (Realtek) | **WASAPI** | **3.0 ms** |

**BUG (fixed).** `resolve_device(None, ...)` returned the system default, i.e.
the 90 ms MME clone — donating ~88 ms to every segment before any model ran.
Fixed by `_fastest_sibling()`, which re-points the default at the same hardware
on the most preferred host API. Complicated by MME truncating device names to 31
characters, so sibling matching is prefix-based with a length guard (otherwise
`"Headphones"` would match `"Headphones 2 (Realtek …)"`).

After the fix: `IN 2.0 ms / OUT 3.0 ms`.

### BUG — the first latency measurement was meaningless

The initial round-trip probe used two independent streams (`AudioCapture` +
`AudioPlayback`). Results: 356–878 ms, spread 521 ms. Independent streams share
no clock, so the correlation peak measured an arbitrary buffer offset rather
than the acoustic delay.

Rewritten to use a **duplex** stream (`sounddevice.playrec`), where both
directions are driven from one callback. Results became repeatable:

```
430.3 / 431.3 / 430.1 / 431.0 / 598.4 / 431.3 ms
median 431.2 ms, five of six within 1.2 ms
```

### FINDING — the laptop's own audio path adds ~430 ms, and the output device is most of it

The measurement is stable, so 431 ms is a real fixed delay, not jitter. Two
follow-up probes isolated where it comes from:

| Path | Median round-trip |
|---|---|
| mic ← Realtek speakers (WASAPI) | **428.2 ms** |
| mic ← HDMI monitor (WASAPI) | **258.3 ms** |
| Changing blocksize 4× (48 kHz) | 437.3 → 423.2 ms (14 ms) |

Blocksize barely moves it, so it is not buffering. Same microphone, different
output: **170 ms attributable to the output device alone** — the Realtek/Nahimic
DSP effect chain. A further ~258 ms is common to both paths (Intel Smart Sound
input DSP, plus whatever PortAudio's cross-device duplex adapter costs; these
two are not yet separated).

Consequences:

1. The built-in speaker path is not viable for the stage. Budget for an external
   USB interface or a direct feed to the venue PA, and re-measure there.
2. The absolute number carries an unseparated constant, so the tool reports it
   as a **comparative** metric. Device-to-device differences are trustworthy;
   the floor is not.

### BUG (fixed) — the gate was sized from the wrong latency

Found by the measurement above, not by reading code. `HalfDuplexGate` reopened
`tail_ms` (250 ms) after the software buffer drained. But with ~430 ms of
hardware delay, audio is still leaving the speakers long after that buffer is
empty — so the microphone reopened mid-sentence and heard the system's own
output. Precisely the loop the gate exists to prevent.

Fixed: `AudioPlayback` now extends the gate tail to
`max(measured_output_latency, reported_latency) + tail_ms`, and logs a warning
while `pipeline.output_latency_ms` is still 0. New config key documents that it
must be measured per venue.

### Tested

```
pytest -q                          →  66 passed
pytest tests/test_audio_io.py -v   →   8 passed (real hardware)
```

The load-bearing test is
`test_capture_discards_audio_while_gate_is_closed`: it starts capture and
playback against real devices, confirms blocks are muted while audio plays, and
confirms capture resumes once the gate reopens. That is the anti-feedback
guarantee, verified end to end rather than asserted in a comment.

### Known cosmetic issue

`sounddevice` 0.5.6 emits a `DeprecationWarning` under NumPy 2.5
(`data.shape = -1, channels`). Upstream, harmless, 300+ occurrences per audio
test run. Not worth pinning NumPy down for.

---

## 2026-08-29 — first real device measurements

Four runs against the built-in microphone array and the AirPods Pro 2. They
produced one tool bug and two hardware findings.

### BUG (fixed) — the bandwidth probe was measuring the talker, not the microphone

The probe reported the built-in 48 kHz microphone array as
`rolloff_hz: 2226.6`, `verdict: narrowband, unusable for accurate ASR`.

That number was wrong, and the method was wrong. It took the spectral peak
(above 150 Hz) and called the roll-off the highest frequency still within 25 dB
of it. Speech falls naturally at roughly **-6 to -12 dB per octave**, so by 8 kHz
it is already 30-40 dB below its 300 Hz peak. A perfect full-band microphone
recording ordinary speech scores 2-3 kHz under that rule. The tool was measuring
the spectral tilt of the human voice.

Replaced with **cliff detection**. A codec cutoff is a discontinuity — tens of dB
across a fraction of an octave; the threshold is -55 dB/octave. Speech decay
never gets near that. The Nyquist ceiling is now reported separately, because a
16 kHz capture cannot reach the 8 kHz band whatever the microphone does, and a
device can hit both limits at once.

`analyse_bandwidth()` was extracted as a pure function so it is tested against
synthetic signals of known bandwidth rather than whatever hardware is plugged in:
speech-shaped noise at tilts from -3 to -15 dB/octave must yield *no* cliff, and
brick walls at 3400 / 7000 / 11000 Hz must be located within 25%.

### FINDING — the AirPods verdict needs no measurement; Windows states it

With the AirPods connected, they appear as **two separate devices**:

| idx | Name | Direction | Channels | Sample rate |
|---|---|---|---|---|
| 14 | `Headphones (AirPods Pro)` | output | 2 | **48000 Hz** (A2DP) |
| 18 | `Headset (AirPods Pro)` | input | **1** | **16000 Hz** (HFP) |

16 kHz native means a Nyquist ceiling of 8 kHz — the exact edge of the band
Whisper uses, before the HFP codec cuts further below it. No configuration
changes this.

The round-trip probe then failed with `no sample rate accepted by both devices in
duplex mode`. That failure *is* the profile conflict: Windows cannot run the HFP
microphone and the A2DP output at the same time, because they are the same radio
link in two mutually exclusive modes. Opening the microphone drags playback down
with it.

Connecting the AirPods also shifted every device index (built-in mic array 12 →
17), which is why config uses name substrings.

### FINDING — upstream noise gating is active, and input levels are far too low

Across all four runs:

| Run | Speech peak | Speech rms | Noise floor |
|---|---|---|---|
| built-in | 0.053 | -42.5 dBFS | **-103.7 dBFS** |
| AirPods | 0.005 | -66.4 dBFS | **-115.9 dBFS** |

A real microphone in a real room sits around -70 to -45 dBFS when nobody speaks.
**-104 dBFS is digital silence**: the driver is zeroing non-speech. Windows audio
enhancements and Intel Smart Sound noise suppression both do this.

Two consequences, neither cosmetic:

1. A gate upstream of ours chews the onset off words, which is where Turkish
   carries a good deal of its consonant information.
2. The measured noise floor becomes useless for setting a VAD threshold — it
   describes the gate, not the room.

Speech peaking at 0.053 (-25 dBFS) is also 15-20 dB quieter than it should be.

The probe now flags a floor below -90 dBFS as gating rather than reporting it as
an excellent noise figure, and flags peaks below 0.05 as too quiet.

### Still unresolved at this point

The bandwidth of the built-in microphone and of the AirPods had **not** actually
been established — every run so far was either silent or gated.

---

## 2026-08-30 — the detector contradicted itself, three times over

A live run reported the built-in microphone as `usable to 15633 Hz, effective
source 32000 Hz`. Re-analysing **the same recording** reported `8742 Hz,
effective source 16000 Hz`. A measurement tool that gives two answers for one
file cannot settle anything, so the tool got fixed before the hardware question
was touched again.

### BUG — dB smoothing was zero-padded

`np.convolve(db, kernel, mode="same")` pads with zeros. These are dB values
relative to the spectral peak, so **every entry is negative and zero means
"peak level"**. Both ends of the spectrum were lifted into a fake plateau — and
the top end is exactly where the cutoff search looks. The detector was reporting
the top of the frequency axis (24000 Hz) as the cutoff whenever the search
threshold was permissive enough to reach it.

Fixed by padding with `mode="edge"` before convolving.

### BUG — "steepest single point" is not a stable estimator

The cutoff was located as `argmin(slope)`. A recording containing two comparable
transitions — the real codec edge and the descent into the quantisation floor —
flips between them on floating-point noise. That is the direct cause of
15633 vs 8742 for one file.

Replaced with a better-posed question: *is there a noise floor up there, and if
so where does content stop?* A noise floor is **flat** (dither is spectrally
white; measured tilt -0.2 to -1.6 dB/octave) and **deep**. Speech-shaped decay
is neither, measuring -9 to -15 dB/octave in the same band. The two are cleanly
separable at 5 dB/octave.

The cutoff threshold is anchored to the in-band level (`ref - 35 dB`), not to
the floor. Anchoring near a floor 140 dB down lands deep in the window's leakage
skirt and overshot the true edge by up to 55%.

Accuracy against synthetic brick walls after the fix:

| True cutoff | Detected | Error |
|---|---|---|
| 3400 Hz | 3421.9 Hz | 0.6% |
| 7000 Hz | 6984.4 Hz | 0.2% |
| 11000 Hz | 10968.8 Hz | 0.3% |
| 7840 Hz (16k source) | 7828.1 Hz | 0.2% |
| 3920 Hz (8k source) | 3937.5 Hz | 0.4% |

Full-band speech at tilts of -3, -9 and -15 dB/octave yields no cutoff at all,
which was the original requirement.

### BUG — recordings were saved lossy

`sf.write(path, signal, rate)` defaults to **PCM_16** for WAV. The saved file
therefore carried a dither floor near -96 dBFS, sitting right where the detector
looks for a codec floor — so re-analysis genuinely saw different data from the
live run. Now written as `subtype="FLOAT"`, with a test asserting the live and
round-tripped verdicts match.

### FINDING — the 16 kHz path is the laptop's own DSP, not the AirPods

With the detector stabilised, both recordings agree:

| Recording | Cutoff | Effective source rate |
|---|---|---|
| AirPods connected | 9047 Hz | **16000 Hz** |
| AirPods off, Bluetooth off | 8695 Hz | **16000 Hz** |

The wall survives disconnecting Bluetooth entirely, so **Intel Smart Sound
Technology is running its voice pipeline at 16 kHz** and upsampling to the
48 kHz endpoint it advertises.

This is not disqualifying. Whisper's own native rate is 16 kHz, and content
reaches ~8.7 kHz, which covers the band the model consumes. The coverage
threshold was moved from a strict 8000 Hz to 7600 Hz for exactly this reason:
every real 16 kHz chain filters at 7.6-7.9 kHz, so the strict test failed the
format Whisper was trained on.

### What is still wrong with the built-in path

| | |
|---|---|
| Round-trip | 429-479 ms, jitter 82-123 ms between chirps |
| Noise floor | -86 dBFS (improved from -104 after disabling enhancements, still processed) |
| Level | peak 0.16, about 10 dB lower than it should be |

The bandwidth question is settled and the answer is "adequate". The latency
question is not, and it points the same way as before: an external interface or
a direct PA feed, measured in the actual venue.

---

---

## 2026-08-30 — Phase 1: Turkish ASR bake-off

Built the ASR backend layer, Turkish-aware evaluation, and the benchmark. Ran
four models over 200 FLEURS Turkish utterances (41.9 minutes). Two bugs surfaced
by running it, both in the measurement rather than the models.

### BUG — Python's lowercase merges two different Turkish letters

Turkish pairs its i-letters differently from every other Latin alphabet:
dotted `i` ↔ `İ`, dotless `ı` ↔ `I`. `str.lower()` follows the default Unicode
mapping and turns `I` into `i`, so `IRAK` (Iraq) becomes `irak` instead of
`ırak` (far). In a WER comparison that invents substitution errors in the
reference and hides them in the hypothesis, in opposite directions. The
normaliser applies the Turkish mapping before case folding.

### BUG — the thousands separator corrupted the reference text

Turkish groups thousands with a full stop: `40.000` is forty thousand. The
number speller matched a bare `\d+`, so it saw `40` and `000` and produced
"kırk sıfır" instead of "kırk bin". This was corrupting the reference every
model was being scored against — visible in the first run's output as
`kırk sıfırin biraz altında`. Now handles grouped integers and decimal commas
(`3,14` → "üç virgül on dört").

### FINDING — repetition loops, and why temperature fallback is the wrong fix

The first run produced this against a 4.2-second utterance:

```
bu videonun ve videonun ve videonun ve videonun ve videonun ve ...
```

The phrase `videonun ve` repeated **55 times**. That one utterance scored 1009%
WER and dragged the aggregate from 50.76% to **127.27%**.

Whisper's built-in remedy is temperature fallback: retry the segment at
increasing temperature until the output looks sane. It works, but it makes both
output *and latency* nondeterministic — a segment can silently cost five times
its decode budget. On stage an unpredictable multi-second stall is worse than a
dropped sentence, so greedy decoding stays and the loop is caught afterwards
instead, using gzip compression ratio plus consecutive-phrase counting.

The threshold is **3.2**, not Whisper's 2.4. Whisper's value is tuned on
English; Turkish shares long suffixes across words (`-lerini`, `-makta`,
`-acağını`) and compresses better at baseline, so 2.4 flags ordinary sentences.
The margin is asserted in tests against real FLEURS Turkish references rather
than assumed.

With the guard on, the same run went from **127.27% → 50.76% WER** for `tiny`.
Across all four models and 800 transcriptions it fired **twice** — it is not
over-triggering.

### Results

200 FLEURS Turkish utterances, `int8_float16`, greedy, RTX 4070 Laptop:

| Model | WER % | CER % | RTF | VRAM MiB | Load s |
|---|---|---|---|---|---|
| large-v3 | **5.59** | **1.31** | 0.075 | 1959 | 269.8 |
| **large-v3-turbo** | 6.15 | 2.02 | **0.024** | **950** | 167.6 |
| medium | 8.27 | 2.07 | 0.053 | 952 | 131.1 |
| small | 14.59 | 3.57 | 0.023 | 371 | 54.3 |

**Chose large-v3-turbo** — see [ADR 0002](docs/adr/0002-asr-model.md). large-v3
is 0.56 points better and 1009 MiB more expensive, which the 8 GB budget does
not have; it does not fit alongside translation and synthesis. `medium` is
strictly dominated by turbo on every axis and is dropped.

Two things worth carrying forward:

- **The ASR budget line drops from a planned 1600 MiB to 950 MiB**, returning
  650 MiB to the rest of the pipeline.
- **turbo mishears fewer words than large-v3** (146 substitutions vs 156); its
  deficit is entirely deletions and insertions at segment boundaries. For a
  pipeline that feeds a translation model, that is the better failure mode.

### Normalisation is worth 10.6 points

Raw WER runs 10.0–10.8 points above normalised WER for every model. That gap is
punctuation, casing and digit formatting — not recognition. Any comparison
against published numbers has to say which of the two it is quoting.

### Incidental

`datasets` 5.x delegates audio decoding to `torchcodec`, which needs FFmpeg's
*shared* libraries; the common Windows FFmpeg builds ship a static executable
and no DLLs. FLEURS audio is plain WAV, so it is decoded with `soundfile`
instead — one fragile dependency removed.

---

---

## 2026-08-30 — Phase 1: VAD and the segmentation policy

Silero VAD 6.2.1, which wants exactly 512 samples at 16 kHz — the same 32 ms
block the capture layer already produces.

The model and the policy are deliberately separate classes. `SileroVAD` turns a
block into a probability; `SpeechSegmenter` decides where sentences start and
end given a stream of those probabilities. The segmenter is pure logic, so the
21 tests covering it run without a model or a sound card and are fully
deterministic — which matters, because this is where end-to-end latency is
actually decided.

### BUG (fixed) — the pre-roll was too short to do its job

Written to prevent onset clipping, and it did not prevent it.

Speech is only *confirmed* after `min_speech_ms` (250 ms) of it has gone past.
The ring buffer holding audio from before confirmation was sized at
`speech_pad_ms` (120 ms). So the emitted segment began at

```
confirmation − 120 ms  =  speech_start + 250 − 120  =  speech_start + 130 ms
```

**130 ms was being cut off the front of every utterance** — the opening
consonant, which in Turkish is frequently the whole difference between words
(`kar`/`var`, `ürün`/`gürün`). Compounding with the laptop's own DSP noise gate,
which already clips onsets.

Fixed by sizing the buffer at `min_speech_ms + speech_pad_ms`. Verified against
real speech: the detector first sees speech at 3.776 s and the segment now
starts at 3.648 s — 128 ms of genuine lead-in, where before the fix it started
at 3.904 s, i.e. 128 ms *late*.

### A test assumption that was wrong, and worth recording

The first version of the onset test assumed the FLEURS clip began speaking at
sample 0, so it prepended 1 s of silence and asserted the segment started before
the 1 s mark. It failed at 3.648 s.

The clip has **~2.8 s of its own leading silence**. The code was right and the
test was wrong. Rewritten to *measure* the onset with the VAD rather than infer
it from how much padding the test added — an assumption about test data is still
an assumption.

### Latency arithmetic, asserted rather than commented

`min_silence_ms` is added to every segment's delay: the audience waits that long
after the speaker stops before the recogniser is even handed audio. A test now
drives the segmenter at 200/700/1200 ms and asserts it closes on exactly the
expected block, so the cost stays visible in the suite instead of buried in a
config comment.

### Measured

- VAD inference: well under 5 ms per 32 ms block on CPU, real-time factor < 0.1.
  It runs on every block ahead of three larger models, so it stays on CPU — not
  worth VRAM.
- Three real utterances separated by 1.2 s pauses segment back into exactly
  three, none truncated.

216 tests passing.

---

---

## 2026-08-30 — Phase 1: the first end-to-end slice

`LiveTranscriber` wires capture → VAD → recognition across three threads: the
PortAudio callback (must never block), the segmenter (cheap, must keep exact
pace with real time), and the recogniser (200–800 ms bursts, so it cannot share
a thread with the segmenter without dropping audio while it works).

### Offline run over a 59 s fixture

Four FLEURS utterances separated by 1.1 s pauses, through the real VAD and the
real recogniser:

| | |
|---|---|
| Utterances detected | 4 of 4 |
| ASR per utterance | 240–315 ms |
| Overall RTF | 0.031 |

Quality confirms the predicted weak point — **proper nouns**: `Meşhed` → `Meşet`,
`Schlegel` → `Şilegel`. And one genuine content-word error, `Romantizm` →
`Avantizm`, which is the dangerous kind: it changes meaning silently rather than
looking obviously wrong.

### BUG — the pipeline was discarding 53% of the microphone input

The live path reported **174 blocks in 12 seconds**, where 32 ms blocks should
give ~375. No audio was being lost by the capture layer: measured directly, it
delivered 5.98 s of samples in 6.03 s of wall time, exactly right.

The loss was downstream. `soxr`'s streaming resampler does not emit a fixed
chunk size when converting 48 kHz to 16 kHz — it buffers internally and returned
**1100 samples** per chunk on this machine, where the arithmetic suggests 512.
Silero VAD requires exactly 512. The pipeline bridged that gap with a helper
that *trimmed each chunk to 512 samples*, throwing away 588 of every 1100.

That helper was a lazy patch over a design gap, and it silently mutilated the
signal: recognition was running on 47% of what the microphone heard, with no
error anywhere to indicate it.

Replaced with a proper re-framer (`audio/reblock.py`) that accumulates and emits
whole frames, carrying the remainder forward. Its tests assert the property the
old code violated — **no sample lost, none duplicated, order preserved** —
including at the measured 1100-sample chunk size.

| Same 12 s window | Before | After |
|---|---|---|
| Blocks processed | 174 | **373** |
| `samples_in` | — | 191,400 (11.96 s) |
| `samples_out + pending` | — | 191,488 |

The 88-sample excess is the final `flush()` zero-padding a partial frame, which
is correct. Sample accounting is now reported in the run summary, so this class
of bug announces itself instead of hiding.

243 tests passing.

---

## 2026-08-30 — VRAM measured, and two findings about LLM translation

### The budget was estimated, and the estimates were wrong

`scripts/measure_vram_budget.py` loads the stages co-resident in one process
rather than summing separate measurements.

| Stage | Estimated | Measured |
|---|---|---|
| ASR large-v3-turbo | 1030 | **1223** |
| ASR large-v3 | 1894 | **2113–2181** |
| MT NLLB-600M (GPU) | 3100 | **844–929** |
| MT NLLB-600M (CPU) | — | **0** |
| TTS Chatterbox | 2000 | **2824–3368** |
| TTS generation peak | not budgeted | **42–284** |

Two errors in opposite directions partly cancelled, so the total looked roughly
right by luck. Synthesis, not translation, is the largest consumer.

Full-pipeline peaks: turbo 7230 (headroom 958), large-v3 7751 (437), large-v3
with translation on CPU 7034 (1154). **large-v3 fits.** ADR 0002 eliminated it
on arithmetic where two of three terms were wrong. The conclusion survives on
latency grounds — p95 724 ms against turbo's 293 ms — but not on the stated
reason. Recorded in [ADR 0003](docs/adr/0003-vram-budget.md).

### The desktop is the noisiest line in the budget

Baseline measured 753 MiB after a reboot with applications closed, and
1374–1857 MiB with a normal session open. The machine drives both displays from
the NVIDIA GPU: the Intel UHD iGPU (`PCI\VEN_8086&DEV_A788`) was disabled in
MUX mode, and after switching to Optimus it enumerates but runs on Microsoft's
generic display driver, so it still cannot take over the desktop.

Worth fixing, but not a blocker: with applications closed, large-v3 already fits
with ~1124 MiB of headroom.

### FINDING — TranslateGemma through Ollama's generate endpoint is unusable

Four prompt formats, all broken:

| Prompt | Result |
|---|---|
| `tr: … \nit:` | translates, then invents further sentence pairs |
| `tur_Latn: … \nita_Latn:` | translates, then adds German and French versions |
| instruction | does not translate; continues the Turkish text |
| plain | echoes the source, then repeats one translation twice |

The model is instruction-tuned and expects its own chat template with stop
tokens. `/api/generate` bypasses the template entirely, so generation never
terminates. This is a harness problem, not a model verdict — retry through
`/api/chat` before drawing any conclusion about quality.

It is also exactly the failure mode the `sanitise` layer in `mt/base.py` was
written for, and confirms that layer is load-bearing rather than defensive.

### FINDING — quantisation choice dominates the VRAM question

The Q8_0 build of TranslateGemma-4B occupies **5.24 GB**, fully GPU-resident.
Against a 4B model budgeted at ~3000 MiB, that is nearly double — because the
budget assumed Q4_K_M and the download was Q8_0.

5.24 + 1.2 (ASR) + 3.6 (TTS) = **10 GB. Does not fit.** A Q4_K_M build should
land near 2.5–3 GB and is the version worth testing.

The general point: "a 4B model" is not a VRAM figure. The quantisation is.

## 2026-08-31 — the architecture was wrong, and the evidence was in the user's home directory

### Synthesis measured, and it broke the budget

Chatterbox Multilingual v3: **4.3–10.5 s per sentence**, RTF 1.29–1.61. End to
end that gave 5.97–7.58 s against a 2–3 s target. Fitting time against text
length showed **5.42 s fixed + 31 ms/char**, so splitting a sentence into
clauses pays the fixed cost per clause and makes it *worse* — 11.97 s becomes
16.63 s split in two. Chunking was a trap.

Splitting the call: T3 (autoregressive token generation) is 86%, the S3Gen
vocoder 14%. Token rate 13–17 tok/s against the ~25 tokens needed per second of
audio, so T3 alone runs at RTF ~1.7. Two hypotheses tested and both wrong:
`cfg_weight=0` does not avoid the batch-of-two and is 24% *slower*; and the
model runs entirely in float32, leaving the 4070's tensor cores unused.

### FINDING — the elimination that cost weeks

Kokoro was in the very first model survey, and was dismissed in one line: *"very
fast but no voice cloning."*

That line contained an assumption nobody examined: **that one component must
both generate speech and carry the speaker's identity.** Nothing required it.

| | Chatterbox | Kokoro |
|---|---|---|
| Longest sentence | 10.54 s | **0.195 s** |
| RTF | 1.29–1.61 | **0.019–0.073** |
| Silence padding | "Sì." → 3.00 s audio | "Sì." → 0.97 s |

**54× faster** on the longest sentence, measured in the main environment.

What prompted the re-examination was not analysis. The user mentioned an earlier
prototype of theirs; it was found in their home directory using `RealtimeTTS`
with a cloud engine plus Applio/RVC for voice conversion. **The split
architecture was already on this machine** while the plan assumed a monolith.

The general lesson, and the reason it is written down: *measurement inside a
chosen architecture cannot detect that the architecture is wrong.* Every ADR up
to 0005 measured carefully. None questioned the shape.

### Seed-VC closes the gap

Zero-shot voice conversion, converting Kokoro's Italian into the presenter's own
recorded voice:

| `diffusion_steps` | Time for 9.72 s of audio | RTF |
|---|---|---|
| 4 | 1.02 s | 0.105 |
| 10 | 1.29 s | 0.132 |
| 25 | 1.96 s | 0.201 |

Zero-shot is the decisive property: RVC and LLVC both train per voice, which
would kill the "clone a volunteer from the audience" demonstration.

Typical 4.4 s sentence, end to end: `0.70 commit + 0.29 asr + 0.49 mt + 0.18
Kokoro + 0.57 Seed-VC = 2.23 s`, against 7.58 s before. Co-resident VRAM ~5.3 GB
versus Chatterbox's 7.2 GB.

Quality is **not** measured — samples at 4, 10 and 25 steps went to the user for
a listening judgement. [ADR 0006](docs/adr/0006-synthesis-architecture.md) stays
*proposed* until that returns.

### LLVC evaluated and rejected without installing

Its headline is <20 ms at 16 kHz. Three reasons it does not apply:

1. 16 kHz output against our 22–24 kHz chain — audibly duller.
2. **Not zero-shot.** The paper's own method trains it by distilling a
   *pretrained RVC model*, so it needs per-voice training plus an RVC model
   first — more work than RVC, not less.
3. The 20 ms is per-frame algorithmic latency for continuously streaming a live
   microphone. We convert a finished sentence; that advantage does not transfer.

### Environment discipline, learned the hard way

`chatterbox-tts` pins `torch==2.6.0` exactly. Installing it silently replaced
the CUDA build with the CPU wheel and broke every other stage — caught only
because CUDA was checked immediately afterwards. The pin turned out to be
over-strict; it runs fine against torch 2.11.

Since then every risky install is isolated and verified. Seed-VC went into its
own `seedvc` environment (it pins torch 2.4.0 and transformers 4.46.3, and ran
fine on 2.6.0 — the pins are conservative, not strict). Kokoro pins no torch at
all and installed into the main environment without disturbing anything.

Windows obstacles worth recording: torch 2.4.0 cannot load `fbgemm.dll` (use
2.6.0); `webrtcvad` needs MSVC build tools (use `webrtcvad-wheels` and install
`resemblyzer` with `--no-deps`); and `SeedVCWrapper(device="cuda")` fails
because the library calls `.type` on it — pass `torch.device("cuda")`.

Also: Seed-VC downloads weights into `./checkpoints` at the repository root and
one commit picked up 1.4 GB of them before it was noticed. Now git-ignored.

### Deployment shape settled: two processes

Tried to collapse everything into one environment. Kokoro moved into `parliamo`
cleanly and got *faster* (RTF 0.019–0.073, longest sentence 0.195 s, 587 MiB).
Seed-VC imports there but fails at load: `BigVGAN._from_pretrained() missing 2
required keyword-only arguments: 'proxies' and 'resume_download'` — a real
`huggingface_hub` API break rather than a cautious pin. Forcing an older hub
into `parliamo` would endanger `datasets` and `transformers`, which is exactly
the trade that broke the environment once already.

So: main pipeline in `parliamo`, voice conversion in `seedvc` behind a local
socket. ~0.57 s of compute per sentence, ~1 ms of IPC — free in latency terms,
and the conflict stays isolated instead of recurring on every upgrade. Same
shape the earlier prototype used with Applio over HTTP.

---

## Where things stand

**Measured and settled**

| Stage | Choice | Evidence |
|---|---|---|
| Audio I/O | WASAPI, half-duplex gate | ADR 0001, 66 tests |
| ASR | `large-v3-turbo`, beam 5, int8, hotwords | ADR 0002 — 14.00% WER in situ, p95 293 ms |
| Translation | NLLB-600M via CTranslate2 | ADR 0004 — chrF++ 45.38, 487 ms p95 |
| VRAM budget | measured co-resident, not summed | ADR 0003 |
| Synthesis | Kokoro + Seed-VC, split | ADR 0006 — 2.23 s end to end |

**Open, in priority order**

1. **`pipeline.output_latency_ms` is still 0.** The gate is sized from
   PortAudio's reported latency, which excludes ~430 ms of vendor DSP. Must be
   measured in the actual venue before any rehearsal with speakers.
2. **A rehearsal with a live microphone** — the chain now runs end to end on
   recorded speech, but never yet on someone talking.
3. **Friulian branch** — data collection and the NLLB fine-tune, untouched.
4. **Operator interface** — untouched.
5. **Watermark demonstration** — the ethics segment's centrepiece.

*(Items 1 and 2 of the previous list — the listening check on the cloned voice,
and wiring Kokoro and Seed-VC into the pipeline — are both done. NLLB-1.3B has
been measured and rejected; see ADR 0007.)*

**Environment map**

| Env | Python | torch | Holds |
|---|---|---|---|
| `parliamo` | 3.12 | 2.11.0+cu128 | pipeline, ASR, MT, Kokoro |
| `seedvc` | 3.10 | 2.6.0+cu124 | voice conversion only |
| `translator` | 3.11 | 2.4.0 | the user's earlier prototype, untouched |
| Applio | — | 2.7.1+cu128 | RVC, fallback path |

## 2026-08-31 — the pipeline closes

`LiveTranslator` adds a fourth thread taking recognised Turkish to spoken
Italian, and `scripts/live_translate.py` runs the whole thing. Verified: models
load, the microphone opens, subtitle mode reports cleanly, and the consent check
refuses a voice without a record before anything is cloned.

The design rule is **degrade, do not stop**, and each degradation has a test:
translation failure still reports the sentence; synthesis failure still delivers
the text, so subtitles survive; conversion failure speaks in the generic voice,
because the wrong voice beats silence; a raising callback is logged and the next
sentence still goes out.

### BUG — config still named a model that was rejected weeks ago

`live_translate.py` failed on startup with `unknown MT backend 'llama_cpp'`.
ADR 0004 chose NLLB in place of an LLM and the config was never updated. Nothing
had caught it because no script had read `mt.backend` until now — the benchmarks
all constructed their backends directly.

Worth noting as a class: **a decision recorded only in prose and not in the
config is not a decision the system knows about.** `mt` now carries the measured
settings and the reason for them.

### Documentation

Added `docs/00-state.md`: one document holding what the project is, the four
environments and why there are four, the measured latency and VRAM tables, every
decision with its ADR, the eleven bugs that only a measurement would have found,
the Windows-specific obstacles, how to run each part, and the open items in
priority order. README and BUILD_LOG both point at it.

## 2026-08-31 (evening) — a full audit, and the first end-to-end run on real speech

Every file in the repository was read: sources, scripts, tests, configs, all the
docs, and the old measurement reports under `runs/`. The point was to find what
had drifted. It had, in one consistent direction.

### FINDING — three tools were still measuring the architecture we replaced

ADR 0006 moved the live path to Kokoro + Seed-VC. Three tools never got the
message:

* `scripts/bench_tts.py` called `create_backend("chatterbox", ...)` — the
  synthesis benchmark could not measure the shipped synthesiser.
* `scripts/measure_vram_budget.py` imported `ChatterboxMultilingualTTS`
  directly and knew nothing about Kokoro or the Seed-VC process. The last VRAM
  report on disk is `large-v3 + Chatterbox`, an architecture that no longer
  ships.
* `scripts/check_env.py` still listed `llama-cpp-python` under `mt` — rejected
  in ADR 0004 — and `chatterbox-tts` under `tts`.

Same class as the `mt.backend: llama_cpp` bug from this morning, three more
times. The lesson generalises past the config: **a decision recorded only in
prose is not a decision the tools know about.** Anything that names a component
has to move with the decision, and the naming has to be checked.

`tests/test_config_matches_code.py` now asserts that every backend name,
language code, voicepack and file path in `config/default.yaml` resolves against
the registries. Fourteen tests, and they would have caught this morning's crash
before it happened.

### BUG — the published VRAM figure was a sum, which ADR 0003 forbids

`docs/00-state.md` gave the peak as "~6.3 GB", assembled by adding per-model
measurements: 1223 + 844 + 587 + 2875. ADR 0003 rule 2 exists precisely to
forbid that, having been written after summed estimates were wrong by 3.3× and
1.8× in opposite directions.

Measured properly — everything loaded at once, conversion service running:

```
desktop                    ~1200 MiB
+ Seed-VC service          ~2900      ->  4130
+ ASR large-v3-turbo        1213      ->  5346
+ MT NLLB-600M               844      ->  6190
+ Kokoro                     348      ->  6538
+ Kokoro generation          396      ->  6934
+ one voice conversion       358      ->  7292
```

**7292 MiB of 8188, headroom 896.** The summed figure was a gigabyte optimistic.

### BUG — Turkish subtitles crash on a cp1252 console, silently

`print()` of `ışık` raises `UnicodeEncodeError` when Windows hands Python a
legacy code page, which it does whenever output is piped or the console has not
been switched to UTF-8. Confirmed on this machine: `stdout encoding: cp1252`.

The consequence is worse than a crash. Subtitles are printed inside the
`on_delivery` callback, and `LiveTranslator` catches callback exceptions so that
one bad sentence cannot stop the show. Combine the two and the operator sees
**no subtitles at all, and no error either**, while the pipeline works perfectly.

`logging_setup.configure_console()` now switches the console to UTF-8 and
reconfigures the streams with `errors="replace"`, guarded for harnesses that
replace `sys.stdout`. It is called from `setup_logging`, so every script gets it,
plus explicitly in the five scripts that print without logging.

### The first end-to-end run on real speech

`scripts/live_translate.py` grew a `--file` mode that pushes a recording through
the identical pipeline — same VAD, same recogniser, and the same
`LiveTranslator.deliver` the microphone path calls, so the offline path cannot
drift from the live one. Its docstring had promised this flag for a day without
it existing.

Five sentences of the presenter's own in-situ recording, concatenated with
pauses:

```
[  1] Merhaba bugün sizlerle biraz teknolojiden bahsetmek istiyorum
      -> Ciao, oggi vorrei parlarvi di tecnologia.           lag 1.21s
[  3] Salonda oturanların çoğu bu konuya benden daha uzun süredir kafa yoruyor
      -> Probabilmente la maggior parte dei presenti...      lag 1.41s
[  5] Bir şeyin nasıl çalıştığını anlamak onu yapmaktan çoğu zaman daha zordur.
      -> Capire come funziona una cosa è spesso più difficile che farlo.
                                                             lag 1.30s
```

**5 of 5 delivered, no failures, 1.21–1.44 s** without voice conversion — which
lands at ~2.3 s with it, confirming the 2.40 s budget that had until now been
assembled from separately measured stages.

### FINDING — the pipeline dropped a whole sentence, fluently and without a trace

Sentence 2 came out as:

```
tr  Kusurumu hazırlarken aklımdan getiren ilk soru şuydu.
    Acaba gerçekten çalışacak mı?
it  Mentre preparavo il mio errore, la prima domanda che mi è venuta in mente
    è stata:
```

*I wonder whether it will really work?* — gone. Fluent Italian, no error, no
`repaired` flag, unremarkable length ratio.

Isolated with seven probes: the trigger is the **cataphoric** `şuydu` / `şudur`
construction, which points forward at what follows. NLLB renders it `è stata:`
and stops. Written with a colon instead of a full stop the same content
survives, and the second sentence alone translates fine. It is not "only
translates the first sentence" — three neutral sentences all came through.

`şuydu`, `şudur`, `şöyle` are ordinary presentation Turkish. That is the
register this talk is written in.

This is the failure ADR 0004 named for LLMs — *"tells the audience the opposite
of what was said, fluently and with no sign anything went wrong"* — appearing in
the dedicated NMT model that was chosen partly because it does not do that. It
does something quieter: rather than saying the wrong thing, it says nothing.

### Both fixes measured, and the cheap one wins

**NLLB-1.3B** fixes it, and is better on everything: chrF++ 47.11 against 45.38,
BLEU 21.34 against 18.91, and `Dışarıda hava kapalı` becomes `Fuori è buio`
rather than `L'aria è chiusa` ("the air is closed"). It costs **+1118 MiB
co-resident**, leaving 359–700 MiB of headroom against a desktop baseline that
moves 381 MiB on its own. On CPU — ADR 0003's escape hatch — it costs **3023 ms
p95**, more than the entire pipeline budget and worse than the LLM latency ADR
0004 rejected.

**Sentence splitting** fixes it for nothing:

| | whole segment | split |
|---|---|---|
| the cataphoric case | second sentence dropped | recovered |
| median latency, probes | 408 ms | **355 ms** |
| batch ms/sentence, 200 FLEURS | 55.1 | **47.4** |
| chrF++, 200 FLEURS | 45.38 | **45.38** |

Faster, because a batch of short decodes beats one long autoregressive decode.
And chrF++ is unchanged to two decimals because FLEURS sentences are single
sentences — the splitter returns them whole and does nothing. A fix that costs
nothing where the bug is absent.

Splitting Turkish is not `split(".")`: ordinals are written `1. Dünya Savaşı`,
and abbreviations `Dr.`, `vb.` end in a stop mid-sentence. Both are vetoed, with
tests. A colon is deliberately not a boundary — measured, splitting there made
the cataphoric case worse. See ADR 0007.

### BUG — a benchmark that lied about which model it ran

`bench_mt.py` hardcoded the row label `"NLLB-600M"` regardless of `--nllb`, so
the first 1.3B run produced a report attributing 1.3B's chrF++ and VRAM to
600M. `measure_vram_budget.py` did the same in its stage detail. Both now name
the model that was actually loaded.

A measurement that lies is worse than no measurement, because it gets trusted.

### Documentation: five referenced documents did not exist

`README.md`, `config/default.yaml` and a docstring in `mt/ctranslate2_nllb.py`
pointed at `docs/02-architecture.md`, `03-hardware-budget.md`,
`04-adding-a-language.md`, `07-ethics-and-consent.md` and
`08-model-licences.md`. None had ever been written. `audio/devices.py` cited an
ADR about microphone choice that never existed — ADR 0003 turned out to be the
VRAM budget.

All five are now written, and `tests/test_documentation_links.py` fails on a
dangling reference or an ADR that nothing links to. It caught ADR 0005 — the one
that reordered the whole project — being referenced from nowhere.

A pointer to a document that does not exist is worse than no pointer: it reads
as though the answer has been written down somewhere.

### Where this leaves things

366 tests, ruff clean. The remaining item that can still ruin the demonstration
is `pipeline.output_latency_ms`, which is 0 and can only be measured in the
venue.
