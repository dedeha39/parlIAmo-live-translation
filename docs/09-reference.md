# parlIAmo — complete reference

Every module, every public function, every route, and the reasoning that put
them there. This is the document to read before designing anything on top of
the system: it says what exists, what each piece guarantees, and where the
sharp edges are.

For *why* the big choices were made, see the ADRs in [adr/](adr/). For the
current status and the measurements, see [00-state.md](00-state.md).

---

## 1. What the system does

Someone speaks Turkish into a microphone. Within about two and a half seconds
the room hears the same thing in Italian, in the speaker's own cloned voice, and
sees it on a projected screen. Nothing leaves the machine — no API, no network
call, no cloud GPU. It runs on one Windows laptop with an 8 GB GPU.

It exists for a one-hour presentation to an Italian audience about AI in speech
and security. That purpose shapes every trade-off in it: **a failure on stage is
worse than a slow success**, and **an audience that is misled is worse than
both**.

Three things it deliberately does *not* do:

- It will not clone a voice without a consent record. `VoiceProfile.validate()`
  refuses, and it has refused since the first week.
- It will not speak a provisional subtitle, by default. Turkish puts negation
  at the end; `değil` reverses the whole sentence. Text can be corrected on
  screen a second later, audio cannot be un-said. The presenter can switch on
  *streaming* from the Live tab, which speaks a sentence once two consecutive
  readings agree it has ended — see §3.5 for what that buys and costs.
- It does not pretend Friulian synthesis exists. Friulian text is spoken with
  the Italian frontend, and that is disclosed rather than hidden.

---

## 2. The shape of the thing

```
microphone
   │
   ▼  32 ms blocks, mono float32 @ 16 kHz
[ AudioCapture ]───── half-duplex gate ◄──────────────┐
   │                                                  │
   ▼  512-sample frames (Reblocker)                    │
[ SileroVAD ] ── speech probability per frame          │
   │                                                   │
   ▼                                                   │
[ SpeechSegmenter ] ── utterance boundaries            │
   │         │                                         │
   │         └──► provisional segment (partial)        │
   ▼                                                   │
[ ASRBackend ] faster-whisper large-v3-turbo           │
   │  Turkish text                                     │
   ▼                                                   │
[ split_sentences ] Turkish-aware                      │
   │                                                   │
   ▼                                                   │
[ MTBackend ] NLLB-200-distilled-600M via CTranslate2  │
   │  Italian text ───────────────► subtitles (SSE)    │
   ▼                                                   │
[ TTSBackend ] Kokoro-82M — generic Italian voice      │
   │                                                   │
   ▼                                                   │
[ VoiceConverter ] Seed-VC or RVC — the identity       │
   │                                                   │
   ▼                                                   │
[ AudioPlayback ]──── closes the gate while speaking ──┘
   │
   ▼
speakers
```

The split at the bottom is the one that made the system usable: **Kokoro
generates the speech, a converter applies the identity.** A single model that
did both took 10.5 s per sentence; splitting it took generation to 0.2 s. See
[ADR 0006](adr/0006-synthesis-architecture.md). Two converters speak the same
socket protocol: Seed-VC (zero-shot, any consented reference, 0.93 s) and RVC
(one voice trained once, 0.43 s); `tts.conversion.port` picks —
[ADR 0012](adr/0012-rvc-beside-seedvc.md).

### Threads

| Thread | Owner | Job |
|---|---|---|
| `segmenter` | `LiveTranscriber` | reads capture blocks, runs VAD, emits segments |
| `recogniser` | `LiveTranscriber` | pops segments, runs ASR, emits `TranscriptEvent` |
| `delivery` | `LiveTranslator` | translate → synthesise → convert → play |
| `ui` | `SubtitleServer` | uvicorn event loop |
| `pipeline-start` | `PipelineController` | builds and warms models on demand |
| PortAudio callbacks | sounddevice | realtime; never blocks, never allocates much |

Two rules hold everywhere:

1. **Never block inside a PortAudio callback.** A slow consumer there produces
   dropouts, not backpressure. The callback downmixes, resamples, enqueues, and
   drops the oldest block when full.
2. **A display fault must never cost a sentence.** Every publish to the browser
   is wrapped; the room can follow a terminal, it cannot follow a pipeline that
   stopped.

### The measured end-to-end budget

Replaying the same 68 s of continuous speech through everything — eleven
sentences, 500 ms commit wait, no failures. Lag from the moment the speaker
stops, commit wait included:

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

Voice conversion is the largest term and the only one with a choice in it.
The tail is driven by **sentence length, not by the pipeline** — synthesis and
conversion both scale with how much audio there is to produce. Subtitles are
much sooner: provisional text appears **1.68 s** into an utterance regardless of
how long it runs on.

### The VRAM contract

Measured co-resident, never summed (an earlier version of this table summed and
was wrong by 3.3×):

| Stage | Added MiB | Running total |
|---|---|---|
| Desktop, apps closed | — | ~1200 |
| Seed-VC service (separate process) | ~2900 | ~4130 |
| ASR large-v3-turbo | 1213 | 5346 |
| MT NLLB-600M | 844 | 6190 |
| Kokoro | 348 | 6538 |
| Kokoro generation activations | 396 | 6934 |
| One voice conversion | 358 | **7292** |

**Peak 7292 of 8188 MiB. Headroom 896.**

---

## 3. Module reference

### 3.1 `parliamo.audio` — getting sound in and out

The hardest-won part of the codebase. See
[ADR 0011](adr/0011-audio-device-opening.md).

#### `audio/devices.py` — which device, and will it open?

| Function | Contract |
|---|---|
| `list_devices(direction=None) -> list[AudioDevice]` | Enumerate, filtered by `"input"`/`"output"`. Names are stripped of control characters — this laptop reports a Bluetooth endpoint with a literal newline inside its name. |
| `resolve_device(spec, direction, devices=None) -> AudioDevice` | Turn `None` / index / `"substring"` / `"substring@hostapi"` into one device. Raises `DeviceResolutionError`. |
| `candidates(spec, direction, limit=6, devices=None) -> list[AudioDevice]` | Devices to try, best first, **each appearing once**. Order: what was asked for → same hardware on another host API, cheapest first → the device Windows uses for this direction → anything else that works. |
| `stutters_under_load(device) -> bool` | Is it on a host API in `STUTTERING_OUTPUT_APIS` (MME, DirectSound)? Their output starves while the pipeline works - 45 underflows in 8 s at rehearsal, WASAPI none. The picker and pre-flight use it. |
| `open_with_fallback(spec, direction, opener, *, retry_pause=0.35) -> T` | Calls `opener(device)` down the candidate list. The **first choice gets two tries** before any fallback is opened. |
| `ensure_com() -> None` | Puts the calling thread in an MTA COM apartment. Windows only, idempotent per thread. |
| `verify_index(device) -> AudioDevice` | Re-resolve by name; PortAudio rebuilds indices whenever anything connects. |
| `preferred_hostapi(available=None) -> str \| None` | Best host API name, or `None` off Windows. |
| `describe_devices(direction=None) -> list[dict]` | Serialisable inventory for the CLI and the page. |

`AudioDevice` is a frozen dataclass: `index`, `name`, `hostapi_name`, channel
counts, `default_samplerate`, low-latency figures, `is_default_*`. Plus
`supports_input` / `supports_output` / `likely_bluetooth` / `label` /
`latency_ms(direction)`.

**Four things here will bite anyone who touches this file:**

- **`ensure_com()` is not optional.** WASAPI is COM and COM is per-thread. The
  pipeline builds on a worker thread; without an apartment, *every* WASAPI
  stream fails at `Pa_StartStream` with `Unanticipated host error -9999`. It
  read as a hardware fault for days because it depended on *where* the code ran.
- **Never call `sd._terminate()`.** It invalidates every open stream, so opening
  the speakers silently kills the microphone.
- **PortAudio's error text is global.** A WASAPI failure is reported with
  whatever the previous WDM-KS attempt left behind. The device named in the line
  is right; the explanation after it may belong to a different device entirely.
- **WDM-KS is ranked last** despite the best latency on paper. On this machine
  all 12 WDM-KS outputs fail and all 11 others open.

Automatic selection never lands on a loopback input (`Stereo Mix`, `PC Speaker`,
`What U Hear`): the pipeline would transcribe its own Italian and speak it
again, and the gate cannot break a loop inside the sound card. An explicitly
named one still works.

#### `audio/capture.py` — the microphone

`AudioCapture(device=None, *, sample_rate=16000, block_ms=32, gate=None, max_queue_blocks=64)`

| Method | Contract |
|---|---|
| `start() -> CaptureInfo` | Open, walking candidates. Idempotent. |
| `stop() -> CaptureStats` | Close and return run statistics. |
| `read(timeout=1.0) -> ndarray \| None` | One block, or `None` on timeout. |
| `blocks(timeout=1.0) -> Iterator` | Yield until `stop()`. |
| `record(seconds) -> ndarray` | Blocking collect. Tooling only. |
| `drain() -> int` | Discard queued blocks; returns how many. |

Always delivers mono `float32` at 16 kHz whatever the device runs at (soxr VHQ
resampling when needed). `CaptureInfo` reports what actually happened —
`resampling`, `latency_ms`, `likely_bluetooth` — as opposed to what was asked
for. `CaptureStats` counts `blocks_delivered/dropped/muted/paused`,
`callback_overflows`, `peak_amplitude`, `recent_peak`. `paused` (set from the
page's P) drops every block after the meter has seen it - so the operator sees
what is being ignored - and before the gate, whose counters stay about the
system's own voice.

When full, the queue drops the **oldest** block: falling a second behind is
recoverable, drifting forever is not.

#### `audio/playback.py` — the speakers, and the gate

`AudioPlayback(device=None, *, sample_rate=24000, gate=None, gain=1.0, channels=2, output_latency_ms=0.0)`

| Method | Contract |
|---|---|
| `start() -> PlaybackInfo` | Open, walking candidates. |
| `submit(samples, sample_rate=None) -> int` | Queue; returns frames queued. Non-blocking. `sample_rate` is the rate the samples were made at - converted to the stream's; omitted, the rate playback was opened with. |
| `play(samples, blocking=True) -> int` | Submit and optionally wait. |
| `wait(timeout=None) -> bool` | Block until drained. `False` on timeout. |
| `flush()` | Discard everything and release the gate — the panic control. |
| `stop()` | Report outstanding underflows, flush, close, release. |
| `report_underflows(force=False) -> int` | Log, as a WARNING, the gaps since the last report; at most once per `UNDERFLOW_REPORT_S` (10 s) unless forced. Called from `submit()`, `wait()` and `stop()`. Returns how many were new. |

`underflows` counts the times the card ran dry **while a sentence was
playing** (starved silence is still silence). The callback only increments it -
logging there could block the audio thread - and the caller's thread reports.

**Playback owns the gate.** It closes before the first sample leaves the card and
releases once the buffer has drained, so feedback protection is on by
construction rather than by remembering. The stream stays open between
utterances because opening a WASAPI stream costs tens of milliseconds.

Output is hard-limited, not allowed to wrap: a clipped word is intelligible, a
wrapped one is a bang through a PA system.

`_extend_gate_tail()` grows the gate to cover real output latency and **warns
loudly when `output_latency_ms` is 0**, because PortAudio reports only its own
buffers and knows nothing about vendor DSP (~170 ms of Realtek/Nahimic on the
reference laptop).

#### `audio/gate.py` — half-duplex

`HalfDuplexGate(enabled=True, tail_ms=250)` — thread-safe, shared by the capture
and playback threads.

`is_open` / `is_closed` / `remaining_ms` / `close()` / `release()` /
`force_open()` / `note_muted_block(frames, sample_rate, had_speech)`.

`close()`/`release()` are reference-counted holds. `force_open()` is the
operator override. `GateStats` records `closes`, `muted_frames`,
`muted_seconds`, and `speech_during_mute_blocks` — so a rehearsal can answer
"how much real speech did we throw away?"

**This is the single most important stage-safety component.** Without it the
speakers feed the microphone and the system translates its own output, forever.

Verified on real hardware rather than assumed — `scripts/verify_feedback_loop.py`
plays speech through the speakers twice, gate off then gate on, and compares
what the pipeline received. On this laptop the loop is 36 dB above the room
floor with the gate off and **completely absent** with it on. It runs both
passes because "the microphone delivered nothing" is also what muted speakers
and headphones look like: the gate is credited for a difference, never for a
silence.

#### `audio/reblock.py` — frames without losing samples

`Reblocker(frame_size)`: `push(chunk) -> list[ndarray]`, `flush()`, `reset()`,
`pending`, `accounting()`. Silero needs exactly 512 samples; capture blocks are
whatever the device gives. `accounting()` asserts samples-in equals
samples-emitted, which is how a 130 ms truncation bug was found.

#### `audio/vad.py` — where sentences begin and end

`SileroVAD(device="cpu", threshold=0.5)`: `load()`, `reset()`,
`probability(block) -> float`, `is_speech(block) -> bool`. Exactly one
`SILERO_BLOCK` (512 samples @ 16 kHz) per call.

`SegmenterConfig`: `sample_rate`, `block_size`, `min_speech_ms`,
`min_silence_ms`, `speech_pad_ms`, `max_segment_ms`, `partial_interval_ms`,
`min_partial_ms`, `soft_cut_after_ms`, `soft_cut_silence_ms`.

`SpeechSegmenter`: `push(block, speech_prob) -> SpeechSegment | None`,
`take_partial() -> SpeechSegment | None`, `flush()`, `reset()`, `state`,
`in_speech`.

`SpeechSegment` carries `audio`, `start_s`, `end_s`, `truncated`,
`mean_speech_prob`, `partial`, `partial_index`, `soft_cut`.

`min_silence_ms` is **500**, and the reasoning is worth repeating because the
assumption was backwards. It was long believed that a shorter wait cuts speakers
off mid-clause. Measured on 68 s of continuous reading:

| ms | segments | WER% | CER% | p95 ASR | truncated | worst wait |
|---|---|---|---|---|---|---|
| 400 | 11 | 11.41 | 3.61 | 0.465 | 0 | 10.5 s |
| **500** | **11** | **11.41** | **3.61** | **0.456** | **0** | **10.6 s** |
| 600 | 8 | 12.08 | 3.71 | 0.500 | 0 | 14.1 s |
| 700 | 7 | 12.08 | 3.90 | 0.545 | 1 | 15.7 s |
| 900 | 6 | 12.08 | 3.71 | 0.500 | 4 | 15.9 s |

At 700 ms the segmenter was failing to split at natural pauses and running on
until `max_segment_ms` cut it mid-phrase. 500 is better on **every** axis.

---

### 3.2 `parliamo.asr` — speech to Turkish text

`create_backend(name, **kwargs)` / `available_backends()` / `register(name, cls)`.

`ASRBackend` contract: `load()`, `unload()`, `warmup(seconds)`,
`transcribe(audio, sample_rate, language=None) -> Transcript`, `describe()`.
`load()` records both wall time and VRAM cost.

`Transcript`: `text`, `segments`, `language`, `language_probability`,
`audio_duration_s`, `compute_s`, `backend`, `dropped_segments`, `rtf`.
`Segment`: `start`, `end`, `text`, `no_speech_prob`, `avg_logprob`.

**`FasterWhisperBackend`** — `large-v3-turbo`, `int8_float16`, beam 5,
`condition_on_previous_text=False`, our own VAD upstream. 14.00% WER in situ,
p95 293 ms against large-v3's 724 ms.

**`asr/repetition.py`** guards Whisper's worst stage failure — the decoder loop.
`compression_ratio(text)` (gzip ratio; higher means more repetitive),
`max_phrase_repeats(text, max_phrase_words)`, `analyse(...) -> RepetitionReport`.
Looping segments are dropped and counted rather than spoken.

**`asr/plausibility.py`** guards the other one — a fluent sentence invented
from a breath ("İzlediğiniz için teşekkür ederim", Whisper's subtitle-track
ending, from the first 0.93 s of an utterance). `syllables(text, language)`:
every vowel in Turkish, vowel runs elsewhere. `too_fast(text, duration_s,
language) -> (invented, rate)`: above `MAX_SYLLABLES_PER_S = 12` with at least
six syllables. Real readings on the reference recording topped out at 9.7;
inventions started at 14. The backend drops the reading and records it in
`dropped_segments` with `syllables_per_s`; the transcriber counts it as
`implausible_readings_caught`, separately from repetition loops.

Two more inventions are dropped by text alone. `subtitle_credit(text)`: the
credit lines of subtitle tracks Whisper was trained on ("Altyazı M.K."),
which are never speech. `hotword_echo(text, hotwords)`: Whisper reciting its
own hotword prompt, which it does when the microphone is too quiet to read -
four or more hotwords in the prompt's order (`HOTWORD_ECHO_RUN`) and at least
80% of the transcript hotwords (`HOTWORD_ECHO_SHARE`). Order is the tell: a
sentence that names the speaker and the company keeps other words around them,
"İpek Nur Yıldız." alone is three, and "yapay zekâ ile ses klonlama" is out of
the list's order. Over all 7,152 transcripts in `runs/` it drops 76, every one
a recital. Words are compared case- and accent-folded: `"İ".casefold()` is
`i` plus a combining dot, and a plain split read "İpek" as two words.

---

### 3.3 `parliamo.mt` — Turkish text to Italian text

`MTBackend`: `load()`, `unload()`, `warmup(text)`,
`translate(text, source_lang=None, target_lang=None) -> Translation`,
`translate_many(texts, ...)`, `describe()`.

`Translation`: `text`, `source`, `source_lang`, `target_lang`, `compute_s`,
`backend`, `repaired`.

**`CTranslate2NLLBBackend`** — NLLB-200-distilled-600M, `int8_float16`, beam 4
(chrF++ 45.38 against 44.46 at beam 1). `_translate_pieces` sends **one batch
per segment**, not one call per sentence.

**`OllamaBackend`** exists and is not used: an LLM took 2671 ms per sentence,
produced the wrong language on 2 of 6 sentences, and **dropped a negation** —
which on stage tells the audience the opposite of what was said. See
[ADR 0004](adr/0004-mt-backend.md).

**`mt/sentences.py`** — `split_sentences(text, max_sentences=...)`. Turkish-aware:
knows ordinals (`3. sınıf`), abbreviations (`Dr.`, `vb.`), and deliberately does
**not** split on a colon (measured: splitting there made it worse).

This exists because NLLB-600M silently dropped a whole clause. Given
*"Aklımdan geçen ilk soru şuydu. Acaba gerçekten çalışacak mı?"* it produced
fluent Italian containing only the first sentence — no error, no flag. Splitting
recovers it **and is faster** (355 ms against 408 ms median), because a batch of
short decodes beats one long autoregressive decode.

**`mt/base.py::sanitise(raw, source)`** strips the artefacts models add —
subtitle dashes, quoting, repeated output — and reports what it repaired.

---

### 3.4 `parliamo.tts` — Italian text to Italian speech

`TTSBackend`: `load()`, `unload()`, `warmup(text)`,
`register_voice(profile)`, `voices()`,
`speak(text, language=None, voice=None) -> Speech`, `describe()`.

`Speech`: `audio`, `sample_rate`, `text`, `language`, `compute_s`,
`first_audio_s`, `backend`, `voice`, `duration_s`, `rtf`.

**`VoiceProfile(name, reference_path, consent, notes="", embedding=None)`** —
`validate()` **refuses a profile with an empty consent record**. This is the
project's ethical position expressed as code, not documentation.

**`KokoroBackend`** — Kokoro-82M, ~0.2 s per sentence, generic Italian voice
(`if_sara`). Identity comes later.

**`ChatterboxBackend`** — kept for the cloning demonstration, where the 10 s wait
is part of the point. Too slow for the live path.

**`tts/conversion.py`** — `VoiceConverter(host, port, diffusion_steps, timeout)`:
`available()`, `convert(audio, sample_rate, reference_path, diffusion_steps=None) -> Conversion`,
`ping()`. Seed-VC runs in a **separate process and a separate conda env**,
because it needs an older `huggingface_hub` than `datasets` and `transformers`
do. Framed messages over a local socket.

`diffusion_steps: 4`. The full story: it went 4 → 15 after a listening check
rejected 4 for digital breaking, then back to 4 the same day once the reference
was rebuilt from a phone recording. The step count was never the problem — it
was compensating for a damaged reference. Measured on a 4.42 s sentence:

| steps | conversion | total |
|---|---|---|
| **4** | **0.67 s** | **2.06 s** |
| 8 | 0.78 s | 2.17 s |
| 15 | 1.10 s | 2.49 s |
| 15, laptop-mic reference | 1.37 s | 2.76 s |

**`tts/reference.py`** — building a reference that is worth cloning from.
`measure_clip`, `trim_gated_regions`, `normalise`, `rank_candidates`,
`build_reference(clips, target_seconds=12, ...)`. `ClipStats` reports
`gated_fraction`, which is the number that matters: the reference laptop's
microphone noise-suppression **gates 16–50% of every file to digital silence**,
and the first reference chosen by hand was 49.5% holes. A phone recording with
0% gating took WER from 14.00% to 11.41%.

---

#### Piper, and which synthesiser speaks which language

**`tts/piper_backend.py`** — `PiperBackend`, VITS voices as ONNX files under
`models/piper` (`<name>.onnx` + `.onnx.json`, from `rhasspy/piper-voices`).
German `de_DE-thorsten-high`, Turkish `tr_TR-dfki-medium`. CPU, 22.05 kHz,
`threads = 8` (the default of every core was slowest on this hybrid CPU); a
CUDA provider is used first if onnxruntime has one. `available_languages()`
lists the languages whose voice is on disk.

**`tts.backend_for(language, preferred="kokoro") -> str | None`** — Kokoro if it
speaks the language, else Piper if its voice is on disk, else None (subtitles
only). Used by `build()`, the Start factory and `/languages`.

**`ui/studio.Studio`** — the operator page's Studio tab. `speak(text, language,
voice_path, options, seed)` refuses any voice without a consent record (or
`AUTO_VOICE`, the model's own choice) and writes a clip with the settings and
seed it was made with (`runs/studio/`, or inside a volunteer's folder so it is
deleted with them); `design(name, gender, age, pitch, accent, whisper, seed)`
invents a voice and saves it under `data/voices/designed/`; `upload(name,
consent, wav_base64, filename)` keeps a recording as a volunteer; `transcript`,
`set_transcript`, `prepare` for what a reference says; `derive(name, source,
semitones, tempo)` moves a consented recording's pitch (±8) and tempo
(0.8-1.25) into a new volunteer-folder voice with the source's consent;
`delete_voice(path)` removes a volunteer's, uploaded, derived or invented voice
with its clips (never the presenter's); `speak(..., redesign=True)` speaks an
invented voice from its description instead of its sample; `languages()`;
`clips()`, `clip_file()`, `delete()`, `load()`. Endpoints: `GET /studio`,
`POST /studio/speak|design|derive|upload|prepare|transcript|voice/delete`, `GET /studio/transcript`,
`GET|DELETE /studio/clip/{id}`, `POST /studio/play/{id}` - through the
pipeline's playback when it runs, so the gate holds the microphone.

**`tts/omnivoice_options`** — `ADVANCED` (sliders: label, range, step,
default, help), `FLAGS`, `NONVERBAL`, `DESIGN`; `clean_options(raw)` clamps
and drops unknown keys; `instruct_for(...)` builds a voice-design instruction
from known words only. The page and the service both read it.

**`tts/conversion.find_voice_service(host, preferred)`** — the port of the voice
service that is running: `preferred`, then `KNOWN_PORTS` (8766 RVC, 8765
Seed-VC, 8767 OmniVoice). `VoiceConverter.convert()` also sends `text` and
`language` (OmniVoice speaks from them; the others ignore them) and takes a
per-call `timeout` - the warm-up uses `LiveTranslator.WARM_TIMEOUT_S` (60 s),
because a new OmniVoice reference is prepared on the CPU first. Start and the pre-flight row both use it, so swapping services needs
only Stop → Start.

**`audio/pitch.py`** — `median_f0(audio, rate)` (pYIN, 60-400 Hz) and
`semitones(from_hz, to_hz)`. At warm-up the translator measures the
synthesiser's voice on `WARM_PHRASES[target]` - two plain sentences, because a
short greeting read ~2.7 semitones high - and, when
`tts.conversion.presenter_f0_hz` is set, sends `pitch` with every conversion;
`rvc_server.py` applies a request's `pitch` over its `--pitch`, Seed-VC ignores
it.

Config: `tts.voices` maps a language to its voice (`it: if_sara`, `es:
em_alex`, `de: de_DE-thorsten-high`, `tr: tr_TR-dfki-medium`); `tts.voice` is the
fallback. `live_translate.py --source LANG` sets recogniser and translator
together. The Turkish hotword list applies to Turkish only
(`load_hotwords(path, language)`).

### 3.5 `parliamo.pipeline` — the threads that connect it

**`LiveTranscriber(backend, *, device, vad, segmenter_config, gate, on_transcript, max_queue_depth)`**

`start()`, `stop(timeout) -> TranscriberStats`, `running`.

Runs `segmenter` and `recogniser` threads. `_enqueue_partial` queues a
provisional segment **only if nothing real is waiting** — a preview that delays
the thing it previews is worse than no preview.

`TranscriptEvent`: `index`, `text`, `segment`, `transcript`, `commit_wait_s`,
`queue_wait_s`, `asr_s`, `partial`, `total_lag_s`.

**`LiveTranslator(transcriber, translator, synthesiser, *, playback, converter, reference_voice, target_lang, on_delivery, max_queue_depth, speak)`**

`warm()` (load and warm every stage **without opening the microphone**),
`start()`, `stop(timeout)`, `deliver(event)` (synchronous, for file replay),
`set_reference_voice(path, consent)`, `set_muted(muted)`, `set_paused(paused)`
(stop listening; output untouched). `LiveTranscriber.set_paused()` survives a
capture restart, and on pause the segment loop commits what was said before
the press as soon as those blocks are read - otherwise it waited for a silence
that only comes after resume and was joined to the next sentence.

`DeliveryEvent`: `index`, `source_text`, `translated_text`, langs, plus the
per-stage cost — `recognition_lag_s`, `translation_s`, `synthesis_s`,
`conversion_s`, `audio_s` — and `spoken`, `voice`, `error`, `partial`, `audio`,
`sample_rate`, `total_lag_s`.

`set_reference_voice` works **while running**: that is the volunteer
demonstration. It validates consent before anything is cloned.

**Streaming** (`stream_sentences`, off by default, live switch on the page).
Speaks a sentence the moment the provisional text shows it has ended, rather
than at the pause. Two guards, both from measurement: an ellipsis does not end
a sentence (it is what the recogniser writes when audio ran out mid-word), and
a sentence is spoken only when two consecutive partials agree on it — *local
agreement*, the standard simultaneous-translation policy. Without the second
guard the first measurement spoke 24 sentences where the speaker said 11, the
extras being confident misreadings of the first few words.

Measured on the 68 s reference recording, audio position at which each
sentence became available:

| sentence | wait for pause | streaming | earlier by |
|---|---|---|---|
| "Pensate al Friuliano che parla la regione." | 46.5 s | 43.7 s | 2.8 s |
| "E non è un problema tecnico." | 53.0 s | 50.8 s | 2.2 s |
| "Oggi voglio mostrarvi tre cose." | 67.5 s | 63.1 s | 4.4 s |

Streamed sentences also skip the commit wait: per-sentence lag ~0.35 s against
~0.9 s. The cost on that recording was one duplication in eleven — a sentence
spoken early as two short ones, then again when the commit merged them. And the
named risk stays: a full stop one word before `değil` speaks the affirmative;
the committed negation is then spoken as a correction rather than dropped.

**Chunk mode** (`stream_mode: chunk`, the third position of the same switch)
is the buffer the presenter asked for by name: words that two consecutive
partials agree on are spoken in pieces of four or more, before the sentence
ends. On the same recording the first audio of the 10 s utterance arrives
**5.1 s earlier** (41.3 s against 46.5 s) — and the room hears this:

```
41.3s  Friuliano che parla alla Regione del Friuli
46.5s  ci sono circa 600.000 persone che parlano la lingua, ...
```

for what the sentence meant, *"Pensate al Friuliano, la lingua parlata in
Friuli, che è parlata da circa 600.000 persone"*. The imperative — *düşünün*,
"think of" — is the last word in Turkish, so the first chunk is a clause with
no verb and the translator makes it a relative clause about a Friulian who
speaks. Thirty deliveries where the speaker said eleven sentences. This is what
the SOV argument sounds like; the mode exists so it can be heard rather than
argued about.

**Both streaming ledgers belong to one utterance.** What has been spoken early
(`_spoken_early`, `_seen_once`, `_prev_words`, `_spoken_words`) is keyed by the
utterance's `segment.start_s`: partials and the commit of one utterance share
it, and a new start clears the ledger (`_track_utterance`). Before this, only
a commit cleared it — so an utterance whose commit never came left its words
behind, and chunk mode compared every later utterance against them and spoke
nothing more for the rest of the talk.

**Within one utterance, chunk mode aligns rather than counts.** Each reading
is a fresh decode of the whole utterance; the commit especially may segment it
differently from the partials ("üç te", then "üçte"). `_spoken_end(spoken,
words)` aligns the words already spoken to the new reading with
`difflib.SequenceMatcher` (matches more than `ALIGN_SLACK = 3` words apart are
ignored as the same short word said again later) and covers what does not
align by its length in characters. Agreement between readings ignores
punctuation, as sentence mode always did: the recogniser flickers between
"istiyorum" and "istiyorum.", and each flip used to hold a chunk back one
partial - 0.15-0.20 s of average word lag on the reference recording.
What no policy can fix: a word the room has already heard that the recogniser
later reads differently ("3" then "üç"). On the reference recording that is
all of the 4 missing and 4 extra words in chunk mode.

---

### 3.6 `parliamo.ethics` — the part the audience can act on

**`ethics/watermark.py`** — `detect(audio, sample_rate) -> WatermarkReport`,
`is_watermarked`, `survives(audio, rate, transform, label)`.
`DETECTION_THRESHOLD = 0.75`.

`WatermarkReport.verdict()` is phrased carefully on purpose: no watermark
**"means only that whatever made this did not mark it"** — not that the audio is
real. Overclaiming here would teach the audience the wrong lesson.

**`ethics/disclosure.py`** — `add_disclosure(audio, sample_rate, announcement, ...)`
interleaves a spoken *"questa voce è sintetica"* at the start, the end, **and at
random interior points**. `interior_positions()` uses `secrets.SystemRandom`,
not `random` — predictability is the entire point: a fixed pattern can be
trimmed out, an unpredictable one cannot. `_nearest_quiet()` moves each cut to
the quietest nearby sample so it lands between words.

This is what makes the fraud demonstration safe to perform: the recording cannot
be lifted and reused.

---

### 3.7 `parliamo.eval` — measurement

- `eval/wer.py` — `evaluate(references, hypotheses, normalizer)` returns WER and
  CER, both raw and normalised. `worst_examples(...)` for reading rather than
  counting.
- `eval/normalize.py` — `TurkishNormalizer`, `turkish_lower/upper` (dotted and
  dotless i), `number_to_turkish`, `spell_numbers` (honours `40.000` grouping).
- `eval/mt_metrics.py` — `score(hyps, refs, target_lang) -> MTScore` (chrF++ and
  BLEU, plus `length_ratio` and `empty_outputs`), `worst_translations(...)`.
- `eval/datasets.py` — `load_fleurs`, `load_fleurs_parallel`, `load_tsv_pairs`,
  `load_wav_manifest`.
- `eval/runner.py` — `run_variant`, `expand_grid`, `rank`,
  `significance_note(a, b)`, which says whether a WER difference is worth acting
  on given the sample size. That function exists because it is otherwise very
  easy to celebrate noise.

---

### 3.8 `parliamo.ui` — the operator surface

**`PipelineController(factory, on_status)`** — owns build/start/stop.

`start()` returns immediately and builds on the `pipeline-start` worker thread;
`stop()`; `state()`; `set_muted()`; `set_paused()` (kept across Stop and Start,
so the page's banner is never wrong); `set_reference_voice()`; `devices()`.
Attributes `pending_voice`, `input_device`, `output_device`.

Statuses: `stopped` → `starting` → `running` → `stopping` → `stopped`, plus
`failed` carrying the exception text. **A page that says nothing for forty
seconds is indistinguishable from a page that has crashed**, so every stage is
published.

The inversion this fixed: the first version built the pipeline and *then* served
the page, so the microphone opened the moment the command ran. There was no
start button because there was nothing to press it on.

`devices()` returns `selected`, `live`, `substituted`, `applies`. `substituted`
matters: a device that will not open is replaced by one that will, which is
right thirty seconds before a talk and wrong to do quietly.

**`SubtitleServer(host, port, state, on_mute, on_voice, controller)`**

`bind(attempts=8)` claims the port **before** anything is printed, stepping
forward if it is busy. `start() -> str` returns a URL that actually works.

> Deliberately no `SO_REUSEADDR`. On Windows it lets a second process bind a port
> another process is actively listening on, and requests then land on whichever
> socket the kernel picks. That cost hours once already, on the conversion
> server.

Also: this file must **not** use `from __future__ import annotations`. FastAPI
resolves handler annotations at runtime; with the future import every POST comes
back 422 with `{"loc": ["query", "request"], "msg": "Field required"}`.

**Nothing slow runs on the event loop.** The routes are `async`, so a
synchronous call inside one — opening a device, pre-flight, loading a voice —
stops every other request, including `/state` and the subtitle stream, until it
returns. Every such call is `await asyncio.to_thread(...)`. A test starts a
real server, holds one action for 1.5 s and checks `/state` still answers in
under 0.5 s; on the old code it waited 1.32 s.

**`running_instance(host, port, span=4) -> str | None`** — asks ports
`port … port+3` for `/state` and returns the address of the one that answers
with parlIAmo's fields. `live_translate.py --ui` calls it first and, if another
copy is running, prints that address and exits with code 3 rather than
starting a second pipeline beside it: two copies fight over the microphone and
the GPU, and the page the operator is looking at may belong to either.

#### HTTP API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | operator application (`app.html`) — five tabs |
| GET | `/screen` | audience screen (`page.html`) — dark, huge, projected |
| GET | `/events` | SSE stream: `delivery`, `state`, `pipeline` |
| GET | `/state` | current `UIState` |
| GET | `/pipeline` | `{status, detail}` |
| POST | `/pipeline/{action}` | `start` \| `stop` |
| GET | `/devices` | `{input[], output[], selected, live, substituted, applies}` |
| POST | `/devices` | choose input/output; returns the same shape |
| GET | `/voices` | reference recordings on disk, volunteer flag |
| POST | `/voice` | switch voice: `{path, consent}` |
| POST | `/voices/record` | record a volunteer and build a reference |
| POST | `/mute` | `{muted: bool}` — silences the *next* sentence |
| POST | `/pause` | `{paused: bool}`, or no body to toggle — stop/resume listening (key P) |
| POST | `/flush` | drops the sentence already in the speakers |
| GET/POST | `/streaming` | `off` / `sentence` / `chunk` — live switch, cycles with no body |
| GET/POST | `/languages` | source and target, from the backends' own tables; next start |
| GET/POST | `/models` | recogniser and translator, only what is on disk, with measured cost; next start |
| GET/POST | `/tempo` | the commit wait, 300/400/500/700 ms, with ADR 0009's numbers; next start |
| DELETE | `/voices` | delete every volunteer recording |
| GET | `/preflight` | the pre-flight checklist, as data — including which voice service answered, a warning when NLLB 1.3B is chosen beside Seed-VC, the **chosen** microphone and speakers (not Windows' default), and a warning when the speakers are on MME or DirectSound |
| GET | `/metrics` | gate statistics and the lag distribution |
| GET | `/friulian` | the review sheet as rows, plus the native speaker's clips |
| GET | `/friulian/audio/{kind}/{name}` | one clip, `pipeline` or `native` |
| GET | `/measurements` | VRAM and output latency, from this machine's reports |

SSE messages are `{"kind": ..., ...}`. On connect the server replays the last
`HISTORY = 4` committed sentences, so a browser that joins late is not blank.

`UIState`: `source_lang`, `target_lang`, `speaking`, `muted`, `voice`,
`gate_warning`, `delivered`, `partials`, `last_lag_s`, `listening`.

---

### 3.9 Support modules

- **`config.py`** — layered `default.yaml` → `local.yaml` → CLI overrides into a
  typed `Config`. **Unknown keys are rejected**, which is how a stale
  `mt.backend: llama_cpp` was caught.
- **`paths.py`** — `find_repo_root`, `resolve`, `ensure_dir`,
  `configure_model_cache` (points every downloader at one directory inside the
  repo).
- **`logging_setup.py`** — `setup_logging`, `log_event`, `LatencyTracer`
  (per-stage timings for every segment), `JsonlHandler`, and
  **`configure_console()`**, which sets code page 65001 and
  `reconfigure(errors="replace")`. Without it, printing Turkish crashes on
  cp1252 — and because subtitles print inside `on_delivery`, whose exceptions
  are swallowed, the operator sees *nothing at all*.

---

## 4. Scripts

### Run it

| Script | What it does |
|---|---|
| `live_translate.py --ui` | The application. Serves the page; you press Start. |
| `live_translate.py --file X.wav` | Replay a recording through everything, no microphone. |
| `live_transcribe.py` | Turkish transcription only, with the latency breakdown. |
| `voice_conversion_server.py` | Seed-VC service, zero-shot, port 8765. **Run from the seedvc env.** |
| `omnivoice_server.py` | OmniVoice service, zero-shot, port 8767. **Run from `venvs\omnivoice`.** Speaks the request's `text` in its `language` with the reference voice; refuses a request without text. `--steps` (16). A reference is cut at word and sentence boundaries to ≤12 s, transcribed once on the CPU into `<name>.transcript.json` (correct it by hand if a word is wrong) and encoded on the CPU. Digits are spelled out first (`tts/numbers.spell_numbers`). A request with `instruct` and no reference speaks in a voice designed from attributes; with `auto` and neither, in the model's own choice; without any of the three it is refused. `options` (checked by `clean_options`) and `seed` pass through; the reply carries the seed used. Ops besides `convert`/`ping`: `languages` (all 646) and `prepare` (transcribe and encode a reference now). A voice is rebuilt when its recording or its transcript changes. A browser opening the port gets a plain-text pointer to the operator page. |
| `rvc_server.py` | RVC service, one trained voice, port 8766. **Run from Applio's env.** `--pitch` per source voice (RVC keeps the pitch it is given: −8 from if_sara, +9 from im_nicola). Serves Applio's `_best_epoch` checkpoint if training marked one, else the highest epoch by number; `--weights` names one explicitly. |

### Set up and check

| Script | What it does |
|---|---|
| `check_env.py` | Phase 0 gate: GPU, torch, disk, binaries, packages, audio. |
| `list_audio_devices.py` | Device inventory, ranked and annotated. |
| `measure_audio_device.py` | Bandwidth, noise floor, **acoustic round-trip latency**. |
| `measure_room_echo.py` | Does the echo of a real Italian sentence outlast the gate? Pipeline's own playback, gate and Silero; per sentence the margin between the echo's last speech-like frame and the gate reopening, and its level. `--write-config` raises `half_duplex_tail_ms` to leave 150 ms over the worst. Start re-reads both gate values. |
| `verify_feedback_loop.py` | Proves the gate stops the speakers reaching the mic. |
| `measure_echo_cancellation.py` | Would WebRTC echo cancellation (`livekit` APM) let the mic stay open while the translation plays? Echo over floor, suppression, what Silero+Whisper would hear before/after, and WER when the presenter talks over it. Mic in WASAPI exclusive mode (bypasses the Intel driver's own processing). `--from DIR` re-analyses a saved run. |
| `measure_vram_budget.py` | Co-resident VRAM, never summed. |

### Voices and ethics

| Script | What it does |
|---|---|
| `build_voice_reference.py` | Assemble a good reference from clips you have. |
| `record_script.py` | Record a reading script sentence by sentence. |
| `clone_volunteer.py` | Clone on stage in 15 s; `--forget-all` deletes everything. |
| `ethics_demo.py` | The clone, the disclosure, the watermark table. |

### Measure

`bench_asr.py`, `bench_mt.py`, `bench_tts.py`, `sweep_asr.py`,
`profile_tts_cost.py`, `profile_tts_stages.py`, `analyse_recording.py`,
`prepare_friulian_demo.py`.

`bench_mt.py --ollama MODEL [--skip-nllb]` scores a model served by a local
Ollama on the same FLEURS pairs as NLLB, one sentence per request.
`mt/ollama_backend.py` picks the prompt by model name (`PROMPT_TEMPLATES`) and,
for models whose chat template Ollama derives wrongly, sends the request raw
with the turn markers written out (`RAW_WRAPPERS` - HY-MT1.5, whose derived
template dropped the prompt entirely). The server is addressed as
`127.0.0.1`, never `localhost`: on Windows the latter tries IPv6 first and cost
~2 s per request.

---

## 5. Configuration

`config/default.yaml` is committed and heavily commented — most values carry the
measurement that chose them. `config/local.yaml` is git-ignored and overrides it.

The values most worth knowing:

| Key | Default | Why |
|---|---|---|
| `vad.min_silence_ms` | 500 | measured better on every axis than 700 |
| `asr.model` | `large-v3-turbo` | p95 293 ms vs 724 ms for +0.56 WER |
| `asr.beam_size` | 5 | 14.50 → 14.00 WER for 21 ms |
| `mt.split_sentences` | true | recovers a dropped clause **and** is faster |
| `mt.beam_size` | 4 | chrF++ 45.38 vs 44.46 |
| `tts.conversion.diffusion_steps` | 4 | 0.67 s vs 1.10 s at 15 (Seed-VC only) |
| `tts.conversion.port` | 8765 | 8765 Seed-VC (0.93 s, any reference) · 8766 RVC (0.43 s, the trained voice) |
| `tts.voice` | `if_sara` | was silently ignored until 2026-09-15; `im_nicola` is the male Italian pack |
| `pipeline.half_duplex` | true | the most important safety flag here |
| `pipeline.output_latency_ms` | **0** | **measure it**, see below |
| `pipeline.emit_partial_transcripts` | true | verified free |

### Measuring `output_latency_ms`, which is not optional

The gate keeps the microphone shut for `output_latency_ms + half_duplex_tail_ms`.
PortAudio reports only its own buffers — 3 ms on this laptop — and cannot see the
vendor DSP chain, which measures ~430 ms through the same speakers. At 0 the
microphone reopens while the last sentence is still coming out of the speakers,
and the system starts translating itself.

```bash
python scripts/measure_audio_device.py --skip-bandwidth --skip-noise --write-config --label venue
```

It plays chirps, hears them back, cross-correlates, writes the **median** into
`config/local.yaml`, and then checks that value against the **worst** chirp it
saw. It edits the file as text rather than re-serialising it, because that file
holds the consent record for the cloned voice and a YAML round-trip would delete
every comment in it.

Measured three times on the reference laptop's speakers:

| run | median | range | gate | spare |
|---|---|---|---|---|
| 1 | 432 ms | 229–552 | 682 ms | 130 ms |
| 2 | 366 ms | 266–548 | 616 ms | 68 ms |
| 3 | 432 ms | 229–597 | 682 ms | 85 ms |

**This output path is jittery** — up to 581 ms of spread within a single run,
which the tool flags. The margin holds, but not by much, and it is a property of
the room and the speakers rather than of the software. Re-measure in the venue,
and again after changing anything about the output.

The chirp says when sound arrives, not how long a hall keeps it.
`measure_room_echo.py` measures the second: it plays a real sentence through
`AudioPlayback` with a real `HalfDuplexGate`, captures through `AudioCapture`
with a stand-in gate that keeps every block and records what the real gate
was - the state the pipeline's capture would have acted on - and runs Silero
at `vad.threshold` over all of it. Run end to end on the reference laptop
(2026-10-02) with its speakers turned down - the presenter could not play
sound aloud there - the sentence reached the Intel array at -72 dBFS and was
never speech, which says nothing about a hall. The leak branch has been
exercised only on synthetic timelines (`tests/test_room_echo.py`); the hall,
at the talk's volume, is the real test.

---

## 6. Testing

**767 tests**, `pytest`, ruff clean. Run them:

```bash
python -m pytest -q
```

Hardware tests are marked `audio` and skip when PortAudio finds nothing, so the
suite passes on a machine with no sound card.

Several tests exist because something specific went wrong, and they are the ones
to leave alone:

- `test_config_matches_code.py` — every config name must resolve to real code.
  Caught a stale backend name that had been wrong since ADR 0004.
- `test_capture_discards_audio_while_gate_is_closed` — caught, within a minute,
  a "fix" that called `sd._terminate()` and silently killed the microphone
  whenever the speakers opened.
- `test_documentation_links.py` — every ADR must be linked from somewhere.
- The UI tests run `node --check` on both pages and assert every
  `getElementById` id exists. A single JavaScript syntax error had previously
  killed the whole page script: it looked correct and every control was dead.

---

## 7. Known gaps

Honest list, not a roadmap.

| Gap | State |
|---|---|
| `pipeline.output_latency_ms` | measured on this laptop (432 ms) — **re-measure in the venue** |
| The output path is jittery here | 229–597 ms across chirps; gate margin ~85 ms |
| No rehearsal with a live microphone | the gate and the loop are now verified without one; a voice into the mic is not |
| The audio tail on long sentences | **inherent, not a bug** — see below |
| Friulian needs a native reviewer | `data/friulian/review-sheet.md` awaits ARLeF |
| Friulian has no TTS anywhere | spoken with the Italian frontend, disclosed |
| Friulian clip playback | built — the pipeline's 11 renderings and 39 sentences from a native speaker (Wikitongues, CC BY-SA 4.0), played as recorded |

**On the tail, because it looks like an unfinished feature and is not.** The
longest segment in the reference recording is 10.1 s, so the audience waits
10.6 s for that sentence. Cutting long segments at clause boundaries was written
as a requirement, implemented within the hour, measured, and **rejected**: it
bought 0.9 s off the worst case for 0.67 WER points and a sub-1.5 s fragment,
and the longest segment stayed 9.2 s at *every* setting including the most
aggressive — because that stretch of speech contains no pause at all. The
speaker did not breathe.

A translation cannot be produced for a sentence that has not finished. Human
simultaneous interpreters solve this by starting before the end, which this
project rejects for Turkish because the verb and its negation arrive last. The
**text** tail is already solved: partials put words on screen 1.68 s in, seven
and a half seconds before the audio on that segment.

`soft_cut_after_ms` is kept and defaults to 0. It goes on with a measurement
from a rehearsal, not on the reasoning that produced it the first time — which
was wrong. See the addendum to [ADR 0009](adr/0009-commit-wait.md).

---

## 8. Where to look next

- [00-state.md](00-state.md) — current status and every measurement
- [02-architecture.md](02-architecture.md) — the pipeline in detail
- [03-hardware-budget.md](03-hardware-budget.md) — the VRAM contract
- [07-ethics-and-consent.md](07-ethics-and-consent.md) — the consent position
- [08-model-licences.md](08-model-licences.md) — NLLB is CC-BY-NC
- [adr/](adr/) — eleven decisions and the evidence for each
