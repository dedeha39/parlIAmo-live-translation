# Architecture

```
microphone ─▶ VAD ─▶ Whisper ─▶ NLLB ──────▶ Kokoro / Piper ─▶ RVC / Seed-VC ─▶ speakers
   (tr)      Silero   turbo     600M/1.3B     (GPU)   (CPU)     (socket)       (cloned)
                                   │
                                   └──▶ subtitles on /screen
```

The target is chosen in Setup: Italian, Spanish, German, English, French,
Friulian (spoken with Italian phonetics), or Turkish for questions from the room.

Five stages, four threads, two processes. Each of those numbers was arrived at
by measurement, and this document says why.

---

## 1. Why cascaded, not end-to-end

Direct speech-to-speech models exist. This pipeline is a cascade — recognise,
translate, synthesise, convert — for three reasons:

**Friulian.** No end-to-end model has seen it. NLLB-200 is the only pretrained
translation model that has, and reaching it requires a stage boundary where
text exists.

**Failure isolation.** When a cascade stage fails, the ones around it still
work: a failed synthesis still leaves subtitles, a failed conversion still
leaves speech. `LiveTranslator` is built on that property (§4). An end-to-end
model fails whole.

**Measurability.** Each stage has a latency and a VRAM number attributable to
it. Every decision in `docs/adr/` rests on being able to take those numbers
separately. The cost of a cascade is compounding errors — a recognition mistake
is faithfully translated, as ADR 0007 shows happening — and that cost is real.

---

## 2. The stages

| Stage | Component | Where it runs | Measured |
|---|---|---|---|
| Capture | PortAudio/WASAPI, 32 ms blocks | callback thread | 2 ms in |
| Segmentation | Silero VAD + `SpeechSegmenter` | segmenter thread | <5 ms per block |
| Recognition | faster-whisper `large-v3-turbo` | recogniser thread | 293 ms p95 |
| Translation | NLLB-200-600M via CTranslate2 | delivery thread | 606 ms p95 |
| Synthesis | Kokoro-82M | delivery thread | 150 ms |
| Voice identity | Seed-VC, separate process | delivery thread waits | 920 ms |
| Playback | PortAudio, gated | callback thread | 3 ms reported |

### Capture is 16 kHz mono, always

Every ASR backend we support consumes 16 kHz. The card usually runs at 48 kHz,
so `AudioCapture` resamples with soxr. That resampler emits **ragged** chunks —
1100 samples where the arithmetic suggests 512 — and Silero requires *exactly*
512. `Reblocker` bridges the two by accumulating and emitting whole frames.

The first version trimmed each chunk to 512 instead, and silently discarded
**53% of the microphone input** while appearing to work perfectly. That is why
`Reblocker.accounting()` exists and why `TranscriberStats` reports it: samples
in must equal samples out plus pending.

### Segmentation is where latency is decided

`vad.min_silence_ms` is the single largest term in the end-to-end delay — 700 ms
of the ~2.4 s total. It is a **policy** choice, not a hardware limit: wait less
and speakers get cut off mid-clause, wait more and the audience waits longer.
Lowering it to 500 ms takes the total to 2.20 s.

The segmenter keeps a pre-roll ring buffer sized `min_speech_ms + speech_pad_ms`.
Sized at only `speech_pad_ms` — the obvious choice — it cut **130 ms off the
front of every utterance**, because speech is not confirmed until
`min_speech_ms` of it has already gone past.

### Translation splits sentences first

NLLB-600M drops trailing sentences after a cataphoric construction. Segments are
therefore split on sentence boundaries and translated as one batch. See
[adr/0007-nllb-size-and-sentence-splitting.md](adr/0007-nllb-size-and-sentence-splitting.md).

### Synthesis and identity are two components, not one

This is the correction that reshaped the project. Kokoro generates speech ~42×
faster than a model that also carries speaker identity; Seed-VC applies the
identity afterwards, zero-shot. Forcing one component to do both cost 7.58 s per
sentence against 2.40 s for the pair. See
[adr/0006-synthesis-architecture.md](adr/0006-synthesis-architecture.md).

---

## 3. Four threads, and why not fewer

The stages have incompatible timing, so they cannot share a thread.

| Thread | Job | Cadence | Must never |
|---|---|---|---|
| PortAudio callback | downmix, resample, enqueue | every 32 ms, hard real time | block |
| Segmenter | VAD each block, decide boundaries | keeps exact pace with audio | fall behind |
| Recogniser | transcribe whole utterances | bursty, 200–800 ms | hold the segmenter |
| Delivery | translate, synthesise, convert, play | bursty, ~1.7 s | hold the recogniser |

Merging recogniser into segmenter would drop audio for the whole duration of
every transcription. Merging delivery into recogniser would make the microphone
deaf for the whole duration of every spoken translation — which is what the
half-duplex gate does *deliberately* and for a bounded time, and would otherwise
happen accidentally and for longer.

### Backpressure: newest wins

Both queues drop their **oldest** entry when full, and count the drop. A speaker
who has moved on is not helped by a translation of what they said ten seconds
ago; falling a second behind is recoverable, drifting forever is not.

---

## 4. Degrade, do not stop

Each stage failure has a defined fallback, and each has a test:

| Fails | Result |
|---|---|
| Translation | sentence still reported; subtitle shows the source |
| Synthesis | subtitles survive; nothing is spoken |
| Voice conversion | speaks in Kokoro's generic voice — **wrong voice beats silence** |
| The `on_delivery` callback | logged; the next sentence still goes out |

Nothing in this list stops the pipeline. On stage, a missing sentence is
recoverable and a dead system is not.

---

## 5. The half-duplex gate

Synthesised Italian leaves the speakers. The microphone hears it. ASR
transcribes it, MT translates it, TTS speaks it — and within a few seconds the
system is translating its own output, louder each pass.

Acoustic echo cancellation is the sophisticated answer and needs tuning per
room. The blunt answer works: **while we are speaking, we do not listen.**
`AudioPlayback` owns the gate, so anything that plays audio through it is
protected by construction.

The gate stays shut for `output_latency_ms + half_duplex_tail_ms`. Sizing it
from PortAudio's *reported* latency is a trap: PortAudio reported 3 ms on the
reference laptop where the real acoustic round-trip was **~430 ms**, because it
cannot see the vendor DSP chain. `pipeline.output_latency_ms` must be measured
per venue — see [03-hardware-budget.md](03-hardware-budget.md) and
`scripts/measure_audio_device.py`.

---

## 6. Two processes

The pipeline runs in the `parliamo` environment. The voice runs in its own
process behind a length-prefixed local socket: Seed-VC in `seedvc` (port
8765, any consented reference), RVC in Applio's environment (port 8766, the
presenter's trained voice) or OmniVoice in its own venv (port 8767, any
consented reference, and it speaks the sentence from its text instead of
converting Kokoro's audio - the client sends both). One runs on the GPU at a
time, and Start uses whichever answers - the configured port first, then the
others (`find_voice_service`). Piper, which speaks German and Turkish, runs inside the
pipeline's process on the CPU.

This is not a microservice preference. Seed-VC has a genuine
`huggingface_hub` API incompatibility with the `datasets` and `transformers`
versions the rest of the project needs, and forcing that downgrade is the trade
that once replaced a CUDA torch build with the CPU wheel and broke every stage
silently.

Conversion costs ~570 ms of compute against 8–34 ms of loopback IPC, so the
split is free in latency terms and permanently isolates a dependency conflict
instead of re-litigating it at every upgrade.

The protocol is deliberately tiny and **never unpickles**: 4-byte length, JSON
header, raw float32 samples. A socket that unpickles what it is sent is a
remote-code-execution hole, and this one listens on a laptop that will be on a
conference network. It binds to loopback only.

---

## 7. The Friulian branch

Friulian is a target like any other - Setup, Heard as Friulian - translated
directly, `tr → fur`, by the same NLLB instance. Pivoting through Italian was
the assumption; measured, direct translation was better and took half the time
([ADR 0010](adr/0010-friulian-baseline.md)).

There is no Friulian TTS model anywhere. Friulian text is synthesised through
the **Italian** frontend: plausible prosody, Italian phonetics. That is a
documented limitation disclosed on stage — and a concrete illustration of the
point the talk is making about minority languages, so it is shown rather than
hidden. See [04-adding-a-language.md](04-adding-a-language.md).

---

## 8. What the interfaces are for

Three registries — `parliamo.asr`, `parliamo.mt`, `parliamo.tts` — each map a
name to a backend class, and `config/default.yaml` names one of each.

An abstraction with one implementation behind it is a guess about what varies.
`MTBackend` earned its keep by having two genuinely different things behind it:
a sequence-to-sequence model and an instruction-following LLM. That is what made
`sanitise()` necessary, and every failure it defends against was then observed
in the wild (ADR 0004).

**The names in the config must resolve.** `mt.backend` said `llama_cpp` for
weeks after ADR 0004 chose NLLB, and nothing caught it because no script read
the value until the live CLI was written. `tests/test_config_matches_code.py`
now checks every backend name, language code and file path the config names.
