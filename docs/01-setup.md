# Setup

Target platform is **Windows 11 with native Python** — not WSL2. The reasoning
is in [adr/0001-windows-native.md](adr/0001-windows-native.md); the short version
is that WSL2 has no direct access to the microphone or speakers, and audio
reliability outranks model flexibility for a live demo.

Linux and macOS will run everything except the Windows-specific host API
handling, which degrades gracefully.

---

## 1. Python environment

Python **3.12**, deliberately not 3.13: CTranslate2, faster-whisper and
Chatterbox all ship reliable 3.12 wheels.

```bash
conda create -n parliamo python=3.12 -y
conda activate parliamo
```

If conda is not on your PATH (common on Windows), call it by full path or use
the Anaconda Prompt. Every command below can also be run as
`<env>\python.exe -m ...` without activating anything.

```bash
pip install -e .
pip install -r requirements/core.txt
```

---

## 2. Verify before building

```bash
python scripts/check_env.py
```

This is a gate, not a formality. It checks Python version, GPU and driver, free
VRAM, free disk, ffmpeg, audio devices, and which packages are present per
stage. It writes a JSON report to `runs/env/` so you can diff today's machine
against the one that worked last week.

Exit code `0` means every hard requirement passed. Warnings are expected until
the later stages are installed.

**Hard requirements**

| Check | Threshold | Why |
|---|---|---|
| Python | 3.12.x | wheel availability |
| Free disk | 40 GB | model weights, fine-tuning checkpoints, recordings |
| Total VRAM | 7000 MiB | three models resident simultaneously |
| ffmpeg | present | audio decoding for evaluation sets |

**Watch the idle VRAM line.** On the reference laptop a normal desktop session
(browser, music, chat clients) held 1465–1712 MiB before any model loaded. That
comes straight out of the model budget. Close them before measuring, and before
presenting.

---

## 3. Choose a microphone — by measurement

Two rules learned at rehearsal, before any measurement:

- **WASAPI for both directions.** WDM-KS never opens on this laptop (the
  picker greys it out). MME and DirectSound open, but the output then depends
  on Python feeding it in time: with the interpreter busy - as it is while
  recognising and translating - a silent MME stream starved 45 times in 8 s
  (150 callbacks instead of 610), WASAPI not once. That is the "choppy" sound.
- **Level before latency.** A microphone that delivers -40 dBFS of speech is
  worse than a slow one: Whisper invents text into the gaps. Raise the input
  volume until normal speech fills the upper third of Windows' meter.

```bash
python scripts/list_audio_devices.py
```

Windows exposes the same physical device once per host API, with very different
latency. On the reference laptop the identical microphone reported **90 ms on
MME** (which Windows names as the system default) and **2 ms on WASAPI**. The
device resolver corrects for this automatically, but you should still see it
once, because it explains why the config prefers explicit device names.

Then measure each candidate microphone:

```bash
python scripts/measure_audio_device.py --label lavalier
python scripts/measure_audio_device.py --label airpods --skip-latency
```

Three probes:

- **bandwidth** — where the microphone's usable content stops. Whisper consumes
  16 kHz audio, so anything below an 8 kHz roll-off is throwing away information
  the model would have used. A Bluetooth headset in HFP mode walls off at 4 kHz
  (narrowband) or 7–8 kHz (wideband mSBC).
- **noise floor** — sets the VAD threshold.
- **round-trip latency** — plays a chirp on a duplex stream and cross-correlates
  the echo.

### Reading the round-trip number

It is a **comparative** metric. The absolute value includes a constant this tool
cannot separate out (input DSP, plus PortAudio's cross-device duplex adapter).
Differences between devices are trustworthy; the floor is not.

On the reference laptop:

| Path | Median |
|---|---|
| built-in mic ← Realtek speakers | 428 ms |
| built-in mic ← HDMI monitor | 258 ms |

170 ms of difference from the output device alone — the vendor DSP effect chain.
Take the measured round-trip and put it in `config/local.yaml`:

```yaml
pipeline:
  output_latency_ms: 430
```

This is not cosmetic. The half-duplex gate keeps the microphone muted for
`output_latency_ms + half_duplex_tail_ms`. Leave it at `0` and the microphone
reopens while the previous sentence is still coming out of the speakers, which
is the feedback loop the gate exists to prevent. The code logs a warning until
you set it.

### On Bluetooth earphones

Opening a Bluetooth device's *microphone* on Windows forces the HFP/HSP profile.
Both directions collapse to compressed mono at 16 kHz or less, and 100–250 ms of
latency appears. This is a profile limitation, not a driver problem — there is no
software fix. Use a wired lavalier for anything that matters.

---

## 4. Machine-specific overrides

`config/local.yaml` is git-ignored and layered on top of `config/default.yaml`:

```yaml
audio:
  input_device: "Wireless GO"      # substring match, survives reboots
  output_device: "Speakers (Realtek"

pipeline:
  output_latency_ms: 430

vad:
  threshold: 0.55                  # from the noise-floor probe
```

Prefer name substrings over numeric indices — indices shift whenever a Bluetooth
device connects. Add `@WASAPI` to disambiguate when a name matches on several
host APIs.

Unknown keys are rejected at startup with the list of valid ones. A typo fails
immediately rather than silently changing behaviour mid-rehearsal.

---

## 5. The voice services

The voice runs in its own process and its own environment - Seed-VC needs an
older `huggingface_hub` than the pipeline does, RVC lives in Applio's, and
OmniVoice in a small venv of its own. All three speak one socket protocol, and
one runs on the GPU at a time: Start uses whichever is running, the configured
port first.

| service | env | port | what it is | per sentence |
|---|---|---|---|---|
| `scripts/voice_conversion_server.py` | `seedvc` | 8765 | Seed-VC, zero-shot: any consented reference, no training. The volunteer demo. | 0.93 s |
| `scripts/rvc_server.py --model presenter` | `Applio\env` | 8766 | RVC: the presenter's voice, trained once from their own recordings. | 0.43 s |
| `scripts/omnivoice_server.py` | `venvs\omnivoice` | 8767 | OmniVoice, zero-shot, *speaks* the sentence in the reference voice rather than converting Kokoro's. Presenter or volunteer without changing service. | ~1.2 s |

Start one before pressing Start in the application; the Setup tab's pre-flight
list says which answered, or that none did. Without one, every sentence is
spoken in the generic voice and the Live tab marks it so.

RVC needs a trained model under `Applio\logs\<name>\`. Training from the
presenter's recordings took 58 minutes on the reference GPU
([ADR 0012](adr/0012-rvc-beside-seedvc.md)); the weights are not in git.
RVC keeps the pitch of the voice it is given, and each language is spoken by a
different synthesiser voice (Kokoro if_sara 223 Hz, em_alex 139, Piper thorsten
125, dfki 105). With `tts.conversion.presenter_f0_hz` set in `local.yaml` (139
for the presenter), the pipeline measures the voice at Start and sends every
sentence with its own shift; the server's `--pitch` is only the fallback.

### OmniVoice

A venv on top of the main environment, so it adds two packages and changes
nothing else ([ADR 0015](adr/0015-omnivoice.md)):

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe -m venv --system-site-packages C:\Users\you\venvs\omnivoice
```

```bash
C:\Users\you\venvs\omnivoice\Scripts\python.exe -m pip install --no-deps omnivoice==0.2.1 accelerate
```

The weights (3.3 GB, `k2-fsa/OmniVoice`) go into the project's model cache
once, with the network; the server then runs with the Hub offline. Choose NLLB
600M beside it. Use `data/voices/phone-sentences.wav` as the presenter's
reference: cut on sentence boundaries, it clones far better than
`phone-12s.wav`, which starts mid-phrase. The first sentence in a new voice
takes 20-30 s while the reference is transcribed and encoded on the CPU.
Weights are CC-BY-NC.

## 5b. German and Turkish voices (Piper)

Kokoro has no German or Turkish. Piper speaks both, on the CPU:

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe -m pip install piper-tts==1.8.0
```

and two voice files into `models/piper/` (git-ignored), each `<name>.onnx` plus
`<name>.onnx.json` from `rhasspy/piper-voices`: `de_DE-thorsten-high` and
`tr_TR-dfki-medium`. A language whose voice file is missing is offered as
subtitles only. See [ADR 0014](adr/0014-spanish-german-turkish-and-calabrese.md).

Optional, for measurement only: `livekit` (echo cancellation,
`scripts/measure_echo_cancellation.py`). Not needed to run the talk.

## 6. Run the tests

```bash
pytest -q                     # everything
pytest -q -m "not audio"      # skip tests needing a sound card
pytest tests/test_audio_io.py -v
```

The audio tests drive real hardware and play a short quiet tone. The one to
watch is `test_capture_discards_audio_while_gate_is_closed` — it verifies the
anti-feedback guarantee end to end.
