# ADR 0001 — Run natively on Windows, not under WSL2

**Status:** accepted · 2026-08-29

## Context

The pipeline needs CUDA, low-latency microphone capture, and low-latency
speaker playback, on a Windows 11 laptop.

WSL2 is attractive because it unlocks `vLLM`, which is the only supported way to
run Qwen3-ASR in streaming mode, and is generally the better-supported target
for Python ML tooling.

## Decision

Run everything natively on Windows. Do not depend on any component that requires
vLLM.

## Reasoning

WSL2 has no direct access to audio devices. Microphone and speaker I/O has to be
bridged — PulseAudio or PipeWire over TCP to a Windows-side server. That bridge
adds latency, adds a process that can die, and adds a failure mode that shows up
as silence rather than an error.

For a live presentation, audio path reliability outranks model selection. A
slightly worse ASR model that always works beats a slightly better one that
occasionally produces silence in front of an audience.

The native Windows toolchain covers what we need:

| Component | Windows native |
|---|---|
| faster-whisper / CTranslate2 | yes, first-class |
| llama.cpp with CUDA | yes, first-class |
| Chatterbox (PyTorch) | yes |
| sounddevice / PortAudio WASAPI | yes, 2–3 ms reported latency |
| vLLM | **no** |

## Consequences

- Qwen3-ASR streaming is unavailable. It remains a Phase 1 bake-off candidate in
  non-streaming mode via `transformers`.
- `flash-attn` is awkward on Windows; components must work without it.
- Model selection is constrained to formats llama.cpp and CTranslate2 support —
  which in practice means GGUF and CT2, both well supplied.

## Escape hatch

If Phase 1 shows Turkish WER is unacceptable with the Windows-native ASR
options, the fallback is a **split deployment**: audio capture and playback stay
in a Windows process, models are served from WSL2 over a localhost socket.
Localhost forwarding costs about a millisecond, and the audio path stays native.

This keeps the option open without paying for it now.

## Revisit if

- vLLM ships usable Windows support
- Turkish WER on the Windows-native path fails the Phase 1 exit criterion (≤ 8%)
