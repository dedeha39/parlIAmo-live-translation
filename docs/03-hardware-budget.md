# Hardware budget

8 GB of VRAM is the binding constraint on this project. Everything else —
which recogniser, which translator, whether voice cloning is possible at all —
was decided against this table.

Reference machine: **NVIDIA RTX 4070 Laptop, 8188 MiB**, Intel i9-13900HX,
32 GB RAM, Windows 11.

---

## 1. The rules

Established by [adr/0003-vram-budget.md](adr/0003-vram-budget.md) after an
estimated budget eliminated a model that in fact fits comfortably.

1. **No component enters the budget as an estimate.** Anything unmeasured is
   named as unmeasured, and a decision that depends on it waits.
2. **Measure co-resident, not summed.** CUDA context, allocator fragmentation
   and cuBLAS workspaces are paid once, not once per model.
3. **Budget against the peak desktop baseline**, not a lucky reading.
4. **An out-of-memory during measurement is a result, not a failure.** It is
   the answer to whether a configuration fits.

Rule 2 has teeth. The original budget summed per-model estimates and was wrong
by **3.3× on translation and 1.8× on synthesis, in opposite directions** — so
the total looked roughly right by luck while both halves were wrong.

---

## 2. Measured, co-resident

`scripts/measure_vram_budget.py`, loading the shipped stack into one process in
sequence, with the Seed-VC service already running in its own:

| Stage | Added | Running total |
|---|---|---|
| Desktop session, apps closed | — | ~1200 MiB |
| Seed-VC service (separate process) | ~2900 | ~4130 |
| ASR — `large-v3-turbo` int8_float16 | 1213 | 5346 |
| MT — NLLB-600M on CUDA | 844 | 6190 |
| TTS — Kokoro-82M | 348 | 6538 |
| TTS generation activations | 396 | 6934 |
| One voice conversion (activations) | 358 | **7292** |

**Peak 7292 MiB of 8188. Headroom 896 MiB.**

Reproduce it with:

```bash
python scripts/measure_vram_budget.py --tts kokoro --conversion \
    --conversion-reference data/insitu/<run>/003.wav
```

Start the conversion service first, in the other environment — the script says
so if it is not reachable, and reports a peak that **excludes** conversion
rather than pretending.

---

## 3. Per-component figures

Measured alone, for comparison. Do not add these up — see rule 2.

| Component | VRAM | Note |
|---|---|---|
| Whisper `large-v3-turbo` int8_float16 | 1213 | the live recogniser |
| Whisper `large-v3` int8_float16 | 2113–2181 | rejected on latency, not memory |
| NLLB-600M, CUDA | 751–844 | the live translator |
| NLLB-1.3B, CUDA | 1713–1962 | better quality; fits beside RVC, not beside Seed-VC — ADR 0013 |
| NLLB-600M, CPU | 0 | costs ~100–150 ms per sentence |
| Kokoro-82M | 348–587 | the live synthesiser |
| Seed-VC | ~2900 | **largest single consumer**, separate process |
| RVC (presenter's voice) | 660 held, 1204 at a conversion's peak | separate process — ADR 0013 |
| OmniVoice | 1950 held, 2100–2300 at a sentence's peak | separate process; references encoded on the CPU, which on the GPU peaked at 5.6 GB (12 s) and 12.5 GB (35 s). Whole machine with NLLB-600M: 6605 — ADR 0015 |
| Piper (German, Turkish) | 0 | runs on the CPU — ADR 0014 |
| Chatterbox Multilingual v3 | 2824–3368 | ethics demo only |

Treat every figure as **±150 MiB**. The same model measured 950, 1011, 1084 and
1292 MiB across runs, because `nvidia-smi` reports the whole card and other
processes allocate during the measurement window. Size the budget from the
upper end.

---

## 4. The desktop is the noisiest line

Measured across three runs minutes apart, with nothing but browser tabs
changing: **1476, 1565 and 1857 MiB** — a 381 MiB spread.

This is why a margin under ~500 MiB is not a margin. A configuration with
400 MiB of headroom works on a quiet machine and fails on a busy one, which is
the worst kind of "works": it passes every test you run and fails in front of
an audience.

Two decisions rest on exactly this:

* **large-v3** was kept off the live path partly because its margin sat inside
  that spread (ADR 0003).
* **NLLB-1.3B** was kept off for the same reason — 359–700 MiB beside Seed-VC
  (ADR 0007). Beside RVC, which holds ~2 GB less, the whole machine peaked at
  6942 MiB with it, and it is recommended there (ADR 0013).

**Before presenting: close the browser, the chat clients and the music player.**
`scripts/check_env.py` warns when more than 1200 MiB is already allocated.

---

## 5. Levers, in the order to pull them

If something needs to grow:

| Lever | Frees | Costs |
|---|---|---|
| Close desktop apps | 300–650 MiB | nothing |
| MT to CPU (`mt.device: cpu`) | ~850 MiB | ~100–150 ms per sentence |
| Drop voice conversion | ~2900 MiB | the cloned voice — the demo's centrepiece |
| Chatterbox out of the live path | already done | — |

MT-on-CPU is the first real lever and it is measured for 600M. It is **not**
available for 1.3B: p95 rises to 3023 ms, more than the entire pipeline budget.

---

## 6. The latency budget it has to fit inside

VRAM is not the only contract. Per sentence, measured:

| Term | Time | Kind |
|---|---|---|
| Commit wait | 0.70 s | **policy** — waiting to be sure the sentence ended |
| Recognition | 0.29 s | compute, p95 |
| Translation | 0.49 s | compute, p95 |
| Synthesis | 0.15 s | compute |
| Voice conversion | 0.92 s | compute, incl. 8–34 ms IPC |
| **Total** | **2.40 s** | target was 2–3 s |

Only the first term is a choice. `vad.min_silence_ms: 500` gives 2.20 s at the
cost of cutting speakers off mid-clause more often.

Confirmed by replaying five recorded sentences through the whole pipeline:
1.21–1.44 s without voice conversion, which lands at ~2.3 s with it.

---

## 7. The number that is still zero

`pipeline.output_latency_ms` is **0** and must be measured in the venue.

The half-duplex gate keeps the microphone muted for
`output_latency_ms + half_duplex_tail_ms`. PortAudio reports only its own
buffers — 3 ms on this laptop — and cannot see the vendor DSP chain. The real
acoustic round-trip measured **428 ms through the Realtek speakers and 258 ms
through an HDMI monitor**: a 170 ms difference caused purely by the output
device.

Left at 0, the microphone reopens while the previous sentence is still audible
in the room, and the system begins translating its own output.

```bash
python scripts/measure_audio_device.py --label venue
```

Then put the measured round-trip in `config/local.yaml`:

```yaml
pipeline:
  output_latency_ms: 430
```

Using the full round-trip rather than an output-only figure is deliberate: it
over-estimates, and the failure mode of over-estimating is a slightly longer
deaf period, while the failure mode of under-estimating is a feedback loop.

`scripts/live_translate.py` prints a warning on every run until this is set.
