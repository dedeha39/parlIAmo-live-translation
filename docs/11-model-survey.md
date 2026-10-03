# Model survey, September 2026

What is newer than what the pipeline runs, whether any of it would help this
talk, and what was measured rather than read. Written 2026-09-23 as part of the
full audit ([00-state.md](00-state.md), "Audit").

The constraints that decide everything below: one laptop, **8 GB GPU**,
**Windows native** (ADR 0001 — no vLLM), **no network at run time**, a
latency budget in which every 0.1 s is felt, and an Italian audience.

---

## Summary

| stage | now | candidate | verdict |
|---|---|---|---|
| recognition | Whisper large-v3-turbo | Qwen3-ASR 1.7B / 0.6B | **no change** — 1.7B ties Whisper-large-v3 on published numbers at more memory; 0.6B is worse |
| recognition | | Parakeet / Canary v3 | **not applicable** — no Turkish |
| translation | NLLB-600M | **NLLB-1.3B** | **adopt with RVC** — measured, [ADR 0013](adr/0013-rvc-frees-memory-for-nllb-1.3b.md) |
| translation | | HY-MT1.5-1.8B | **measured, no gain** — chrF++ 45.30 against NLLB-600M's 45.38 and 1.3B's 47.11; more fluent in places, ungrammatical in others |
| translation | | TranslateGemma 4B | **no** — measured in ADR 0004; too slow and too large here |
| synthesis + voice | Kokoro + RVC / Seed-VC | Qwen3-TTS 0.6B Base | **measured, too slow live** — RTF 2.2–2.4 on this GPU (a 6 s sentence took 14.5 s; Kokoro + RVC 0.54 s); intelligibility excellent — usable for pre-rendered clips, not for the live path |
| room | half-duplex gate | WebRTC echo cancellation (`livekit` APM) | **promising, not decided** — at low volume it removed everything the pipeline would hear, and cost 5% → 15% WER when talking over it; performance volume not yet measured |
| Calabrese | — | NLLB Sicilian (`scn_Latn`), MADLAD-400 | **no model** — MADLAD has no `nap`/`scn`; NLLB's Sicilian is Corsican in 148-174/200 sentences ([ADR 0014](adr/0014-spanish-german-turkish-and-calabrese.md)) |
| German / Turkish voice | — | Piper (CPU) | **adopted** — [ADR 0014](adr/0014-spanish-german-turkish-and-calabrese.md) |
| synthesis + voice | Qwen3-TTS 0.6B vs OmniVoice, head to head | same reference, same 8 short talk sentences, several seeds (2026-09-26) | **Qwen3-TTS reads right, OmniVoice is fast** - Qwen WER 4.3%, 0 of 16 renders wrong, 16 s per sentence; OmniVoice WER 27-28% and 9 of 40 renders a different sentence from `phone-sentences`, 13.1% and 4 of 40 from `ref-12s`, 0.85 s. Qwen for prepared clips, OmniVoice where time matters and a render can be checked first |
| synthesis + voice | Kokoro + RVC / Seed-VC | **OmniVoice** (k2-fsa, March 2026) | **added as a third voice service** — ~1.2 s per sentence, 2.5 s end to end, WER 1.7% read back once digits are spelled; no Friulian in its 646 languages; weights CC-BY-NC — [ADR 0015](adr/0015-omnivoice.md) |
| end-to-end speech translation | — | Seamless, Hibiki | **no** — no Turkish→Italian with cloning at this quality |

---

## Measured today, no download needed

### RVC holds 0.7 GB, not 3

The rehearsal guide said both voice services hold ~3 GB. Measured with
`nvidia-smi` every 50 ms: RVC +660 MiB loaded, **+1204 MiB at the peak of a
conversion**; Seed-VC ~2900 MiB held. That 2 GB is exactly what NLLB-1.3B was
rejected for. With RVC, 1.3B peaks the whole machine at 6942 MiB of 8188 and
costs 0.2 s per sentence; read sentence by sentence it fixed three meaning
errors and introduced one. **Choose NLLB 1.3B on the Setup tab when the voice
service is RVC.** [ADR 0013](adr/0013-rvc-frees-memory-for-nllb-1.3b.md).

### "İzlediğiniz için teşekkür ederim" from a breath

Whisper was trained on subtitled video, and on the first fraction of a second
of an utterance it can produce what a subtitle track ends with. On the
reference recording it did — "thank you for watching", 13 syllables in 0.93 s —
and "Friulian Friuli Börgesine Kodur" at the same position elsewhere. A reading
faster than 12 syllables per second is now dropped (`asr/plausibility.py`):
real speech on that recording topped out at 9.7, every committed sentence under
7.5. WER unchanged at 11.41%; both inventions gone.

### The word the talk is about

The opening sentence says *taklit edilebildiğini* (can be imitated); the
recogniser hears **takdir** edebildiğini (can appreciate), and both translators
faithfully render "appreciate the human voice". Tried and measured:

| change | WER, 68 s | opening sentence |
|---|---|---|
| none | 11.41% | *takdir* |
| `taklit` added to hotwords | **12.08%** — worse | still *takdir* |
| topic `initial_prompt` | 11.41% | still *takdir* |

Neither helps; the hotword is reverted. What does work on the same recording:
*kopyalanabildiğini* is recognised correctly every time it is said. **Say
"kopyalanabildiğini" (or "klonlanabildiğini") in the opening sentence.** A
wording change is free; a model change for one word is not.

### What did not change

On the same recording, the chunk-mode alignment fix changed nothing measurable;
the spoken warm-up saved ~0.15 s on Seed-VC's first sentence and nothing on
RVC's. Both are in [00-state.md](00-state.md) with their numbers.

---

## Recognition

**Qwen3-ASR** (Alibaba, January 2026, Apache-2.0; 0.6B and 1.7B; 30 languages
including Turkish). Published WER, averaged over the eight FLEURS languages
that include Turkish: **1.7B 6.62, Whisper-large-v3 6.85, 0.6B 10.37**. The
1.7B is a tie with large-v3, which this project already measured as too slow
(724 ms p95 against turbo's 293, ADR 0002) — and 1.7B parameters in bf16 is
more memory than large-v3-turbo int8. Streaming needs vLLM, which does not run
natively on Windows. **Not worth it.** The errors that matter here — *takdir*
for *taklit*, driver gating on the laptop microphone — are acoustic, and a
microphone fixes more of them than a model.

**Parakeet-TDT 0.6B v3 / Canary-1B v2** (NVIDIA): 25 European languages,
Turkish not among them (checked on the model card; one search summary claimed
otherwise).

## Translation

**NLLB-1.3B** — see above; adopted with RVC.

**HY-MT1.5-1.8B** (Tencent, December 2025): a translation-only model, 33
languages including Turkish and Italian, GGUF at Q4_K_M (1133 MB). Measured
2026-09-23 through Ollama, Q4_K_M, the same 200 FLEURS pairs as ADR 0007,
sentence-split as live:

| | chrF++ | BLEU | single sentence p95 | memory |
|---|---|---|---|---|
| NLLB-600M | 45.38 | 18.91 | 500 ms | 844 MiB |
| **HY-MT1.5-1.8B Q4** | 45.30 | 16.50 | 758 ms | ~1.7 GB, in Ollama's process |
| NLLB-1.3B | **47.11** | **21.34** | 929 ms | +1118 MiB |

On the eleven presentation sentences (mean 379 ms) it read more fluently than
either NLLB in three places - "offrire opportunità reali nelle lingue in cui il
numero di parlanti diminuisce" is the best rendering any model produced - and
made errors neither NLLB made in three others: *la prima cosa a cui mi sono
chiesta* (broken, and in a woman's voice), *quasi nessun spazio*, *questa ...
un problema*. It opened the talk with "Ciao". The negation that sank
TranslateGemma in ADR 0004 it got right. Net: level with NLLB-1.3B at best,
behind on the benchmark, one more process and ~0.6 GB more. **Not adopted.**

Two things had to be fixed before it could be measured at all, both recorded
because they would bite any other Ollama model:

- Ollama's chat template for HY-MT's architecture drops the prompt entirely -
  every answer was "onse }". The backend now sends such models raw with the
  turn markers written out (`RAW_WRAPPERS` in `mt/ollama_backend.py`).
- The backend addressed Ollama as `localhost`, which on Windows tries IPv6
  first and waited **~2 s on every request**: 2274 ms per sentence against
  222 ms through `127.0.0.1`. ADR 0004's TranslateGemma latency (2671 ms)
  was measured through the same default; its rejection stands on quality and
  memory, and ADR 0004 now says so.

*License:* the Tencent HY license states it "does not apply in the European
Union, United Kingdom and South Korea". The talk is in Italy. The presenter
has said this does not matter for a non-commercial talk; recorded here so the
choice is visible, not re-argued. The same terms very probably cover Hy-MT2
(May 2026: 1.8B, 7B, 30B-A3B, 33 languages).

**TranslateGemma** (Google, January 2026, Gemma license; 4B/12B/27B, 55
languages). The 4B was measured in [ADR 0004](adr/0004-mt-backend.md) at Q8
through Ollama: wrong target language on 2 of 6 sentences, one dropped
negation, and 2671 ms median - of which ~2 s was the `localhost` fault above.
Re-run today through `127.0.0.1` with the same prompt it ran on past each
sentence into other languages until the token limit. The model card requires a
structured chat template (`source_lang_code`/`target_lang_code`) that the ADR
0004 prompts did not use, which plausibly explains that; it does not change the
2.9 GB it occupied at Q8 or the ~2.5 GB it needs at Q4. **Not on this card.**

## Synthesis and the voice

**Qwen3-TTS** (Alibaba, January 2026, Apache-2.0). 0.6B and 1.7B "Base"
models clone a voice from ~3 s of reference audio; ten languages including
Italian; published Italian WER 1.53 (0.6B) and 0.95 (1.7B) — very
intelligible. One model would replace Kokoro *and* the voice service, and
~2–2.5 GB of VRAM would replace Kokoro's 0.74 GB plus RVC's 0.7–1.2 GB.

**Measured 2026-09-23** on this laptop: the official `qwen-tts` package in its
own venv (it needs transformers 4.57; the pipeline runs 5.16, so it was not
installed there), bf16, PyTorch attention - FlashAttention 2 has no official
Windows wheel. The package's own documentation says its "streaming" only
simulates streaming *text input*; audio comes out whole.

| sentence | Qwen3-TTS 0.6B, with transcript | voice print only | Kokoro + RVC (today's path) |
|---|---|---|---|
| 2.8 s | 6.59 s | 6.87 s | 0.43 s |
| 6.2 s | 14.50 s | 21.18 s | 0.54 s |
| 4.0 s | 9.53 s | 13.35 s | 0.46 s |

**RTF 2.2-2.4: more than twice slower than real time.** FlashAttention's
reported 30-40% would not bring it under 1.0, and ADR 0005 already showed what
RTF 1.3 does to this pipeline. Peak memory 2.5 GB. What it does well: Whisper
read all six Italian outputs back almost word for word. **Not for the live
path.** It could render clips *before* the talk - the "fifteen seconds is
enough" demonstration in the presenter's own voice, prepared at home - which
needs no latency at all. The presenter has three comparison files; which one
sounds like them is theirs to say.

## The room: echo cancellation instead of the gate

The half-duplex gate closes the microphone while the translation plays, so the
system never translates itself. It also means the presenter cannot talk over
the translation — the streaming modes the presenter asked for are limited by
it, and the gate card counts the words it throws away.

WebRTC's echo canceller (AEC3) does the opposite: it knows exactly what the
speakers are playing and subtracts it from the microphone. It ships inside the
`livekit` Python package (11 MB Windows wheel, runs locally, no LiveKit
server): `AudioProcessingModule(echo_cancellation=True)`,
`process_reverse_stream()` for the speaker signal, `process_stream()` for the
microphone, 10 ms frames, `set_stream_delay_ms()`.

The risk is this laptop: the speaker path carries ~430 ms of vendor DSP with
chirps measured between 229 and 597 ms. Whether AEC3's delay estimator tracks
that is a measurement, and `verify_feedback_loop.py` already has the harness:
play speech, record, and ask whether Whisper hears anything in the residual.
If it holds, the gate becomes a fallback instead of the rule.

**Measured 2026-09-23, laptop speakers at low volume** (the presenter chose to
leave performance volume for the rehearsal),
`scripts/measure_echo_cancellation.py`:

- In shared mode the Intel microphone array's **own driver** takes the
  speakers - and then the whole room - to digital silence (-120 dBFS) within
  seven seconds, and stays there after playback stops. That is driver
  processing, and probably part of why references recorded on it were half
  holes. Exclusive mode bypasses it; the script uses exclusive.
- Echo 10.6 dB over the room floor, arriving 232 ms after the reference.
  AEC3 took it **to the floor** (-75.1 against -74.9 dBFS).
- Without AEC, Silero + Whisper would have passed **two segments** of the
  echo and transcribed the Italian - the feedback loop the gate exists for.
  With AEC: **none**.
- Talking over it, at the presenter's recorded level: WER **5.1% without AEC,
  15.3% with it**. AEC3 removes the echo and some of the presenter with it.
  The gate today removes *all* of what is said during playback, so for the
  streaming modes 85% may still be the better trade - at performance volume,
  where the echo is 30 dB louder and laptop speakers distort, it may not be.

The rehearsal guide has it as step 9b.

## End-to-end speech translation

SeamlessStreaming (Meta, 2023) accepts Turkish speech and produces Italian
speech, but under a non-commercial license, without the presenter's voice, and
below a cascade's quality. Kyutai's Hibiki family is French→English. Nothing
found does Turkish→Italian with cloning at a quality this cascade does not
already beat.

---

## Sources

- Qwen3-ASR: [GitHub README](https://github.com/QwenLM/Qwen3-ASR), [technical report](https://arxiv.org/pdf/2601.21337)
- Qwen3-TTS: [GitHub README](https://github.com/QwenLM/Qwen3-TTS), [streaming fork, RTF figures](https://github.com/xmillogx-cmd/Qwen3-tts-streaming)
- HY-MT1.5: [GitHub](https://github.com/Tencent-Hunyuan/HY-MT), [technical report](https://arxiv.org/pdf/2512.24092), [license](https://github.com/Tencent-Hunyuan/HY-MT/blob/main/License.txt); Hy-MT2: [paper](https://arxiv.org/html/2605.22064)
- TranslateGemma: [model card](https://huggingface.co/google/translategemma-4b-it), [technical report](https://arxiv.org/pdf/2601.09012)
- Parakeet-TDT 0.6B v3: [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- LiveKit APM: [source](https://github.com/livekit/python-sdks/blob/main/livekit-rtc/livekit/rtc/apm.py), [docs](https://docs.livekit.io/reference/python/livekit/rtc/index.html)
