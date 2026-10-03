# Model licences

The code in this repository is Apache-2.0. **The models it loads are not.**

One of them is non-commercial and sits on the critical path, so this matters
before anyone builds a product on top of this.

---

## 1. What ships

| Component | Model | Licence | Commercial use |
|---|---|---|---|
| ASR | `deepdml/faster-whisper-large-v3-turbo-ct2` | MIT (weights); Whisper is MIT | **yes** |
| VAD | Silero VAD | MIT | **yes** |
| MT | **NLLB-200-distilled-600M** | **CC-BY-NC-4.0** | **no** |
| TTS | Kokoro-82M | Apache-2.0 | **yes** |
| TTS, German / Turkish | Piper runtime (`piper-tts`) | GPL-3.0 | separate package |
| Voice, German | `de_DE-thorsten-high` | dataset CC0 | **yes** |
| Voice, Turkish | `tr_TR-dfki-medium` | dataset CC-BY-NC-SA-4.0 | **no** |
| Voice conversion | RVC via Applio (presenter's trained voice) | MIT (code) | the model is the presenter's voice, used with their consent |
| Voice conversion | Seed-VC | GPL-3.0 | see §3 |
| Voice (speaks the sentence) | OmniVoice (`k2-fsa/OmniVoice`) | code Apache-2.0; **weights CC-BY-NC-4.0**; audio tokenizer: Boson Higgs Audio 2 Community Licence | non-commercial, like NLLB. Resellers quote Apache-2.0 as permitting commercial use - true of the code only. ADR 0015 |
| Runtime | CTranslate2 | MIT | yes |
| Runtime | PyTorch, sounddevice, soxr | BSD/MIT | yes |

## 2. Also present, not on the live path

| Component | Licence | Note |
|---|---|---|
| Chatterbox Multilingual v3 | MIT | kept for the ethics demo — watermarks by default |
| NLLB-200-distilled-1.3B | CC-BY-NC-4.0 | recommended beside RVC, ADR 0013 |
| Qwen3-TTS 0.6B | Apache-2.0 | measured, too slow live; possible for pre-rendered clips (11-model-survey) |
| HY-MT1.5-1.8B | Tencent HY licence (excludes the EU) | measured, not adopted (11-model-survey) |
| livekit (WebRTC AEC) | Apache-2.0 | measurement script only |
| FLEURS dataset | CC-BY-4.0 | evaluation |
| FLORES-200 | CC-BY-SA-4.0 | Friulian evaluation |

---

## 3. The two that constrain you

### NLLB-200 is CC-BY-NC

**Non-commercial.** This is fine for a research presentation and a public
GitHub repository, and it is a hard blocker for a product, a paid service, or
anything bundled into commercial software.

It is also not replaceable for this project's purpose. NLLB-200 is the **only**
pretrained translation model that has seen Friulian, which is why ADR 0004
selected it and why no amount of quality argument would change it. Any
commercial derivative of this work needs a different translator, and would lose
Friulian in the process.

Commercially usable alternatives for the well-resourced pairs, none of which
cover Friulian:

* **Opus-MT** (Helsinki-NLP) — CC-BY-4.0, per-pair models, small and fast
* **MADLAD-400** — Apache-2.0, 419 languages
* **M2M-100** — MIT

### Seed-VC is GPL-3.0

GPL obligations attach to distribution of a derivative work. This project does
not link it into the same process — it runs in a **separate environment behind
a socket**, for dependency reasons that predate any licence consideration.

That separation is architecturally convenient here, but do not lean on it as a
legal opinion. If you distribute something built on this, get advice.

---

## 4. Voice models are not just a licence question

The RVC models present on the development machine from an earlier experiment
(three models of public figures' voices) raise a question no licence
answers: a model trained on a real person's voice encodes that person's
biometric identity, whatever the code licence says.

They are **not used in this project** and should not be on the presentation
machine. See [07-ethics-and-consent.md](07-ethics-and-consent.md).

---

## 5. What to check before reusing this

- [ ] Is this commercial? If so, NLLB has to go, and Friulian goes with it.
- [ ] Are you distributing a binary that includes Seed-VC? Get advice on the GPL.
- [ ] Are you shipping trained voice models? That is consent, not licensing.
- [ ] Are you republishing evaluation data? FLEURS is CC-BY, FLORES CC-BY-SA.

---

## 6. Keeping this honest

Licences change and model cards get updated. Every row above was read from the
model card at the time it was added, and nothing here is machine-checked — if a
figure in this table is load-bearing for you, verify it at the source rather
than trusting this file.
