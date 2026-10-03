# Adding a language

Nothing in the pipeline is hardcoded to Turkish or Italian. The language pair
lives in `config/default.yaml` and every stage reads it from there.

But "add a language" means two very different jobs depending on the language,
and it is worth being honest about which one you are facing:

* **A well-resourced language** — a config change, then measurement.
* **A low-resource language** — a data problem, and the models will not save
  you. The Friulian branch of this project is the worked example.

---

## 1. A well-resourced language

### What to change

```yaml
asr:
  language: de              # what is spoken on stage

mt:
  source_lang: de
  targets: [en]

tts:
  language: en
  voice: af_heart           # a voicepack that exists for that language
```

### What each stage needs from you

| Stage | Requirement | How to check |
|---|---|---|
| ASR | Whisper covers the language | it covers 99; check the WER, not the list |
| MT | an NLLB FLORES-200 tag | `LANG_TAGS` in `mt/ctranslate2_nllb.py` |
| TTS | a Kokoro voicepack, or a Piper voice for what Kokoro lacks | `LANG_CODES` in `tts/kokoro_backend.py`; `VOICES` in `tts/piper_backend.py`; `tts.voices` in the config |
| ASR hotwords | a list for that language, or none | `hotwords.<lang>.txt`; the Turkish list is applied to Turkish only |
| Voice conversion | nothing — it is language-agnostic | — |

Voice conversion is the easy part: it transfers timbre, not phonetics, so it
works on any language the synthesiser can produce. RVC keeps the pitch it is
given, so pick a synthesiser voice near the presenter's pitch where there is a
choice (em_alex for Spanish sits exactly at 139 Hz); the per-sentence shift
covers the rest when `presenter_f0_hz` is set.

`tests/test_config_matches_code.py` checks all three mappings against the
config, so a language the pipeline cannot actually speak fails at test time
rather than on stage.

### Then measure, because the list is not the answer

"Whisper supports 99 languages" is a claim about a training set. Measured on
this project's own audio, `large-v3` and `large-v3-turbo` differed by 0.56 WER
points on public data and **3.93 points** on the presenter's real voice through
a real microphone.

So:

```bash
python scripts/bench_asr.py --fleurs-config de_de --limit 200
python scripts/record_script.py --script data/insitu/script-de.txt
python scripts/bench_asr.py --manifest data/insitu/<run>/manifest.tsv
python scripts/bench_mt.py --source de --target en --limit 200
python scripts/bench_tts.py --language en
```

Public data ranks the models. In-situ audio decides. The whole sequence takes
an afternoon and has overturned a decision every time it has been run here.

### Two traps that are language-specific

**Normalisation.** `parliamo/eval/normalize.py` is Turkish-aware because Turkish
needed it — `str.lower()` merges the dotless `I` with the dotted `i`, inventing
errors in the reference and hiding them in the hypothesis *at the same time*.
Every language has something like this. German has ß/ss; Greek has final sigma.
If you do not know what yours is, your WER is measuring your normaliser.

**Number formatting.** Turkish groups thousands with a full stop, so `40.000`
means forty thousand. A naive `\d+` regex spelled it "kırk sıfır" and corrupted
the reference text that every model was being scored against. Check what your
language does before trusting a single WER figure.

---

## 2. A low-resource language: the Friulian case

Friulian (*furlan*) is spoken by roughly 600,000 people in Friuli Venezia
Giulia. It is the reason this project's translation stage is NLLB and not
something better on `tr→it`.

### What exists

| Resource | Status |
|---|---|
| **NLLB-200** `fur_Latn` | the **only** pretrained MT model that has seen it |
| FLORES-200 `fur_Latn` | 997 dev + 1012 devtest sentences, human-translated |
| OPUS | ~1000 natural sentence pairs |
| *La Bibie* (2019) | aligned to the Italian CEI Bible, ~31k verses |
| `fur.wikipedia` | usable for back-translation |
| **Any TTS model** | **none exists** |
| **Any ASR model** | none; Whisper has not seen it |

Every other candidate translation model covers 33–140 well-resourced languages
and none of them include Friulian. That single fact decided the MT backend
before quality was measured at all.

### The approach

The plan was to pivot through Italian (`tr → it → fur`), run it on the CPU and
fine-tune NLLB on the Bible alignment. Measured first, none of it was needed
([ADR 0010](adr/0010-friulian-baseline.md)):

1. **Translate directly, `tr → fur`.** The pivot took 585 ms against 331 and
   introduced a tense error by inheriting the Italian hop's drift. On FLORES,
   Turkish → Friulian scores chrF++ 42.7 (600M) and 46.3 (1.3B), against
   47.5 / 49.2 for Italian ([ADR 0014](adr/0014-spanish-german-turkish-and-calabrese.md)).
2. **Live, with the main translator.** Friulian is a Setup target like any
   other, on the same NLLB instance and GPU; no separate model.
3. **No fine-tuning.** The baseline is real Friulian with ordinary mistakes. If
   a native speaker finds the prepared sentences inadequate, the resources
   above are where a fine-tune would start - the Bible alignment diluted with
   back-translated Wikipedia, or it will speak in Bible register.
4. **Evaluate with chrF++, not BLEU.** Friulian is morphologically rich and the
   test sets are small; BLEU is close to noise at this scale. See
   `parliamo/eval/mt_metrics.py`.

### Speech, honestly

There is no Friulian TTS model. Friulian text is synthesised through the
**Italian** frontend: the prosody is plausible, the phonetics are wrong.

The code does not hide this. `KokoroBackend.resolve_language("fur")` returns the
Italian code *and a note*, and the returned `Speech` is labelled
`"fur (spoken as it)"` so a caller writing subtitles can state it rather than a
log line nobody reads.

This is disclosed on stage rather than hidden, because it is a concrete
illustration of the point the talk is making: the gap between languages that
have models and languages that do not is not a technical inevitability, it is a
consequence of where effort has been spent.

### Worth contacting

**ARLeF** (Agjenzie regjonâl pe lenghe furlane) is building Friulian MT and TTS
under the DIGI R-L-F project. Anyone reproducing this branch should talk to them
before building a dataset from scratch.

---

## 3. What "supported" should mean before you go on stage

A language is ready when all of these are true and **measured**:

- [ ] WER measured on the actual speaker, microphone and room — not on FLEURS
- [ ] chrF++ measured on a test set in the target language
- [ ] Synthesis audible and correct, listened to by someone who speaks it
- [ ] Any substitution (like Friulian-as-Italian) written into the slides
- [ ] `tests/test_config_matches_code.py` passing with the new config
- [ ] One full `live_translate.py --file` replay of real recorded speech

The last one is not optional. Every stage in this project passed its own
measurement while the pipeline as a whole still contained a bug that only a
full replay exposed.
