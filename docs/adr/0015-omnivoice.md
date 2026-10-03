# 15. OmniVoice as a third voice service: the sentence spoken, not converted

Date: 2026-09-26

Status: accepted as an option beside RVC and Seed-VC. Which voice the talk
uses is the presenter's ear; the comparison files are in
`runs/voice-compare-20260926/`.

## Context

The presenter's first rehearsal with RVC through the speakers sounded "slowed
down and robotic". Half of that was a bug - RVC's 48 kHz audio played into a
24 kHz stream, half speed an octave down (fixed the same day, see
[00-state.md](../00-state.md)). The other half is what conversion is: RVC and
Seed-VC repaint the timbre of a sentence Kokoro or Piper has already spoken,
and the rhythm and intonation stay the synthesiser's.

The presenter asked for a better clone, free and local, and was willing to
wait for it ("30 seconds is fine, I will show it to the guests"). OmniVoice
had been released in the meantime and they asked for it by name.

## What it is

OmniVoice (k2-fsa - the Kaldi / k2 group - 31 March 2026): zero-shot
text-to-speech that clones a voice from a few seconds of reference and its
transcript, in 646 languages. From its language table: Italian 9,402 hours of
training data, Turkish 125, Sicilian 13. **No Friulian, no Neapolitan or
Calabrese** - the project's "no Friulian voice exists" stands.

Licences, read from the files rather than the landing page: the code is
Apache-2.0; the **weights are CC-BY-NC** "due to constraints from its training
data"; the audio tokenizer it ships is under the Boson Higgs Audio 2 Community
Licence (Llama 3 based). Fine for a non-commercial talk, like NLLB. A hosted
site that resells it (omnivoice.app) tells its customers that Apache-2.0
permits commercial use - true of the code only.

## Measured (RTX 4070 Laptop, fp16, 2026-09-26)

**Memory.** 1.95 GB resident, 2.1-2.3 GB at the peak of a sentence. Encoding a
*reference* on the GPU peaked at 5.6 GB for 12 s and 12.5 GB for 35 s - the
whole card, beside the pipeline. References are therefore cut to at most 12 s
at word boundaries and encoded on the CPU (peak stays 2.1 GB).

**Steps.** Italian, eight sentences, from the presenter's Turkish reference,
Whisper reading each render back:

| steps | per sentence | WER |
|---|---|---|
| 8 | 0.5-0.9 s | 17.5% |
| **16** | **~1.0 s** | **7.5%** |
| 32 | ~1.9 s | 15.0% |

More steps is not better here; 16 is the default (`--steps`).

**Where the reference is cut matters more than anything else measured.** The
Seed-VC reference (`phone-12s.wav`) starts mid-phrase. With it OmniVoice lost
the first word of 6 of 6 Turkish test sentences (WER 25.6%). Cut on sentence
boundaries from the same phone recording (`phone-sentences.wav`, 10.5 s): 1 of
6, WER 7.0-9.3%. New references are cut the same way automatically - the start
at the first word, the end on a sentence end where one exists.

**Numbers.** The translator writes digits; OmniVoice read "15 secondi" as "chi
me è secondo". Spelled out first (`tts/numbers.py`, num2words): end-to-end WER
5.6% → **1.7%** over 178 words.

**End to end**, the 68 s reference recording, NLLB 600M, zero conversion
failures in both runs:

| | mean lag | p95 | voice stage | machine GPU peak |
|---|---|---|---|---|
| RVC (ADR 0012/0013) | 1.58-1.72 s | 1.80 s | 0.43 s | 6.9 GB with NLLB 1.3B |
| Seed-VC (ADR 0012) | 2.29 s | 2.96 s | 0.93 s | - |
| **OmniVoice** | **2.50-2.55 s** | **2.92 s** | **1.18 s** | **6.6 GB** with NLLB 600M |

**A new reference** costs 17-24 s of CPU transcription and 3-5 s of encoding,
once; the transcript is kept beside the recording (`*.transcript.json`,
git-ignored, deleted with it). The pipeline's warm-up waits up to 60 s for it
(`WARM_TIMEOUT_S`) so the first real sentence does not queue behind it.

## Decision

**A third voice service, `scripts/omnivoice_server.py`, port 8767, same socket
protocol.** The client now sends the sentence's text and language with every
request; RVC and Seed-VC ignore them, OmniVoice speaks from them and ignores
the audio. Everything else - discovery, the pre-flight row, the Voices tab,
the consent check, the fallback to the generic voice - is unchanged.

It is zero-shot like Seed-VC, so a volunteer is a new reference, not a new
service: the volunteer demonstration no longer needs RVC stopped and Seed-VC
started. It needs NLLB 600M, like Seed-VC; the pre-flight list warns about
1.3B beside it.

Kokoro still synthesises each sentence before OmniVoice speaks it (0.15-0.2
s). That is deliberate for now: it is the generic voice ready when the service
fails or falls behind.

## Not done

- **Public voices of real people.** The hosted OmniVoice site lists "public
  voices" including Elon Musk, Donald Trump, MrBeast and SpongeBob, usable
  with two free trial credits. None is used here: a clone of a real person
  without their consent is the harm this talk is about. The site's list is
  evidence for the talk, not material for it.
- **Voice design** was left out at first; it is now the Studio's "Invent a
  voice" (below).

## The Studio (added the same day)

The presenter asked for an interface to OmniVoice on the operator page. The
**Studio** tab (`ui/studio.py`) speaks a typed sentence in a chosen language
through the service, keeps it as a clip, plays it in the browser or through
the speakers - through the pipeline's own playback when it is running, so the
gate keeps the microphone shut and the system does not translate its own
clip. Measured through the page: 5.0 s of speech in 1.1 s once the voice is
prepared.

It offers exactly two kinds of voice: a recording with a consent record (the
presenter's, a volunteer's), and an **invented** one - gender, age, pitch,
whisper, the attributes OmniVoice's voice design accepts, chosen from lists so
nothing else can be asked for. An invented voice is saved as a reference with
a consent record saying it belongs to no one, so the same character can speak
again, here or live. Three were made as samples (`nonna-inventata`,
`nonno-narratore`, `bambino`); Whisper read the first back word for word.

A volunteer's clips are written inside the volunteer's folder, so deleting
the volunteer deletes what was made from their voice. Building this found that
`data/voices/volunteers/` - names, consent records, recordings - was not in
`.gitignore`; it is now, with `data/voices/designed/`.

Each render differs. One clip in the presenter's voice mangled "ho avuto un
incidente"; the next render of a sentence may not. Render before the moment,
listen in the browser, then play.

**Everything OmniVoice has, the same evening** (the presenter asked for all of
it, with sliders). One table, `tts/omnivoice_options.py`, drives both the
page's sliders and the service's checks - steps, guidance, speed, fixed
length, variation (class temperature), order temperature, layer penalty, time
shift, padding, fade; denoise and output tidying; values clamped, unknown keys
dropped. Also: a **seed** (the service reports the one it used; the same seed
gave a byte-identical clip, so "Reuse" repeats a good render), the model's
**own voice** ("auto", asked for by name - a missing reference never quietly
becomes a stranger), all **646 languages** from the service, the 13
**non-verbal tags** (`[laughter]`, `[sigh]`, ...) inserted at the cursor,
**one clip per line**, **accents** in invented voices (English accents, so the
invented voice's sample is read in English), **upload** of any recording the
browser can decode - an iPhone voice memo included - as a consented volunteer,
**recording** through the Voices tab's path, and the reference's
**transcript**, shown, prepared on demand and correctable (the service now
rebuilds a voice when its transcript changes, not only its audio). Each was
exercised through the page against the real model.

**Against Qwen3-TTS, head to head** (same reference `phone-sentences`, eight
short talk sentences, numbers in words, several seeds): Qwen3-TTS 0.6B WER
4.3% with no render turned into another sentence (0/16), at 16 s a sentence;
OmniVoice 27-28% with 9 of 40 renders another sentence ("Sceglietela stasera"
came out "Grazie a tutti" four times over), at 0.85 s. From `ref-12s` instead,
OmniVoice improved to 13.1% and 4 of 40. The 1.7% end to end earlier was on
long sentences; short ones are its weakness. Qwen is the better voice for a
prepared clip; OmniVoice is the only one of the two fast enough to be live, and
its renders need hearing before an audience does.

**A recording is changed by changing the recording.** Asked to "invent from an
uploaded voice", the first thing measured was whether a design instruction
added to a clone does anything. It does not: the presenter's clone, same seed,
read 137.4 Hz plain, 137.8 with "very high pitch", 137.8 with "very low
pitch", 133.9 as "female, elderly" - and one of those renders turned "Nonna"
into "No, non sono io", a negation the sentence never had. What does work is
moving the reference itself: shifted -6, +4 and +8 semitones with librosa, it
gave clones at -5.1, +4.0 and +8.3, words intact to about ±6 and breaking at
+8 ("ho abuto inizidenza"). So the Studio's **derive** takes a consented
recording (or an upload), moves pitch (±8) and tempo (0.8-1.25), and saves it
as a new voice that carries the source's consent and lives with the
volunteers. **Invented voices can also be spoken from their description** in
the target language instead of cloned from their saved sample - the accent
attributes then apply to that language; both read back correctly in Italian,
and which sounds more accented is for the ear (`runs/voice-compare-20260926/
accent-and-derived/`). Volunteers', uploaded, derived and invented voices can
be **deleted** from the Studio with their clips; the presenter's own
recordings cannot. The advanced settings open by default: shipped closed,
they looked empty.

The service also answers a browser now. The first thing anyone does with a
port number is open it; the socket protocol read "GET " as a 1,195,725,856-byte
header and the log filled with "malformed request". It replies with a line
pointing to the operator page instead.

## Consequences

- Three services, one at a time on the GPU: RVC (fastest, presenter only),
  Seed-VC (zero-shot, converts), OmniVoice (zero-shot, speaks; slowest of the
  three, and the only one whose rhythm is not the synthesiser's).
- The presenter's reference for OmniVoice should be `phone-sentences.wav`, not
  `phone-12s.wav`.
- Runs in its own venv (`C:\Users\you\venvs\omnivoice`), built on the main
  environment's packages with only `omnivoice` and `accelerate` added: the
  main environment is untouched.
