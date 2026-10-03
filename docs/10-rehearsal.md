# Rehearsal — what to run, in what order, and what each step should show

This is the test script for a person. Everything the machine can verify on
its own, it already has — 767 tests, and hardware checks that need no one in
the room. What follows is the part that needs you: a voice into the
microphone, and your judgement on what comes out.

Do the steps in order. Each one says what a pass looks like and what to tell
me if it does not. Budget about forty minutes the first time.

The one rule: **when something is wrong, do not fix it by feel.** Note what the
page showed — the Setup list, the Live tab's cards, the number — and stop.
Every card on that page exists because a guess was wrong once.

---

## Quick start

From `C:\Users\you\Desktop\translation_local`, two terminals:

```bash
C:\Users\you\Applio\env\python.exe scripts\rvc_server.py --model presenter --port 8766 --f0-method fcpe --index-rate 0 --pitch -8
```

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\live_translate.py --ui
```

Open `http://127.0.0.1:8770/`. Setup: Spoken Turkish, Heard as Italian,
translator NLLB 1.3B. **Start**, wait ~35 s for `running · listening`, speak.
`http://127.0.0.1:8770/screen` is the audience's screen. To finish: **Stop** on
the page, then Ctrl+C in both terminals. The application uses whichever voice
service is running, so the order of the two terminals does not matter.

**OmniVoice instead of RVC** ([ADR 0015](adr/0015-omnivoice.md)) - it speaks the
sentence in the voice instead of repainting Kokoro's, and a volunteer is only a
new reference, not a new service. Slower (~2.5 s lag against ~1.7). Terminal 1:

```bash
C:\Users\you\venvs\omnivoice\Scripts\python.exe scripts\omnivoice_server.py
```

Then Setup: translator **NLLB 600M**; Voices: `phone-sentences`. The first
sentence in a new voice takes 20-30 s to prepare.

**The Studio tab** speaks typed sentences through OmniVoice - the grandchild's
call in a volunteer's voice, without the microphone. Render it before the
moment, listen in the browser (each render differs), then press Play: with the
pipeline running, Play goes through its playback and the microphone stays shut;
the browser's own player does not. It also invents voices that belong to no
one (three samples are there: `nonna-inventata`, `nonno-narratore`, `bambino`).
Port 8767 is not a web page; the Studio is on 8770.

## The Turkish script

Slide by slide, in Turkish: what the screen says, what to say (one short sentence per
line - each line is a pause for the translator), what to press, and a minute plan with
a rehearsal clock: the private page "parlIAmo Prova Metni". Interaction is by show of hands:
the translation runs Turkish -> Italian the whole talk, so a spoken Italian answer is
neither translated nor understood. Speech only with the volunteer and in the questions.

## A text to read while testing

Read it at speaking pace, pausing after each sentence unless it says not to.
What the Italian should mean is in brackets.

1. Merhaba, bugün sizinle yapay zekâ hakkında konuşmak istiyorum.
   *(Salve, oggi vorrei parlarvi di intelligenza artificiale.)*
2. Bu bilgisayar internete bağlı değil. *(... non è connesso a internet -
   the negation must survive.)*
3. Bir insanın sesi sadece üç saniyede kopyalanabiliyor. *(... può essere
   copiata in soli tre secondi.)*
4. Torununuz sizi arayıp para isterse ne yaparsınız? *(a question, with the
   question mark.)*
5. Önce telefonu kapatın, sonra bildiğiniz numarayı geri arayın.
   *(Prima riattaccate, poi richiamate il numero che conoscete.)*
6. **Without pausing:** Valdastra'da Acme'de çalışmış insanlarla konuşuyorum.
   Onlar yıllarca makine üretti. Bugün makinelerin nasıl konuştuğunu
   göreceğiz. *(all three spoken, none lost.)*
7. Hong Kong'da bir şirket yirmi beş milyon dolar kaybetti. *(the number.)*
8. For Heard as Friulian: Friuli'de yaklaşık altı yüz bin kişi Friulice
   konuşuyor.

For Spoken Italian, Heard as Turkish (questions from the room): *Come faccio a
capire se una telefonata è vera?*

---

## 0. Before anything: the room

Do this once per room, every room. It is not a software setting; it is a
property of the speakers, the microphone and the air between them.

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\measure_audio_device.py --write-config --label venue
```

Plays five chirps, hears them back, writes the median into
`config/local.yaml`. **Pass:** the last line says `config updated:
pipeline.output_latency_ms = NNN ms` and the line after it does *not* say
`WARNING`. A note saying the margin is thin is acceptable; a warning that the
gate is shorter than the slowest chirp is not — raise `half_duplex_tail_ms`
as it tells you.

**If it says the microphone heard nothing:** the output is on headphones, or
the volume is down. Fix that first; nothing below works without it.

That measures when the sound *arrives*. What the gate also has to outlast is
how long the last word keeps *sounding* - the PA, then the hall's echo. This
measures that, with a real Italian sentence through the pipeline's own
playback and gate:

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\measure_room_echo.py --input "MIC@Windows WASAPI" --output "SPEAKERS@Windows WASAPI" --write-config
```

- `MIC` and `SPEAKERS`: a piece of the names the Devices picker shows, e.g.
  `--input "Realtek@Windows WASAPI" --output "Dante@Windows WASAPI"`.
- **Stop** the pipeline on the page first; the page may stay open.
- Speakers at the **volume of the talk**, the microphone **where it will be
  during the talk** (at your mouth). Say nothing while it plays - three
  sentences, about 20 s.

**Pass:** it ends with `covered: the echo stops NNN ms before the gate
reopens`, or `the microphone does not hear the speakers as speech at all` -
the second is what a microphone at the mouth should give. Measured in the hall
on 2026-10-02 with the headset microphone and the Dante speakers: round trip
306 ms (the old 38 left the gate open before the last word even arrived),
echo covered by 512 ms or more with the resulting 556 ms gate. **If it says `the
echo outlives the gate` or `covered by only`:** with `--write-config` it has
raised `half_duplex_tail_ms` in `config/local.yaml` for you; press Start again
and it is used. Each sentence's line also says how loud the speakers reached
the microphone. An empty hall echoes more than a full one, so a pass here
holds on the day.

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\verify_feedback_loop.py
```

Plays recorded speech through the speakers twice, gate off then gate on.
**Pass:** three `->` lines and no `XX`. The gate-off line should report a
loop well above the room floor (36 dB on the reference laptop); the gate-on
line should say the pipeline received nothing. **If the first pass reports a
warning that the leak was too quiet to transcribe:** turn the volume up to
what the room will actually hear and run it again — a test at whisper volume
proves nothing about a talk at speaking volume.

---

## 1. Start the voice service

Its own terminal, left open. Pick one:

```bash
C:\Users\you\Applio\env\python.exe scripts\rvc_server.py --model presenter --port 8766 --f0-method fcpe --index-rate 0 --pitch -8
```

Your voice, trained. **Pass:** `loaded presenter_...pth` then `listening on
127.0.0.1:8766`. The application finds whichever service is running, RVC or
Seed-VC; the pitch is set per language on its own (`--pitch` is only a
fallback).

Or, for the volunteer demonstration:

```bash
C:\Users\you\miniconda3\envs\seedvc\python.exe scripts\voice_conversion_server.py
```

**Pass:** `listening on 127.0.0.1:8765`, after ~20 s of loading. To switch
during the talk: Ctrl+C the RVC terminal, start this one, then Stop → Start on
the page - Start finds Seed-VC by itself. Switch the translator back to NLLB
600M first; 1.3B does not fit beside Seed-VC.

**Not both at once during the talk** unless the GPU is otherwise idle; each
holds GPU memory — Seed-VC ~2.9 GB, RVC ~0.7 GB (1.2 GB at the peak of a
conversion, measured). For the rehearsal, one is enough.

---

## 2. Start the application

Second terminal:

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\live_translate.py --ui
```

**Pass:** `parlIAmo is at http://127.0.0.1:8770/`.

**If it says `parlIAmo is already running at …`** and exits: another copy is
open in some terminal. Either use the address it printed, or close that
terminal (Ctrl+C) and run this again. It refuses on purpose — two copies fight
over the microphone and the GPU, and the page you are looking at may belong to
either of them — the most likely reason every card was empty in the last
rehearsal.

If it says 8771, something that is *not* parlIAmo holds 8770 — use the address
it printed.

Open it. **Pass:** the header pulse is green (the page is connected), the
Setup tab has an amber dot only if something needs attention.

---

## 3. Setup, top to bottom

Click **Setup**. Read every row of the pre-flight list; it is written for this
moment.

| Row | Pass | If not |
|---|---|---|
| Output latency | green *measured* | step 0 was skipped |
| Voice conversion | green, and it names the service and model you started | the service is not up, or the config points at the other port |
| Microphone | green, WASAPI, a few ms, and the one you picked | it picked an MME clone at 90 ms — choose the WASAPI one in the picker below; "is not here" means the one you picked is unplugged |
| Speakers | green, WASAPI | yellow **"Speakers on MME"** (or DirectSound): it stutters while the pipeline works — on a silent test stream, MME had 45 gaps in 8 s under load and WASAPI none. Choose the WASAPI entry of the same speakers; the picker lists WASAPI first and marks the others "may stutter while translating" |
| Volunteer recordings on disk | none, or ones you mean to keep | `Delete all volunteer recordings` on the Voices tab |

Then the pickers, each applies at the next Start:

- **Languages:** Spoken `Turkish`, Heard as `Italian`. The note under it should
  read `applies at the next start`.
- **Responsiveness:** leave `500 ms` for the first run. You can try `300`
  later — it is what the earlier prototype felt like, and it splits sentences
  at breaths on your speech; the note says so.
- **Models:** with **RVC**, choose translator **NLLB 1.3B** — it fixed three
  meaning errors in eleven sentences for 0.2 s each and fits beside RVC
  (ADR 0013). With **OmniVoice** or **Seed-VC**, keep NLLB 600M; the picker
  and pre-flight both say 1.3B does not fit beside them. Leave the recogniser
  as it is.
- **Devices:** the microphone you will use on stage, and the speakers - both
  on **WASAPI**. Never a WDM-KS entry (none opens on this laptop; they are
  greyed out) and not MME for the speakers (it stutters under load, see the
  table). The banner under them shows *selected* and *live* separately; they
  will differ until Start, and a ⚠ means the one you picked did not open.
- **Microphone level:** speak a sentence and look at Windows' input meter
  (Settings → System → Sound → your microphone → Input volume). At rehearsal
  on 2026-10-02 the speech arrived at about **-40 dBFS** - 10-25 dB too quiet;
  Whisper then made little of it and filled the gaps with the hotword list.
  (Those recitals are now dropped instead of spoken, but the sentences they
  replaced are still lost - level is the fix.)
  Aim for loud speech near the top third of the meter: raise the input volume,
  enable Microphone Boost if the driver has it, keep the microphone 15-20 cm
  from your mouth - or use the room's own microphone through the Dante input.
- **After a run:** search the log for `ran dry`. Each line counts gaps the
  room heard; none means the speakers were fed in time.
- **Cloned voice speed:** 0.85 is the default (5.3 syllables/s, measured);
  0.8 is slower still.

---

## 4. Voices

Click **Voices**. **Pass:** your recording (`phone-12s`) shows a
`configured` tag and a consent string, and pressing **use** turns the button
to *in use* with no error.

If you are on RVC this tab is inert — the service speaks in its trained voice
whatever is selected, and the pre-flight row said so. If you are on Seed-VC,
*use* is what makes the next sentences yours.

---

## 5. Start, and the first sentence

Header, **Start**. **Pass:** `Starting → loading models → warming up →
running · listening` in about 35 s. If it stops at *Could not start*, the
reason is printed under it — send me that line.

Now say one short Turkish sentence, and stop.

Watch the **Live** tab:

1. Within about two seconds a grey italic line appears — the provisional
   subtitle — then is replaced by the committed sentence in bold. **Pass:**
   the Italian means what you said.
2. The **Microphone** card: while you speak, green *hearing speech*; between
   sentences, *room noise only* or *hearing nothing*. **If it stays amber
   while you talk, nothing else can work** — wrong device, or the level is
   down. Note the *level now* number and stop. **If it reads *hearing
   nothing* (0.000) with the right device chosen:** pull the headset plug out
   and push it fully back in. On 2026-10-02 in the hall the headset
   microphone - plugged in, WASAPI chosen, Windows showing it active and
   unmuted - delivered pure digital zeros until it was re-seated.
3. The **voice** stat card under the subtitles: after the sentence is spoken,
   a small green line — `✓ cloned · conversion 0.4 s`. **If it shows amber
   `spoken GENERIC`**, the reason is on the same line: service down, or
   conversion failed. This is the single most useful card on the page; it
   says whether what the room heard was your voice.
4. The **Last sentence** card: five colours. On RVC the last band — voice
   conversion — should be the shortest or near it. On Seed-VC it is the
   longest. Either is correct; a *missing* band means that stage did not
   run.

Then listen. The Italian should be in a voice you recognise as yours. If it is
not, and the voice card says `✓ cloned`, that is a quality question, not a
plumbing one — tell me which of the three samples I sent sounded most like
you, and whether this sounded like that one.

---

## 6. Three sentences without pausing

Say three sentences in a row, breathing but not stopping. This is the case
that broke on 2026-09-15.

**Pass:** all three appear and are spoken; the **Half-duplex gate** card's
*speech lost to the gate* stays at 0 or near it; the voice card does not show
amber `delivery behind, conversion skipped` — that means sentences arrived
faster than conversion could take them, and one went out generic to keep up.

**If sentences go missing:** look at the gate card first. *Speech lost to the
gate* counts words you said while the system was still speaking — the
microphone is deaf then, by design, so the speakers cannot feed back. On a
stage with a PA and a lavalier this is the number that decides whether the
gate can be relaxed; note it.

Then check `runs/<latest>/dropped/`. Anything the recogniser heard but made
nothing of is saved there as a WAV with a JSON saying why. Listen to it. It
is either a cough, or a sentence — and if it is a sentence, send me the file.

---

## 7. The live switch

Header, the button that says **Wait for pause** — or press **L**. It cycles
three ways; the label always says what the pipeline is doing *now*.

Repeat step 6 in each:

- **Live: sentence ends** — the first sentence should be spoken before you
  finish the third. Watch for the same sentence spoken twice; one duplication
  in eleven is the measured rate.
- **Live: chunks (buffer)** — words go out in fours as they settle. This is
  the earliest audio and the worst Italian, on purpose: the verb is the last
  word in Turkish, so the first chunk has none. Listen to what the room would
  hear and decide whether any part of the talk can afford it.

Set it back to **Wait for pause** before the next step.

---

## 7b. Other languages

Setup → Languages, then Stop → Start. **Heard as Spanish** or **German**: the
same voice, yours, in that language - the pitch is set per language on its own.
German is about 1.5 s slower a sentence (its voice runs on the CPU). For
questions: **Spoken Italian, Heard as Turkish** - you hear the question in
Turkish on headphones. Set it back before the next part of the talk.

## 8. Mute, flush, generic

While a long sentence is being spoken:

- **M** — the red *AUDIO MUTED* banner appears, audio stops, subtitles
  continue. **M** again clears it.
- **P** — the red *LISTENING PAUSED* banner appears; the microphone meter
  still moves but nothing is transcribed. What you said just before pressing
  it is still translated. **P** again listens. Use it for **every video**
  (press before playing it, again after), for applause and for questions from
  the floor: the gate covers only the system's own voice, so a video through
  the room's speakers would otherwise be subtitled and spoken over. M would
  not help - it stops the voice, not the subtitles.
- **Flush playback** — the sentence being spoken stops mid-word; the next one
  plays normally. This is the panic button; know where it is.
- **G** — the next sentence is spoken in the generic Kokoro voice, and the
  voice chip in the header reads `voice: generic`. On the Voices tab, *use*
  brings yours back.

---

## 9. Friulian

Setup → Languages → Heard as **Friulian** → Stop → Start. Say a sentence.
**Pass:** Friulian text on screen (`al è`, `cheste`, `lenghe`) spoken with
Italian phonetics in your voice.

Then the **Friulian** tab: press a numbered button under *Marco Moroldo —
real Friulian*, then a ▶ on one of the pipeline's rows. That contrast is the
segment. **Stop the pipeline before playing these** — the tab warns you —
or the microphone will hear the clip and translate it.

---

## 9b. Optional: could the microphone stay open?

Only if there is time, at performance volume, in the room:

```bash
C:\Users\you\miniconda3\envs\parliamo\python.exe scripts\measure_echo_cancellation.py
```

Plays 25 s of Italian through the speakers. It answers whether echo
cancellation could replace the half-duplex gate, so that words said while the
translation plays are not thrown away. **Look at two lines:** *with AEC: 0
segment(s)* (nothing the pipeline would translate back) and the WER *as
recorded* with AEC (how much of your speech survives when you talk over the
translation - 15% on the laptop at low volume, against 5% without). Send me
the folder it names. Nothing in the running system changes because of it yet.

## 10. The audience screen

Open `http://127.0.0.1:8770/screen` in a second window and drag it to the
projector. **Pass:** the same sentences, huge, on black; the newest largest.
**F** hides the footer for the room. Its **MUTE** button is the same mute.

Reload it mid-talk. **Pass:** the last four sentences come back at once — a
blank screen on stage looks exactly like the system having died, and this is
what prevents it.

---

## The talk itself

The talk is **parlIAmo**, for a seniors' association in Friuli. The slides - Italian on screen,
Turkish speaker notes on every slide saying what to say and what to press - are
the private deck "parlIAmo · La voce non basta", 31 slides in the flyer's
colours with the group's logo. Download it as PDF or PPTX before the day: the
venue's internet is not to be trusted, and the checklist slide is meant to be
printed and handed out.

Its two live demos are this system: the translation (steps 5-7) and Friulian
for a Friulian audience (step 9 - the pipeline's Friulian beside Marco
Moroldo's real voice). After the first rehearsal (2026-09-27) the presenter
took out the volunteer demo and the "which is my real voice?" slide, and a
seventh case went in first: Treviso, 84-year-old woman, the daughter's voice
cloned, 30,000 euros - 50 km from the venue. The cases carry videos where a
reputable one exists (Tgcom24, Fanpage.it, The Telegraph's recording of the
fake Biden call, SCMP, Geopop; for video calls, Jim Browning testing a real
scammer's face filter). None for Brad Pitt - TF1 pulled its interview after
the victim was mocked - or Ferrari. Download the videos beforehand; with all
of them the talk runs past the hour. **Press P before each video and again
after it** (step 8).

## What to send me

For anything that did not pass: the step number, what the page showed, and
the run folder — `runs/<timestamp>/` has `events.jsonl` and `dropped/`.
A screenshot of the Live tab at the moment it went wrong is worth more than a
description of it.

For everything that did: which voice sample was you, and how the three
streaming positions felt in a real sentence. Those two are the decisions
left that a measurement cannot make.
