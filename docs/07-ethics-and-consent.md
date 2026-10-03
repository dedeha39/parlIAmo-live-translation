# Ethics and consent

This project clones voices. That capability is the **subject** of the
presentation, not a side effect of it: the argument is that voice cloning has
become accessible enough to run on a laptop, and the demonstration is the
evidence.

A talk making that argument has to hold itself to the standard it is asking for.

---

## 1. The rules, and where they are enforced

| Rule | Enforced by |
|---|---|
| No voice is cloned without a written consent record | `VoiceProfile.validate()` — raises, in code |
| Reference recordings never leave the machine | no network calls at run time |
| Reference recordings are deleted after the event | operator checklist, §5 |
| Cloning a public figure is out of scope | policy — see §3 |
| Synthesised output is disclosed as synthetic | the talk itself; watermarking where available |

The first one is not a guideline. `parliamo/tts/base.py`:

```python
def validate(self) -> None:
    if not self.consent.strip():
        raise ValueError(
            f"voice profile {self.name!r} has no consent record. "
            "Set consent to the signed authorisation reference before cloning."
        )
```

`scripts/live_translate.py` calls it **before anything is loaded or cloned**, so
a missing consent record is a refusal at startup, not a warning buried in a log:

```
  voice profile 'target' has no consent record.
```

A policy that lives only in a document is a policy that gets skipped at 23:00
the night before a talk. This one is in the call path.

---

## 2. Whose voice, and on what terms

### The presenter's own voice

No consent question. Recorded via `scripts/record_script.py`, 3–4 minutes of
reading a prepared script. This is the voice used for the live translation.

### A volunteer from the audience

This is the demonstration that lands: fifteen seconds of someone's voice,
cloned live. Seed-VC is zero-shot, so no per-voice training is needed — which is
precisely why it is alarming, and precisely why it needs handling.

Before recording anyone:

1. **Say what will happen**, in Italian, in plain language: their voice will be
   recorded, a model will imitate it, the imitation will be played to the room.
2. **Say what happens to the recording**: it stays on this laptop and is deleted
   at the end of the session.
3. **Get it in writing** — the form in §4, signed, before the microphone is
   opened.
4. **Offer them the chance to stop**, including after hearing the result.

A volunteer who has been asked in front of two hundred people is under social
pressure to say yes. Ask for volunteers before the segment, not during it, and
make refusing easy.

### Public figures — not done

Cloning a politician's or celebrity's voice is technically identical and takes
the same fifteen seconds. It is out of scope for this project regardless.

The reasons are practical as well as principled:

* A recording of a real public figure saying something they did not say is a
  fabricated record, and it does not stop being one because it was made to
  illustrate a point. It outlives the room it was played in.
* Italian and EU law on defamation and image rights does not have a "but it was
  a demonstration" exception.
* The point does not need it. *"This is your colleague's voice, recorded ninety
  seconds ago"* is more alarming to an audience than a celebrity impression,
  because the audience knows the colleague.

Voice models trained on public figures exist on this machine from an earlier
experiment (RVC/Applio). **They are not used in the talk.** If they are kept at
all, they stay off the presentation machine.

---

## 3. What the demonstration should show

The persuasive sequence is not "listen to this clone". It is:

1. **How little is needed.** Fifteen seconds, one laptop, no internet.
2. **How good it is.** Play the clone next to the real recording.
3. **How to tell.** This is the part that is usually missing, and the part the
   audience can act on.
4. **What protects you.** Verification habits that do not depend on the audio:
   call back on a known number, agree a family code word, treat urgency as a
   signal rather than a reason.

Point 3 is where watermarking belongs. **Chatterbox watermarks every output by
default** (Perth), which is why it is kept in the project after being removed
from the live path — being able to clone a voice and then detect the watermark
in the same breath is the demonstration. Kokoro and Seed-VC do **not**
watermark, and the talk should say so: watermarking is a property of some tools,
not of the technology.

Be honest about its limits. A watermark survives ordinary re-encoding and does
not survive a determined attacker. It is evidence, not protection — and an
attacker who cares will simply use a tool that does not watermark, exactly as
the live path here does.

---

## 4. Consent form

Short enough to be read in the room, in Italian, on paper.

> **Consenso alla clonazione vocale — parlIAmo**
>
> Acconsento alla registrazione della mia voce e alla creazione di una copia
> sintetica ("clone vocale") durante questa presentazione.
>
> Comprendo che:
> - la registrazione e il clone restano **su questo computer** e non vengono
>   caricati su alcun servizio esterno;
> - il clone verrà riprodotto in sala a scopo dimostrativo;
> - registrazione e clone verranno **cancellati al termine della sessione**;
> - posso ritirare il consenso in qualsiasi momento, anche dopo aver ascoltato
>   il risultato, e in tal caso il materiale sarà cancellato immediatamente.
>
> Nome: ______________________  Firma: ______________________
>
> Data: ____________  Riferimento: __________
>
> *Titolare del trattamento: [nome], [contatto].*

The **Riferimento** is what goes into `--consent`, so the running system carries
a pointer to a signed piece of paper:

```bash
python scripts/live_translate.py --voice data/voices/volunteer.wav \
    --consent "signed 2026-09-14, ref 007"
```

---

## 5. Operator checklist

**Before**
- [ ] Consent forms printed, in Italian
- [ ] Volunteers asked in advance, not put on the spot
- [ ] `data/voices/` empty except for the presenter's own reference

**During**
- [ ] Form signed before the microphone opens
- [ ] Reference number entered into `--consent`
- [ ] Anyone who changes their mind is honoured immediately

**After**
- [ ] Delete every volunteer recording and every generated clone
- [ ] Empty `runs/` of any audio containing a volunteer's voice
- [ ] Confirm deletion to the volunteers, out loud

---

## 6. The law

**EU AI Act, Article 50** — transparency obligations for AI systems that
generate or manipulate audio.

| | |
|---|---|
| In force since | **2 August 2026** |
| Watermarking grace period ends | **2 December 2026** |

Two obligations bear on this talk:

* **Disclosure.** People interacting with an AI system, or exposed to
  AI-generated audio, must be told. In a presentation whose entire subject is
  that fact, this is satisfied by the talk existing — but the slides should say
  it explicitly, because "it was obvious from context" is not a defence.
* **Machine-readable marking.** Providers of generative systems must mark
  outputs as artificial. This binds providers rather than someone demonstrating
  on a laptop, but it is exactly the thing the watermark segment is about, and
  the grace period ending in December 2026 makes it timely rather than
  theoretical.

Separately, a voice is **biometric data** under the GDPR. That is the legal
reason the reference recordings never leave the machine, and it is a stronger
argument for local inference than latency ever was.

*This is a summary written by an engineer, not legal advice. If this system is
ever used outside a research presentation, get proper advice first.*

---

## 7. Why local is the ethical position, not just the practical one

The project's own README lists three reasons to stay local. Only one of them is
about ethics, and it is the one that matters most:

**A volunteer's reference recording is their biometric identity.** Sending it to
an API means it exists on someone else's disk, under someone else's retention
policy, subject to someone else's breach. Fifteen seconds of audio is now enough
to impersonate them.

"It runs on this laptop and nothing left the room" is a promise that can
actually be kept, and it can be demonstrated on stage by unplugging the network.
