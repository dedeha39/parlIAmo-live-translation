# 12. RVC beside Seed-VC: clone once for the presenter, zero-shot for volunteers

Date: 2026-09-15

Status: accepted

## Context

The presenter measured 6 s of lag on stage with the voice service up, closed
the service, and asked why the earlier prototype had been faster. Three
questions were folded into that, and each has a measured answer.

**Where the time goes.** Same 68 s recording, lag from the moment the speaker
stops (commit wait included):

| configuration | mean | p95 | asr | mt | tts | vc |
|---|---|---|---|---|---|---|
| subtitles only | 1.10 s | 1.33 | 0.37 | 0.23 | — | — |
| generic voice (Kokoro) | 1.29 s | 1.55 | 0.37 | 0.23 | 0.19 | — |
| cloned, Seed-VC | 2.29 s | 2.96 | 0.39 | 0.25 | 0.22 | **0.93** |

Kokoro costs 0.19 s. Voice conversion costs 0.93 s and is 41% of the total.
The six seconds on stage were not in this table: the run log showed three
sentences dropped as "delivery behind" in eight seconds — streaming mode had
produced sentences faster than a serial ~1 s conversion could take them, and
the number on the screen was the queue.

**Why the earlier prototype felt faster.** It used Whisper `medium`,
argostranslate, and `edge-tts` — Microsoft's cloud synthesiser — with a 0.3 s
commit wait. Reproduced as closely as the offline constraint allows (medium,
wait 0.3, generic voice): 1.19 s mean. This pipeline in generic voice at the
same wait: **1.04 s**. Its recogniser was slower than ours (0.56 s against
0.36). It felt faster because it cloned nothing, waited less, and streamed
cloud audio before synthesis finished. The cloud is not available in the
venue, and the claim of the talk is that none of this needs it.

**"Isn't cloning a one-time thing?"** With Seed-VC it is not. Seed-VC is
zero-shot: the reference recording is conditioning, applied at every
sentence. That is what makes a fifteen-second volunteer possible on stage and
what costs 0.93 s per sentence forever. The presenter's own voice does not
need that property — it is the same voice every night.

## Decision

**RVC for the presenter's voice, Seed-VC for volunteers. Both, by port.**

RVC (via Applio, installed since the earlier prototype) trains a voice once —
here from 7.3 minutes of the presenter's own recordings, 300 epochs, 58
minutes on this GPU — and applies it at a fraction of Seed-VC's cost.
`scripts/rvc_server.py` serves the trained model over the *same* socket
protocol as `voice_conversion_server.py`, so the pipeline's client does not
know which is behind the port. `tts.conversion.port` chooses: 8765 Seed-VC,
8766 RVC. The reference path the client sends is ignored by RVC and echoed
back, so a mismatch is visible, and the pre-flight list names which service
answered.

Measured end to end, same recording, zero conversion failures:

> **Corrected 2026-09-23.** Until then every figure in the tables in this ADR was 0.50 s
> too high: the script that summarised the run reports added the commit wait
> to `total_lag_s`, which already contains it. The stage columns were right;
> only the totals were wrong. Recomputed from the same report files
> (`runs/live/translate-20260915-*.json`).


| | mean | p95 | vc |
|---|---|---|---|
| Seed-VC | 2.29 s | 2.96 s | 0.93 s |
| **RVC** | **1.58 s** | **1.80 s** | **0.43 s** |

0.71 s off the mean, 1.16 s off the tail.

> **Found 2026-09-26:** until that day RVC played through the speakers at half
> speed an octave down - its 48 kHz audio went to a 24 kHz stream without its
> rate. The latency figures here are unaffected (they end when playback
> starts); what the presenter *heard* was. Fixed in `AudioPlayback.submit()`.
>
> **Since 2026-09-26** the port no longer has to be chosen: Start uses whichever
> service is running, the configured port first. And the pitch is no longer
> one `--pitch` for all: each sentence carries the shift for the voice that
> spoke it ([ADR 0014](0014-spanish-german-turkish-and-calabrese.md)).

Two things RVC needed that Seed-VC did not, both found by measuring pitch:

- **RVC keeps the source pitch.** From Kokoro's `if_sara` (216 Hz) it produced
  231 Hz — the presenter's timbre at a woman's pitch; the presenter measures
  139 Hz. Seed-VC had moved it to 153 Hz by itself. RVC takes a semitone shift
  instead: `--pitch -8` from `if_sara` lands at 135 Hz, `--pitch +9` from the
  male `im_nicola` (83 Hz) at 159 Hz. The server takes the shift as a flag
  because the source voice is fixed per run.
- **`fcpe` and no index search.** With `rmvpe` and `index_rate 0.5` RVC took
  1.06 s — no faster than Seed-VC. `fcpe` and `index_rate 0` bring it to
  0.43 s. RVC's real-time reputation is for small streamed windows; a whole
  sentence through the full pitch extractor is not that.

## Also found

- `tts.voice` was silently ignored. The voicepack lookup tried the language
  default first and the configured voice only for a language *without* one —
  which is no language. Found when a male source was wanted and `im_nicola`
  measured 216 Hz. Fixed; a configured voice from another language falls
  back with a warning, because a voicepack must match the frontend.
- `im_nicola.pt` downloaded into `~/.cache`, not the project's `models/`
  cache that the venue runs from. Copied. Anything fetched on demand is a
  venue failure waiting; the pre-flight list should check voicepacks too.
- The pipeline's sanitiser was stripping "La traduzione" from the front of
  real sentences (ADR-worthy on its own; fixed in the same batch).

## Consequences

- The cloned-voice path is 0.71 s faster with the presenter's own voice and
  no longer the largest term in the budget.
- Two services to start, not one, when both are wanted. The volunteer
  demonstration still needs Seed-VC on 8765; the talk needs RVC on 8766. A
  run points at one port at a time.
- RVC quality is bounded by 7.3 minutes of mixed recordings, some through the
  laptop microphone that gates 16–50% of what it captures. Ten minutes read
  into a phone in a quiet room would retrain in an hour and almost certainly
  sound better. The trained weights are not in git.
- The streaming modes and a ~0.4 s conversion no longer overrun the delivery
  queue on the reference recording; and when they do, a sentence goes out
  generic rather than being dropped.

## Still open

The choice of source voice — `if_sara −8` or `im_nicola +9` — is the
presenter's ear, not a measurement. Both samples were delivered.
