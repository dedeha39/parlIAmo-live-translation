# 11. Opening audio devices: COM, fallback, and the -9999 that was not a device fault

Date: 2026-09-02

Status: accepted

## Context

Starting the pipeline from the application failed with:

```
RuntimeError: could not open any audio output after 3 attempts. Last error:
Error starting stream: Unanticipated host error [PaErrorCode -9999]:
'GetNameFromCategory: usbTerminalGUID = 7D1E ' [Windows WDM-KS error -9999]
```

The message points at the audio hardware, and for some time so did we. It was
wrong in three separate ways, and each one hid the next.

### The retry could not succeed

The retry loop re-resolved the *same* device specification on every attempt, so
three attempts produced three identical failures. A retry only means something
if the next attempt differs from the last.

### WDM-KS does not work on this machine at all

Measured across the whole device table:

| Host API | outputs | open | fail |
|---|---|---|---|
| WASAPI | 3 | 3 | 0 |
| DirectSound | 4 | 4 | 0 |
| MME | 4 | 4 | 0 |
| **WDM-KS** | **12** | **0** | **12** |

Not intermittent - a broken driver path. Any fallback that stays inside it is
a guaranteed second failure, which is why WDM-KS is ranked last despite having
the lowest latency on paper.

### The real cause: WASAPI is COM, and COM is per-thread

The application builds the pipeline on a worker thread, so the page can show
progress instead of freezing for forty seconds while models load. A fresh
thread has no COM apartment, and WASAPI is COM. Measured, all three WASAPI
outputs, repeatedly:

```
main thread     -> OPENS  OPENS  OPENS
worker thread   -> FAILS  FAILS  FAILS
worker + COM    -> OPENS  OPENS  OPENS
```

This is why it read as a hardware fault for so long: it depended on *where* the
code ran, not on the device. Bluetooth and torch were both suspected and both
ruled out by running the same probe before and after loading them - the failure
is identical either way.

The reported text is also a red herring. PortAudio's last-host-error is global,
so a WASAPI failure is reported with whatever the previous WDM-KS attempt left
behind:

```
Headphones (...) [Windows WASAPI]: Unanticipated host error [PaErrorCode -9999]:
'Failed to create render pin: sr=48000,ch=2,bits=8,align=2' [Windows WDM-KS error]
```

The device named in that line is right; the explanation after it belongs to a
different device on a different host API. Our logs therefore record which
device was being opened rather than trusting the message to say.

## Decision

**Initialise COM on any thread that opens a stream.** `ensure_com()` calls
`CoInitializeEx(NULL, COINIT_MULTITHREADED)`, tolerates `RPC_E_CHANGED_MODE`
(the thread already has an apartment - fine), and is a no-op off Windows. MTA
rather than STA because an apartment-threaded thread owes Windows a message
pump and an audio worker has nothing to pump it with.

The apartment is never released. It was verified that a stream keeps running
both after its opening thread exits and after `CoUninitialize` - 48000 callback
frames in the second following each, at 48 kHz - because PortAudio holds its
own reference. There is nothing to gain by testing that on stage.

**Walk a candidate list, each device once.** In order: what was asked for, then
the same hardware on another host API cheapest-first, then anything else that
works. The last step matters because a device can vanish between setup and the
button being pressed.

**The first choice gets a second chance, before any fallback is opened.** The
remaining failures are transient - a Bluetooth headset mid-connection takes the
WASAPI endpoints down for a moment, then they open normally. Without this the
walk settles on DirectSound's "Primary Sound Driver" and stays there for the
whole hour, on a device the operator did not choose. Retrying *after* a
fallback opened would mean holding two streams and deciding which to close, on
the audio path, on stage; doing it before costs 0.35 s and owns nothing.

**Never fall back onto a loopback input.** "Stereo Mix" and "PC Speaker" are
present on this laptop and capture what the sound card is playing. Falling back
onto one would have the pipeline transcribe its own Italian, translate that,
and speak it again - and the half-duplex gate cannot break that loop, because
it is inside the card rather than across the room. An operator who names one
explicitly still gets it; only automatic selection refuses.

**Show the substitution.** A device that will not open is replaced by one that
will. That is right thirty seconds before a talk and wrong to do quietly, so
`GET /devices` now reports `live` and `substituted` alongside the lists, and
Setup marks a replaced device. Previously that state was reachable only from
`POST`, so the page could not show it unless the operator changed something.

## Consequences

- The pipeline starts on hardware where it previously refused to.
- It runs on WASAPI from the worker thread, rather than silently spending the
  talk on DirectSound. That is the difference the COM fix actually buys.
- A dead preferred device costs one extra failed open plus 0.35 s, once.
- `sd._terminate()` must never be used to force re-enumeration: it invalidates
  every open stream, so opening the speakers silently kills the microphone.
  `test_capture_discards_audio_while_gate_is_closed` caught that within a
  minute of it being written. `verify_index` re-enumerates without it.
- Device names are stripped of control characters at the source. This laptop
  reports a Bluetooth endpoint with a literal newline inside its name, which
  split log lines in half and made the Setup list unreadable.

## Still open

`pipeline.output_latency_ms` is still 0 and has to be measured in the venue.
None of the above changes that: the gate is sized from PortAudio's reported
latency, which excludes vendor DSP (~170 ms of Realtek/Nahimic effects on this
laptop).
