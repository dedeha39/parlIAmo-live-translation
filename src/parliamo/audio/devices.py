"""Audio device discovery and selection.

The pipeline needs to name devices in config files that survive reboots, so a
raw PortAudio index is not enough - indices shift when a Bluetooth headset
connects. Devices can therefore be selected by:

* ``None``   - the system default for that direction
* ``int``    - a PortAudio index (fast, but brittle across reboots)
* ``str``    - a case-insensitive substring of the device name, optionally
               suffixed with ``@hostapi`` (e.g. ``"lavalier@WASAPI"``)

Bluetooth caveat that matters for this project
----------------------------------------------
Opening a Bluetooth headset's *microphone* on Windows forces the HFP/HSP
profile, which collapses both directions to mono 16 kHz (often 8 kHz) with
aggressive compression, and adds 100-250 ms of latency. ``AudioDevice`` flags
likely-Bluetooth devices so that tooling can warn instead of silently degrading
recognition accuracy. See docs/03-hardware-budget.md and scripts/measure_audio_device.py.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

Direction = Literal["input", "output"]

# Host APIs ordered best-to-worst for our purposes on Windows. WASAPI in shared
# mode is the sweet spot: low latency without exclusive-mode device grabbing.
#
# **WDM-KS is last, despite being the lowest-latency on paper.** It is the
# kernel-streaming path and it is the one that fails, repeatedly and without
# warning, on this machine::
#
#     Unanticipated host error [PaErrorCode -9999]
#     'GetNameFromCategory: usbTerminalGUID = 7D1E ' [Windows WDM-KS error]
#
# It appears while a Bluetooth device is connecting and goes away afterwards,
# which is exactly the moment someone walks on stage with AirPods in their
# pocket. DirectSound and MME are slower and boring, and boring is what a
# fallback is for: the whole point of falling back is to reach something that
# works, and a path that intermittently refuses to start is not that.
_HOSTAPI_PREFERENCE = ("Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS")

#: Output paths that starve while the pipeline works. The callback that feeds
#: the speakers is Python; with the interpreter busy recognising and
#: translating, a silent MME stream was fed 150 times in 8 s instead of 610 and
#: underflowed 45 times - the "choppy" translation at the 2026-10-02 rehearsal.
#: WASAPI, measured the same way, not once.
STUTTERING_OUTPUT_APIS = ("MME", "Windows DirectSound")


def stutters_under_load(device: AudioDevice) -> bool:
    """Would speakers opened through *device* starve while the pipeline runs?"""
    return device.hostapi_name in STUTTERING_OUTPUT_APIS

_BLUETOOTH_HINTS = (
    "bluetooth",
    "airpods",
    "hands-free",
    "handsfree",
    "headset",
    "hfp",
    "a2dp",
    "wireless",
    "buds",
    "wh-",
    "wf-",
)


class DeviceResolutionError(LookupError):
    """Raised when a device specification matches zero or many devices."""


@dataclass(frozen=True, slots=True)
class AudioDevice:
    index: int
    name: str
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    default_low_input_latency: float
    default_low_output_latency: float
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def supports_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def supports_output(self) -> bool:
        return self.max_output_channels > 0

    @property
    def likely_bluetooth(self) -> bool:
        """Heuristic: does this name look like a wireless headset?

        False positives are cheap (a warning), false negatives are expensive
        (a narrowband microphone nobody noticed until the WER came back bad).
        """
        lowered = self.name.lower()
        return any(hint in lowered for hint in _BLUETOOTH_HINTS)

    @property
    def label(self) -> str:
        return f"{self.name} [{self.hostapi_name}]"

    def latency_ms(self, direction: Direction) -> float:
        latency = (
            self.default_low_input_latency
            if direction == "input"
            else self.default_low_output_latency
        )
        return round(latency * 1000.0, 1)


def _sd() -> Any:
    """Import sounddevice lazily so the package imports without PortAudio."""
    import sounddevice

    return sounddevice


#: Windows driver names can contain real control characters. This laptop
#: reports a Bluetooth endpoint whose name has a literal newline inside it::
#:
#:     Output 1 (@System32\\drivers\\btha2dp.sys,#1;%1%0
#:     ;(AirPods Pro - Find My))
#:
#: which splits log lines in half and makes the Setup list unreadable. The name
#: is an identifier, so it is flattened rather than truncated - it still has to
#: match its siblings across host APIs.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")


def _clean_name(name: str) -> str:
    return _CONTROL_CHARS.sub(" ", name).strip()


def list_devices(direction: Direction | None = None) -> list[AudioDevice]:
    """Enumerate audio devices, optionally filtered by *direction*."""
    sd = _sd()
    hostapis = sd.query_hostapis()
    try:
        default_in, default_out = sd.default.device
    except Exception:  # pragma: no cover - no default device configured
        default_in, default_out = -1, -1

    devices: list[AudioDevice] = []
    for index, raw in enumerate(sd.query_devices()):
        device = AudioDevice(
            index=index,
            name=_clean_name(str(raw["name"])),
            hostapi_name=str(hostapis[raw["hostapi"]]["name"]),
            max_input_channels=int(raw["max_input_channels"]),
            max_output_channels=int(raw["max_output_channels"]),
            default_samplerate=float(raw["default_samplerate"]),
            default_low_input_latency=float(raw["default_low_input_latency"]),
            default_low_output_latency=float(raw["default_low_output_latency"]),
            is_default_input=(index == default_in),
            is_default_output=(index == default_out),
        )
        if direction == "input" and not device.supports_input:
            continue
        if direction == "output" and not device.supports_output:
            continue
        devices.append(device)
    return devices


def preferred_hostapi(available: list[str] | None = None) -> str | None:
    """Return the best available host API name, or None on non-Windows."""
    if not sys.platform.startswith("win"):
        return None
    if available is None:
        available = [a["name"] for a in _sd().query_hostapis()]
    for name in _HOSTAPI_PREFERENCE:
        if name in available:
            return name
    return available[0] if available else None


def _split_spec(spec: str) -> tuple[str, str | None]:
    """Split ``"name@hostapi"`` into its parts."""
    if "@" in spec:
        name, _, host = spec.rpartition("@")
        return name.strip(), host.strip()
    return spec.strip(), None


def _score(device: AudioDevice) -> int:
    """Lower is better: rank candidates by host API preference."""
    try:
        return _HOSTAPI_PREFERENCE.index(device.hostapi_name)
    except ValueError:
        return len(_HOSTAPI_PREFERENCE)


# MME truncates device names to 31 characters, so "Microphone Array (Intel(R)
# Smart Sound Technology...)" arrives as "Microphone Array (Intel(R) Smar".
# Sibling matching therefore compares on the shorter of the two names.
_MME_NAME_LIMIT = 31


def _same_hardware(a: AudioDevice, b: AudioDevice) -> bool:
    """Do two entries plausibly describe the same physical device?"""
    name_a = a.name.strip().casefold()
    name_b = b.name.strip().casefold()
    if name_a == name_b:
        return True
    shorter, longer = sorted((name_a, name_b), key=len)
    # Only accept prefix matches that look like MME truncation, otherwise
    # "Headphones" would swallow "Headphones 2 (Realtek ...)".
    return len(shorter) >= _MME_NAME_LIMIT - 2 and longer.startswith(shorter)


def _fastest_sibling(device: AudioDevice, candidates: list[AudioDevice]) -> AudioDevice:
    """Return the same hardware on the most preferred host API available."""
    siblings = [d for d in candidates if _same_hardware(d, device)]
    if not siblings:
        return device
    return sorted(siblings, key=_score)[0]


def resolve_device(
    spec: int | str | None,
    direction: Direction,
    devices: list[AudioDevice] | None = None,
) -> AudioDevice:
    """Resolve a device specification into a concrete :class:`AudioDevice`.

    Raises :class:`DeviceResolutionError` with an actionable message listing the
    candidates when the specification is ambiguous or unmatched.
    """
    candidates = devices if devices is not None else list_devices(direction)
    candidates = [
        d for d in candidates
        if (d.supports_input if direction == "input" else d.supports_output)
    ]
    if not candidates:
        raise DeviceResolutionError(f"no {direction} devices found on this system")

    # --- default -----------------------------------------------------------
    if spec is None:
        defaults = [
            d for d in candidates
            if (d.is_default_input if direction == "input" else d.is_default_output)
        ]
        if defaults:
            # Windows reports the *MME* clone as the system default, which
            # carries ~90 ms of buffering. The identical hardware is usually
            # also exposed on WASAPI at ~2 ms, so re-point the default at the
            # fastest host API offering the same device name before returning.
            return _fastest_sibling(defaults[0], candidates)
        return sorted(candidates, key=_score)[0]

    # --- explicit index ----------------------------------------------------
    if isinstance(spec, int):
        for device in candidates:
            if device.index == spec:
                return device
        raise DeviceResolutionError(
            f"{direction} device index {spec} not found. "
            f"Available: {', '.join(f'{d.index}={d.name}' for d in candidates)}"
        )

    # --- substring (optionally host-api qualified) -------------------------
    needle, hostapi = _split_spec(str(spec))
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    matches = [d for d in candidates if pattern.search(d.name)]
    if hostapi:
        host_pattern = re.compile(re.escape(hostapi), re.IGNORECASE)
        matches = [d for d in matches if host_pattern.search(d.hostapi_name)]

    if not matches:
        raise DeviceResolutionError(
            f"no {direction} device matching {spec!r}. "
            f"Available: {'; '.join(d.label for d in candidates)}"
        )
    if len(matches) == 1:
        return matches[0]

    # Multiple host APIs expose the same hardware; pick the best one rather
    # than making the user spell out "@Windows WASAPI" every time.
    ranked = sorted(matches, key=_score)
    if _score(ranked[0]) < _score(ranked[1]):
        return ranked[0]
    raise DeviceResolutionError(
        f"{spec!r} is ambiguous for {direction}: {'; '.join(d.label for d in ranked)}. "
        "Add '@<hostapi>' or use the numeric index."
    )


#: Inputs that carry the machine's own output back in. Falling back onto one of
#: these would have the pipeline transcribe its own Italian, translate that, and
#: speak it again - and the half-duplex gate cannot help, because the loop is
#: inside the sound card rather than across the room. Present on this laptop as
#: "Stereo Mix (Realtek HD Audio Stereo input)" and two "PC Speaker" entries.
#: An operator who names one explicitly still gets it; only automatic fallback
#: refuses.
_LOOPBACK_HINTS = ("stereo mix", "what u hear", "wave out", "loopback",
                   "pc speaker", "monitor of ")


def _is_loopback(device: AudioDevice) -> bool:
    name = device.name.casefold()
    return any(hint in name for hint in _LOOPBACK_HINTS)


def candidates(
    spec: int | str | None,
    direction: Direction,
    limit: int = 6,
    devices: list[AudioDevice] | None = None,
) -> list[AudioDevice]:
    """Devices to try opening, best first, each appearing once.

    The fix this is. Retrying a failed open used to re-resolve the *same*
    specification, so three attempts produced three identical failures::

        could not open any audio output after 3 attempts. Last error:
        Unanticipated host error [PaErrorCode -9999]
        'GetNameFromCategory: usbTerminalGUID = 7D1E ' [Windows WDM-KS error]

    A retry is only worth anything if the next attempt differs from the last.

    On the reference laptop every WDM-KS entry fails to open and every other
    entry succeeds - all twelve and all eleven respectively, measured. That is
    not intermittent, it is a broken driver path, so a fallback has to leave
    the host API rather than try the same one again. Order:

    1. what was actually asked for,
    2. the same hardware on another host API, cheapest latency first,
    3. the device Windows itself uses for this direction, on its fastest API,
    4. anything else that can make sound.

    Step 3 is from rehearsal. The jack microphone is "Realtek HD Audio Mic
    input" on WDM-KS and "Microphone (Realtek(R) Audio)" everywhere else - one
    device, two names, so step 2 cannot pair them. Chosen on WDM-KS, it
    refused to open, and step 4 settled on the laptop's own array: the
    microphone that gates 16-50% of speech to silence. Windows had the jack
    microphone as its default, one entry further on.

    Step 4 matters because a device can vanish between the talk being set up
    and the button being pressed - unplugged, or a headset that went to sleep -
    and any working speaker beats silence in front of an audience.
    """
    available = list_devices(direction) if devices is None else list(devices)
    if not available:
        raise DeviceResolutionError(f"no {direction} devices on this machine")

    ordered: list[AudioDevice] = []
    seen: set[tuple[int, str]] = set()

    def add(device: AudioDevice) -> None:
        key = (device.index, device.hostapi_name)
        if key not in seen:
            seen.add(key)
            ordered.append(device)

    # Choosing on the operator's behalf must never land on a loopback input.
    # `spec is None` means "you decide", and deciding to listen to our own
    # output is never the right answer; naming one explicitly still works.
    choosing = spec is None and direction == "input"
    pool = [d for d in available if not _is_loopback(d)] if choosing else available
    if not pool:  # a machine with nothing but loopbacks: better to try than to stop
        pool = available

    try:
        first = resolve_device(spec, direction, devices=pool)
    except DeviceResolutionError:
        # An explicit choice that no longer matches anything. Say so, but do
        # not stop: the fallbacks below are exactly what this moment needs.
        log.warning("no %s device matches %r any more; falling back", direction, spec)
        first = None
    else:
        add(first)

    if first is not None:
        for sibling in sorted(
            (d for d in available if _same_hardware(d, first)), key=_score
        ):
            add(sibling)

    if spec is not None:
        # No default configured is fine: step 4 still has every device.
        with contextlib.suppress(DeviceResolutionError):
            add(resolve_device(None, direction, devices=pool))

    for device in sorted(available, key=_score):
        if direction == "input" and _is_loopback(device):
            continue
        add(device)

    return ordered[:limit]


_COINIT_MULTITHREADED = 0x0
_RPC_E_CHANGED_MODE = -2147417850  # already in a different apartment: fine, proceed
_com_ready: set[int] = set()


def ensure_com() -> None:
    """Put the calling thread in a COM apartment. Windows only; safe to repeat.

    The root cause of the -9999 that would not go away.

    WASAPI is COM, and COM is per-thread. The application builds the pipeline on
    a worker thread (so the page can show progress instead of freezing for forty
    seconds), and a fresh thread has no apartment - so every WASAPI stream
    opened there failed at ``Pa_StartStream``::

        Error starting stream: Unanticipated host error [PaErrorCode -9999]

    Measured, on this laptop, all three WASAPI outputs, ten times over:

        main thread    -> OPENS, OPENS, OPENS
        worker thread  -> FAILS, FAILS, FAILS
        worker + COM   -> OPENS, OPENS, OPENS

    which is why it looked intermittent for so long: it depended on *where* the
    code ran, not on the hardware. Nothing to do with Bluetooth or with torch -
    both were ruled out by running the same test before and after loading them.

    MTA rather than STA, because an apartment-threaded thread owes Windows a
    message pump and an audio worker has nothing to pump it with. The apartment
    is deliberately never released: it was verified that a stream keeps running
    after its opening thread exits *and* after CoUninitialize (48000 callback
    frames in the second following each), because PortAudio holds its own
    reference - but there is nothing to gain by testing that on stage.
    """
    if not sys.platform.startswith("win"):
        return
    ident = threading.get_ident()
    if ident in _com_ready:
        return
    try:
        import ctypes

        hresult = ctypes.windll.ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    except Exception as exc:  # pragma: no cover - no ole32 is not a Windows we know
        log.debug("could not initialise COM on this thread: %s", exc)
        return
    if hresult < 0 and hresult != _RPC_E_CHANGED_MODE:
        log.warning("CoInitializeEx failed with 0x%08x; WASAPI may refuse to start",
                    hresult & 0xFFFFFFFF)
        return
    _com_ready.add(ident)


def open_with_fallback[T](
    spec: int | str | None,
    direction: Direction,
    opener: Callable[[AudioDevice], T],
    *,
    retry_pause: float = 0.35,
) -> T:
    """Open the best device that will actually open, and say so when it is not the best.

    *opener* is called with a device and either returns something or raises.

    Two things make this more than a loop over :func:`candidates`.

    **The first choice gets a second chance.** The failures seen on the
    reference laptop are transient - a Bluetooth headset mid-connection takes
    the WASAPI endpoints down for a moment, then they open normally three times
    in a row. Without a retry the walk settles on DirectSound's "Primary Sound
    Driver" and stays there for the whole talk, which costs tens of
    milliseconds on every utterance and is not even the device the operator
    chose. So if a fallback opened but the *preferred* device had failed, pause
    and try the preferred device once more; keep the fallback only if it fails
    again.

    **The error text lies.** PortAudio's last-host-error is global, so a WASAPI
    failure is reported with whatever text the previous WDM-KS attempt left
    behind::

        Headphones (...) [Windows WASAPI]: Unanticipated host error
        [PaErrorCode -9999]: 'Failed to create render pin: ...'
        [Windows WDM-KS error -9996]

    The device named in that line is right and the explanation after it is not.
    That is why the log records which device was being opened, rather than
    trusting the message to say.
    """
    ensure_com()
    order = candidates(spec, direction)
    tried: list[str] = []
    last: Exception | None = None

    for position, device in enumerate(order):
        # The first choice gets its second chance *here*, before any fallback
        # is opened. Retrying after a fallback succeeded would mean holding two
        # streams at once and deciding which to close, on the audio path, on
        # stage. Cheaper to spend 0.35 s than to own that.
        attempts = 2 if position == 0 else 1
        for attempt in range(attempts):
            try:
                result = opener(verify_index(device))
            except Exception as exc:
                last = exc
                log.warning("could not open %s: %s", device.label, exc)
                if attempt + 1 < attempts:
                    time.sleep(retry_pause)
                continue

            if tried:
                log.warning(
                    "fell back to %s after %d device(s) refused to open",
                    device.label, len(tried),
                )
            return result
        tried.append(device.label)

    raise DeviceResolutionError(
        f"no {direction} device would open. Tried: {', '.join(tried) or 'nothing'}. "
        f"Last error: {last}"
    )


def verify_index(device: AudioDevice) -> AudioDevice:
    """Re-resolve *device* by name, in case the index list has moved.

    PortAudio addresses devices by index and rebuilds that list whenever
    anything connects. A device resolved a moment ago can therefore be opened
    as something else entirely - a Bluetooth headset pairing during startup is
    enough. Checking costs one enumeration and closes the window.
    """
    for candidate in list_devices():
        if (candidate.index == device.index
                and candidate.name == device.name
                and candidate.hostapi_name == device.hostapi_name):
            return device

    # The index moved. Find the same device by name and host API instead.
    for candidate in list_devices():
        if candidate.name == device.name and candidate.hostapi_name == device.hostapi_name:
            log.warning(
                "device index moved: %s was %d, now %d",
                device.name, device.index, candidate.index,
            )
            return candidate
    log.warning("device %s has gone away since it was resolved", device.name)
    return device


def describe_devices(direction: Direction | None = None) -> list[dict[str, Any]]:
    """Serialisable device inventory, used by the CLI and the env report."""
    rows: list[dict[str, Any]] = []
    for d in list_devices(direction):
        rows.append(
            {
                "index": d.index,
                "name": d.name,
                "hostapi": d.hostapi_name,
                "in_ch": d.max_input_channels,
                "out_ch": d.max_output_channels,
                "default_sr": d.default_samplerate,
                "low_in_latency_ms": d.latency_ms("input"),
                "low_out_latency_ms": d.latency_ms("output"),
                "default_input": d.is_default_input,
                "default_output": d.is_default_output,
                "likely_bluetooth": d.likely_bluetooth,
            }
        )
    return rows
