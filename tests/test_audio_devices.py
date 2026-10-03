"""Audio device discovery and selection.

Most tests here run against a synthetic device table so they are deterministic
on any machine (and in CI, where there is no sound card at all). The handful
that touch real hardware are marked ``audio`` and skipped when PortAudio cannot
enumerate anything.
"""

from __future__ import annotations

import sys
import threading

import pytest

from parliamo.audio.devices import (
    AudioDevice,
    DeviceResolutionError,
    _fastest_sibling,
    _same_hardware,
    list_devices,
    resolve_device,
)

FULL_NAME = "Microphone Array (Intel(R) Smart Sound Technology for Digital Microphones)"
MME_NAME = FULL_NAME[:31]  # Windows MME truncates to 31 characters


def _dev(
    index: int,
    name: str,
    hostapi: str,
    *,
    in_ch: int = 2,
    out_ch: int = 0,
    in_lat: float = 0.01,
    out_lat: float = 0.01,
    default_in: bool = False,
    default_out: bool = False,
) -> AudioDevice:
    return AudioDevice(
        index=index,
        name=name,
        hostapi_name=hostapi,
        max_input_channels=in_ch,
        max_output_channels=out_ch,
        default_samplerate=48000.0,
        default_low_input_latency=in_lat,
        default_low_output_latency=out_lat,
        is_default_input=default_in,
        is_default_output=default_out,
    )


@pytest.fixture
def table() -> list[AudioDevice]:
    """A device table shaped like a real Windows machine."""
    return [
        _dev(1, MME_NAME, "MME", in_lat=0.090, default_in=True),
        _dev(6, FULL_NAME, "Windows DirectSound", in_lat=0.120),
        _dev(12, FULL_NAME, "Windows WASAPI", in_lat=0.002),
        _dev(14, "Microphone (Realtek HD Audio Mic input)", "Windows WDM-KS", in_lat=0.010),
        _dev(23, "AirPods Pro (Hands-Free AG Audio)", "Windows WASAPI", in_lat=0.180),
    ]


# ---------------------------------------------------------------------------
# name matching
# ---------------------------------------------------------------------------


def test_same_hardware_matches_mme_truncation() -> None:
    a = _dev(1, MME_NAME, "MME")
    b = _dev(12, FULL_NAME, "Windows WASAPI")
    assert _same_hardware(a, b)


def test_same_hardware_rejects_short_prefix_collision() -> None:
    """'Headphones' must not be treated as the same device as 'Headphones 2 (...)'."""
    a = _dev(22, "Headphones", "Windows WDM-KS", out_ch=2)
    b = _dev(16, "Headphones 2 (Realtek HD Audio 2nd output)", "Windows WDM-KS", out_ch=2)
    assert not _same_hardware(a, b)


def test_fastest_sibling_upgrades_mme_to_wasapi(table: list[AudioDevice]) -> None:
    mme = table[0]
    best = _fastest_sibling(mme, table)
    assert best.hostapi_name == "Windows WASAPI"
    assert best.index == 12


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_default_resolution_prefers_wasapi_over_system_default(table) -> None:
    """The regression this test guards.

    Windows names the 90 ms MME clone as the system default. Taking it at face
    value donates ~88 ms to every segment before a single model has run.
    """
    device = resolve_device(None, "input", devices=table)
    assert device.hostapi_name == "Windows WASAPI"
    assert device.latency_ms("input") == 2.0


def test_resolve_by_index(table) -> None:
    assert resolve_device(14, "input", devices=table).hostapi_name == "Windows WDM-KS"


def test_resolve_by_unknown_index_lists_alternatives(table) -> None:
    with pytest.raises(DeviceResolutionError, match="not found"):
        resolve_device(999, "input", devices=table)


def test_resolve_by_substring_picks_best_hostapi(table) -> None:
    device = resolve_device("Intel(R) Smart Sound", "input", devices=table)
    assert device.hostapi_name == "Windows WASAPI"


def test_resolve_by_substring_is_case_insensitive(table) -> None:
    assert resolve_device("airpods", "input", devices=table).index == 23


def test_resolve_with_hostapi_qualifier(table) -> None:
    device = resolve_device(f"{FULL_NAME}@DirectSound", "input", devices=table)
    assert device.hostapi_name == "Windows DirectSound"


def test_resolve_unmatched_substring_raises(table) -> None:
    with pytest.raises(DeviceResolutionError, match="no input device matching"):
        resolve_device("shure sm7b", "input", devices=table)


def test_resolve_filters_by_direction(table) -> None:
    """No device in the fixture has output channels, so output must fail loudly."""
    with pytest.raises(DeviceResolutionError, match="no output devices"):
        resolve_device(None, "output", devices=table)


# ---------------------------------------------------------------------------
# bluetooth heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "AirPods Pro (Hands-Free AG Audio)",
        "Headset (Bluetooth SCO)",
        "Galaxy Buds3 Pro",
        "WH-1000XM5 Hands-Free",
    ],
)
def test_bluetooth_heuristic_flags_wireless(name: str) -> None:
    assert _dev(0, name, "Windows WASAPI").likely_bluetooth


@pytest.mark.parametrize(
    "name",
    [
        FULL_NAME,
        "Microphone (Realtek HD Audio Mic input)",
        "Rode Wireless GO II RX",  # deliberately excluded from the fixture set below
    ],
)
def test_bluetooth_heuristic_on_wired_devices(name: str) -> None:
    flagged = _dev(0, name, "Windows WASAPI").likely_bluetooth
    # "Rode Wireless GO" contains "wireless" and is expected to trip the
    # heuristic: it is a false positive we accept, because the warning is cheap.
    assert flagged is ("wireless" in name.lower())


# ---------------------------------------------------------------------------
# real hardware
# ---------------------------------------------------------------------------


@pytest.mark.audio
def test_real_machine_has_input_and_output() -> None:
    try:
        inputs = list_devices("input")
        outputs = list_devices("output")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PortAudio unavailable: {exc}")
    if not inputs or not outputs:
        pytest.skip("no audio hardware on this machine")
    assert resolve_device(None, "input", devices=inputs).supports_input
    assert resolve_device(None, "output", devices=outputs).supports_output


# ---------------------------------------------------------------------------
# host API order, and the one that fails
# ---------------------------------------------------------------------------


def test_wdm_ks_is_the_last_resort() -> None:
    """It is the lowest-latency host API on paper and the one that breaks.

    Observed repeatedly on the reference laptop while a Bluetooth device was
    connecting:

        Unanticipated host error [PaErrorCode -9999]
        'GetNameFromCategory: usbTerminalGUID = 7D1E ' [Windows WDM-KS error]

    A fallback exists to reach something that works. A path that intermittently
    refuses to start is not that, whatever its latency.
    """
    from parliamo.audio.devices import _HOSTAPI_PREFERENCE

    assert _HOSTAPI_PREFERENCE[0] == "Windows WASAPI"
    assert _HOSTAPI_PREFERENCE[-1] == "Windows WDM-KS"
    assert _HOSTAPI_PREFERENCE.index("Windows DirectSound") < _HOSTAPI_PREFERENCE.index(
        "Windows WDM-KS"
    ), "DirectSound must be preferred over WDM-KS"


def test_fallback_never_lands_on_wdm_ks_when_anything_else_exists() -> None:
    from parliamo.audio.devices import _score

    devices = [
        _dev(0, "Mic", "Windows WASAPI"),
        _dev(1, "Mic", "Windows WDM-KS"),
        _dev(2, "Mic", "Windows DirectSound"),
        _dev(3, "Mic", "MME"),
    ]
    ordered = sorted(devices, key=_score)
    assert ordered[0].hostapi_name == "Windows WASAPI"
    assert ordered[1].hostapi_name == "Windows DirectSound", (
        "the first fallback is the one that must work"
    )
    assert ordered[-1].hostapi_name == "Windows WDM-KS"


# ---------------------------------------------------------------------------
# indices move
# ---------------------------------------------------------------------------


def test_verify_index_accepts_an_unmoved_dev(monkeypatch) -> None:
    from parliamo.audio import devices as mod

    target = _dev(7, "Mic", "Windows WASAPI")
    monkeypatch.setattr(mod, "list_devices", lambda direction=None: [target])
    assert mod.verify_index(target) is target


def test_verify_index_follows_a_device_that_moved(monkeypatch) -> None:
    """PortAudio rebuilds the index list whenever anything connects.

    A Bluetooth headset pairing between resolving a device and opening it is
    enough to make the index point at something else entirely.
    """
    from parliamo.audio import devices as mod

    resolved = _dev(7, "Mic", "Windows WASAPI")
    moved = _dev(11, "Mic", "Windows WASAPI")
    monkeypatch.setattr(mod, "list_devices",
                        lambda direction=None: [_dev(7, "Something else", "MME"), moved])
    assert mod.verify_index(resolved).index == 11


def test_verify_index_returns_the_original_when_the_device_vanished(monkeypatch) -> None:
    """Better to try and fail with a real error than to guess at a substitute."""
    from parliamo.audio import devices as mod

    resolved = _dev(7, "Mic", "Windows WASAPI")
    monkeypatch.setattr(mod, "list_devices", lambda direction=None: [])
    assert mod.verify_index(resolved) is resolved


# ---------------------------------------------------------------------------
# a retry is only worth anything if the next attempt differs
# ---------------------------------------------------------------------------


@pytest.fixture
def outputs() -> list[AudioDevice]:
    """Shaped like the reference laptop, where every WDM-KS entry fails."""
    return [
        _dev(14, "Headphones (AirPods Pro)", "Windows WASAPI", in_ch=0, out_ch=2),
        _dev(16, "Speakers (Realtek(R) Audio)", "Windows WASAPI", in_ch=0, out_ch=2),
        _dev(11, "Headphones (AirPods Pro)", "Windows DirectSound", in_ch=0, out_ch=2),
        _dev(6, "Speakers (Realtek(R) Audio)", "MME", in_ch=0, out_ch=2),
        _dev(19, "Output 1 (btha2dp.sys)", "Windows WDM-KS", in_ch=0, out_ch=2),
        _dev(22, "Headphones (btha2dp.sys)", "Windows WDM-KS", in_ch=0, out_ch=2),
    ]


def _candidates(monkeypatch, table, spec, direction="output", **kwargs):
    from parliamo.audio import devices as mod

    return mod.candidates(spec, direction, devices=table, **kwargs)


def test_candidates_start_with_what_was_asked_for(monkeypatch, outputs) -> None:
    order = _candidates(monkeypatch, outputs, 19)
    assert order[0].index == 19


def test_candidates_leave_the_broken_host_api_after_the_first_try(monkeypatch, outputs) -> None:
    """The bug this fixes.

    Retrying used to re-resolve the same specification, so three attempts
    produced three identical failures::

        could not open any audio output after 3 attempts. Last error:
        Unanticipated host error [PaErrorCode -9999]
        'GetNameFromCategory: usbTerminalGUID = 7D1E ' [Windows WDM-KS error]

    On the reference laptop all twelve WDM-KS outputs fail to open and all
    eleven others succeed, so a second attempt on the same host API is a
    guaranteed second failure.
    """
    order = _candidates(monkeypatch, outputs, 19)
    assert order[1].hostapi_name != "Windows WDM-KS", (
        "the first fallback must leave the host API that just failed"
    )


def test_candidates_never_repeat_a_device(monkeypatch, outputs) -> None:
    order = _candidates(monkeypatch, outputs, 14)
    keys = [(d.index, d.hostapi_name) for d in order]
    assert len(keys) == len(set(keys)), f"a device is tried twice: {keys}"


def test_candidates_prefer_the_same_hardware_before_anything_else(monkeypatch, outputs) -> None:
    """A headset that failed on one host API is still the headset the operator chose."""
    order = _candidates(monkeypatch, outputs, 22)
    assert order[0].index == 22
    assert order[1].name == "Headphones (AirPods Pro)", (
        "the same physical device on a working host API comes before other hardware"
    )


def test_candidates_fall_through_to_other_hardware(monkeypatch, outputs) -> None:
    """A device can vanish between setup and the button being pressed."""
    order = _candidates(monkeypatch, outputs, 22)
    assert any(d.name.startswith("Speakers") for d in order), (
        "any working speaker beats silence in front of an audience"
    )


def test_candidates_survive_a_spec_that_no_longer_matches(monkeypatch, outputs) -> None:
    """An unplugged device must not stop the talk."""
    order = _candidates(monkeypatch, outputs, "shure sm7b")
    assert order, "an unmatched choice should still offer the working devices"
    assert order[0].hostapi_name == "Windows WASAPI"


def test_candidates_raise_when_there_is_no_hardware_at_all(monkeypatch) -> None:
    order_fn_input: list[AudioDevice] = []
    with pytest.raises(DeviceResolutionError, match="no output devices"):
        _candidates(monkeypatch, order_fn_input, None)


def test_candidates_are_capped(monkeypatch, outputs) -> None:
    """Walking every device on a busy machine would stall the start for minutes."""
    assert len(_candidates(monkeypatch, outputs, None, limit=3)) == 3


@pytest.mark.audio
def test_real_machine_falls_back_off_a_device_that_will_not_open() -> None:
    """End-to-end on whatever hardware is here.

    Skips unless the machine actually has a device that refuses to open, since
    there is nothing to fall back *from* otherwise.
    """
    import sounddevice as sd

    from parliamo.audio.devices import candidates as real_candidates

    try:
        outs = list_devices("output")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PortAudio unavailable: {exc}")

    broken = None
    for device in outs:
        try:
            stream = sd.OutputStream(
                device=device.index,
                channels=min(2, max(1, device.max_output_channels)),
                samplerate=int(round(device.default_samplerate)),
                dtype="float32",
            )
            stream.start()
            stream.stop()
            stream.close()
        except Exception:
            broken = device
            break
    if broken is None:
        pytest.skip("every output device on this machine opens")

    order = real_candidates(broken.index, "output")
    assert len(order) > 1, "a broken device must have somewhere to fall back to"
    assert order[1].hostapi_name != broken.hostapi_name


def test_fallback_refuses_a_loopback_input() -> None:
    """Never let the pipeline listen to its own output.

    "Stereo Mix" and "PC Speaker" capture what the sound card is playing. Fall
    back onto one and the system transcribes its own Italian, translates that,
    and speaks it again - and the half-duplex gate cannot break the loop,
    because it is inside the card rather than across the room. Both are present
    on the reference laptop.
    """
    from parliamo.audio import devices as mod

    table = [
        _dev(29, "Stereo Mix (Realtek HD Audio Stereo input)", "Windows WASAPI"),
        _dev(33, "PC Speaker (Realtek HD Audio 2nd output with SST)", "Windows WASAPI"),
        _dev(17, "Microphone Array (Intel Smart Sound)", "MME"),
    ]
    order = mod.candidates(None, "input", devices=table)
    names = [d.name for d in order]
    assert not any("Stereo Mix" in n or "PC Speaker" in n for n in names), names
    assert "Microphone Array (Intel Smart Sound)" in names, (
        "the slow real microphone is still better than a feedback loop"
    )


def test_an_explicitly_named_loopback_is_still_honoured() -> None:
    """Refusing to fall back onto one is not the same as refusing to use one."""
    from parliamo.audio import devices as mod

    table = [
        _dev(29, "Stereo Mix (Realtek HD Audio Stereo input)", "Windows WASAPI"),
        _dev(17, "Microphone Array (Intel Smart Sound)", "MME"),
    ]
    order = mod.candidates(29, "input", devices=table)
    assert order[0].index == 29


# ---------------------------------------------------------------------------
# opening: the first choice gets a second chance
# ---------------------------------------------------------------------------


def _opener_that_fails(failing: set[int], record: list[int]):
    def opener(device: AudioDevice):
        record.append(device.index)
        if device.index in failing:
            raise OSError(f"Unanticipated host error [PaErrorCode -9999] on {device.index}")
        return device
    return opener


def test_the_preferred_device_is_tried_twice_before_falling_back(monkeypatch, outputs) -> None:
    """Transient failures must not cost the whole talk.

    A Bluetooth headset mid-connection takes the WASAPI endpoints down for a
    moment; measured on the reference laptop, they then open normally three
    times in a row. Settling permanently on DirectSound's "Primary Sound
    Driver" after one blip would spend tens of milliseconds on every utterance
    for the rest of the hour, on a device the operator did not choose.
    """
    from parliamo.audio import devices as mod

    monkeypatch.setattr(mod, "verify_index", lambda d: d)
    monkeypatch.setattr(mod, "candidates", lambda *a, **k: list(outputs))

    record: list[int] = []
    attempts = {"n": 0}

    def opener(device: AudioDevice):
        record.append(device.index)
        if device.index == outputs[0].index:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("busy, just this once")
        return device

    got = mod.open_with_fallback(None, "output", opener, retry_pause=0.0)
    assert got.index == outputs[0].index, "should have recovered the preferred device"
    assert record == [outputs[0].index, outputs[0].index], (
        f"a fallback was opened before the retry: {record}"
    )


def test_a_dead_preferred_device_falls_back_after_two_tries(monkeypatch, outputs) -> None:
    from parliamo.audio import devices as mod

    monkeypatch.setattr(mod, "verify_index", lambda d: d)
    monkeypatch.setattr(mod, "candidates", lambda *a, **k: list(outputs))

    record: list[int] = []
    got = mod.open_with_fallback(
        None, "output", _opener_that_fails({outputs[0].index}, record), retry_pause=0.0
    )
    assert got.index == outputs[1].index
    assert record[:2] == [outputs[0].index] * 2, "the first choice gets exactly two tries"
    assert record.count(outputs[1].index) == 1, "fallbacks get one try each"


def test_only_one_stream_is_ever_opened(monkeypatch, outputs) -> None:
    """Retrying *after* a fallback opened would mean holding two streams.

    Deciding which to close, on the audio path, on stage, is not a position to
    be in - so the retry happens before any fallback is attempted.
    """
    from parliamo.audio import devices as mod

    monkeypatch.setattr(mod, "verify_index", lambda d: d)
    monkeypatch.setattr(mod, "candidates", lambda *a, **k: list(outputs))

    opened: list[int] = []

    def opener(device: AudioDevice):
        if device.index in {outputs[0].index, outputs[1].index}:
            raise OSError("no")
        opened.append(device.index)
        return device

    mod.open_with_fallback(None, "output", opener, retry_pause=0.0)
    assert len(opened) == 1, f"more than one stream was opened: {opened}"


def test_every_device_failing_raises_with_what_was_tried(monkeypatch, outputs) -> None:
    from parliamo.audio import devices as mod

    monkeypatch.setattr(mod, "verify_index", lambda d: d)
    monkeypatch.setattr(mod, "candidates", lambda *a, **k: list(outputs))

    record: list[int] = []
    failing = {d.index for d in outputs}
    with pytest.raises(DeviceResolutionError, match="no output device would open"):
        mod.open_with_fallback(
            None, "output", _opener_that_fails(failing, record), retry_pause=0.0
        )


def test_control_characters_are_stripped_from_device_names() -> None:
    """A real name from this laptop, with a literal newline inside it.

    Left alone it splits log lines in half and makes the Setup list unreadable.
    """
    from parliamo.audio.devices import _clean_name

    raw = "Output 1 (@System32\drivers\btha2dp.sys,#1;%1%0\n;(AirPods Pro - Find My))"
    cleaned = _clean_name(raw)
    assert "\n" not in cleaned
    assert cleaned.startswith("Output 1 (@System32")
    assert cleaned.endswith("(AirPods Pro - Find My))")


def test_cleaning_leaves_ordinary_names_alone() -> None:
    from parliamo.audio.devices import _clean_name

    assert _clean_name(" Speakers (Realtek(R) Audio) ") == "Speakers (Realtek(R) Audio)"


# ---------------------------------------------------------------------------
# COM: the actual root cause
# ---------------------------------------------------------------------------


def test_open_with_fallback_initialises_com_first(monkeypatch, outputs) -> None:
    """WASAPI is COM, and COM is per-thread.

    The pipeline is built on a worker thread so the page can show progress
    instead of freezing for forty seconds. A fresh thread has no COM apartment,
    so every WASAPI stream opened there failed at ``Pa_StartStream`` with
    ``Unanticipated host error [PaErrorCode -9999]`` - which read as a hardware
    fault for days, because it depended on *where* the code ran rather than on
    the device.
    """
    from parliamo.audio import devices as mod

    calls: list[str] = []
    monkeypatch.setattr(mod, "verify_index", lambda d: d)
    monkeypatch.setattr(mod, "candidates", lambda *a, **k: list(outputs))
    monkeypatch.setattr(mod, "ensure_com", lambda: calls.append("com"))

    mod.open_with_fallback(None, "output", lambda d: (calls.append("open"), d)[1])
    assert calls[0] == "com", f"COM must be ready before the first open: {calls}"


def test_ensure_com_is_idempotent_per_thread(monkeypatch) -> None:
    """It is called on every open; it must not pay for COM every time."""
    from parliamo.audio import devices as mod

    if not sys.platform.startswith("win"):
        pytest.skip("COM is a Windows concern")

    monkeypatch.setattr(mod, "_com_ready", set())
    calls = {"n": 0}

    class FakeOle:
        def CoInitializeEx(self, _reserved, _mode):  # noqa: N802 - Windows spelling
            calls["n"] += 1
            return 0

    class FakeWindll:
        ole32 = FakeOle()

    import ctypes

    monkeypatch.setattr(ctypes, "windll", FakeWindll(), raising=False)
    mod.ensure_com()
    mod.ensure_com()
    mod.ensure_com()
    assert calls["n"] == 1


def test_ensure_com_accepts_a_thread_already_in_another_apartment(monkeypatch) -> None:
    """RPC_E_CHANGED_MODE means COM is up, just not ours. That is fine."""
    from parliamo.audio import devices as mod

    if not sys.platform.startswith("win"):
        pytest.skip("COM is a Windows concern")

    monkeypatch.setattr(mod, "_com_ready", set())

    class FakeOle:
        def CoInitializeEx(self, _reserved, _mode):  # noqa: N802 - Windows spelling
            return mod._RPC_E_CHANGED_MODE

    class FakeWindll:
        ole32 = FakeOle()

    import ctypes

    monkeypatch.setattr(ctypes, "windll", FakeWindll(), raising=False)
    mod.ensure_com()  # must not raise
    assert threading.get_ident() in mod._com_ready


@pytest.mark.audio
def test_a_worker_thread_gets_wasapi_on_real_hardware() -> None:
    """The regression, end to end, on whatever this machine has.

    Before the fix, measured ten times over on the reference laptop: all three
    WASAPI outputs opened from the main thread and none of them from a worker,
    so the application silently ran the whole talk on DirectSound.
    """
    import threading as _threading

    from parliamo.audio.playback import AudioPlayback

    try:
        outs = list_devices("output")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PortAudio unavailable: {exc}")
    if not any(d.hostapi_name == "Windows WASAPI" for d in outs):
        pytest.skip("no WASAPI on this machine")

    result: dict[str, object] = {}

    def build() -> None:
        playback = AudioPlayback(sample_rate=24000)
        try:
            result["info"] = playback.start()
            result["playback"] = playback
        except Exception as exc:  # pragma: no cover
            result["error"] = exc

    worker = _threading.Thread(target=build, name="pipeline-start")
    worker.start()
    worker.join(timeout=60)

    if "error" in result:  # pragma: no cover
        pytest.skip(f"no output device available: {result['error']}")
    info = result["info"]
    try:
        assert info.device.hostapi_name == "Windows WASAPI", (
            f"a worker thread fell back to {info.device.hostapi_name}; "
            "COM was probably not initialised"
        )
    finally:
        result["playback"].stop()


# ---------------------------------------------------------------------------
# a choice that will not open falls back to what Windows itself is using
# ---------------------------------------------------------------------------


@pytest.fixture
def inputs_2026_10_02() -> list[AudioDevice]:
    """The machine at rehearsal: a microphone in the jack, a Dante interface.

    The jack microphone is "Realtek HD Audio Mic input" on WDM-KS and
    "Microphone (Realtek(R) Audio)" everywhere else - two names, one device,
    so the sibling match cannot pair them. Windows had it as the default.
    """
    return [
        _dev(1, "Microphone (Realtek(R) Audio)", "MME", default_in=True),
        _dev(2, "Microphone Array (Intel® Smart", "MME"),
        _dev(3, "Microphone (Dante USB I/O Modul", "MME"),
        _dev(23, "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)",
             "Windows WASAPI"),
        _dev(22, "Microphone (Realtek(R) Audio)", "Windows WASAPI"),
        _dev(24, "Microphone (Dante USB I/O Module)", "Windows WASAPI"),
        _dev(26, "Microphone (Realtek HD Audio Mic input)", "Windows WDM-KS"),
    ]


def test_a_choice_that_will_not_open_falls_back_to_the_windows_default_first(
        monkeypatch, inputs_2026_10_02) -> None:
    """Rehearsal: the WDM-KS entry of the jack microphone refused (-9996), and
    the walk went to the laptop's own array - the microphone that gates 16-50%
    of speech to silence - while the jack microphone Windows was using sat one
    entry further on."""
    order = _candidates(monkeypatch, inputs_2026_10_02, 26, direction="input")
    assert order[0].index == 26
    assert (order[1].name, order[1].hostapi_name) == (
        "Microphone (Realtek(R) Audio)", "Windows WASAPI")


def test_the_windows_default_is_not_tried_twice(monkeypatch, inputs_2026_10_02) -> None:
    order = _candidates(monkeypatch, inputs_2026_10_02, 22, direction="input")
    keys = [(d.index, d.hostapi_name) for d in order]
    assert len(keys) == len(set(keys))
    assert order[0].index == 22


def test_a_choice_by_name_survives_windows_renumbering(monkeypatch, inputs_2026_10_02) -> None:
    """The page now sends "name@host API". At rehearsal the index of the jack
    microphone's entry pointed at a "PC Speaker" loopback minutes later."""
    renumbered = [
        _dev(d.index + 3, d.name, d.hostapi_name, default_in=d.is_default_input)
        for d in inputs_2026_10_02
    ] + [_dev(26, "PC Speaker (Realtek HD Audio 2nd output with SST)", "Windows WDM-KS")]
    order = _candidates(monkeypatch, renumbered,
                        "Microphone (Realtek(R) Audio)@Windows WASAPI", direction="input")
    assert (order[0].name, order[0].hostapi_name) == (
        "Microphone (Realtek(R) Audio)", "Windows WASAPI")
    assert order[0].index == 25


def test_the_page_sends_device_names_not_indices() -> None:
    import re

    from parliamo.ui.server import APP

    page = APP.read_text(encoding="utf-8")
    assert "const key = (d) => `${d.name}@${d.hostapi}`;" in page
    assert re.search(r"const value = el\.value === \"\" \? null : el\.value;", page), (
        "the picked value must be sent as the name, not converted to a number")


def test_mme_and_directsound_speakers_are_known_to_stutter() -> None:
    """45 underflows in 8 s on MME under load at rehearsal; WASAPI none."""
    from parliamo.audio.devices import stutters_under_load

    def speakers(api: str) -> AudioDevice:
        return _dev(1, "Speakers (Dante USB I/O Module)", api, in_ch=0, out_ch=2)

    assert stutters_under_load(speakers("MME"))
    assert stutters_under_load(speakers("Windows DirectSound"))
    assert not stutters_under_load(speakers("Windows WASAPI"))


def test_the_speaker_picker_says_mme_stutters_and_offers_wasapi_first() -> None:
    from parliamo.ui.server import APP

    page = APP.read_text(encoding="utf-8")
    assert "may stutter while translating" in page
    assert ('dir === "output" && (d.hostapi === "MME" || d.hostapi === "Windows DirectSound")'
            in page)
    assert "sort((a, b) => rank(a) - rank(b))" in page
