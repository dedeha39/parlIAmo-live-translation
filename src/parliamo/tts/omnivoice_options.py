"""OmniVoice's generation settings, in one table.

The Studio draws its sliders from this table and the OmniVoice service checks
every request against it, so a slider cannot offer a value the service would
refuse, and a request cannot smuggle in a setting the page does not show.
Ranges are wider than the model's defaults but narrower than "anything":
values far outside them produce noise, not a different voice.

Defaults are OmniVoice's own (``OmniVoiceGenerationConfig``), except
``num_step``: 16, not 32, measured on this laptop (ADR 0015) - 32 was both
slower and less intelligible.
"""

from __future__ import annotations

import math
from typing import Any

#: name -> slider. ``default`` None means "the model decides" (speed from the
#: text, duration from the speed).
ADVANCED: dict[str, dict[str, Any]] = {
    "num_step": {
        "label": "Steps", "min": 4, "max": 64, "step": 1, "default": 16, "kind": "int",
        "help": "Decoding passes. 16 was the most intelligible here; 8 is twice as fast "
                "and drops words; 32 was slower and no better.",
    },
    "guidance_scale": {
        "label": "Guidance", "min": 0.0, "max": 6.0, "step": 0.1, "default": 2.0,
        "kind": "float",
        "help": "How strictly it follows the text and the voice. Higher is stricter; "
                "too high sounds strained.",
    },
    "speed": {
        "label": "Speed", "min": 0.5, "max": 2.0, "step": 0.05, "default": None,
        "kind": "float",
        "help": "Speaking rate. Auto lets the model estimate it from the reference.",
    },
    "duration": {
        "label": "Fixed length (s)", "min": 0.5, "max": 60.0, "step": 0.5, "default": None,
        "kind": "float",
        "help": "Force the clip to this many seconds. Overrides speed.",
    },
    "class_temperature": {
        "label": "Variation", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.0,
        "kind": "float",
        "help": "Token sampling temperature. 0 is the most stable; higher varies more "
                "between renders.",
    },
    "position_temperature": {
        "label": "Order temperature", "min": 0.0, "max": 10.0, "step": 0.1, "default": 5.0,
        "kind": "float",
        "help": "How freely it chooses which parts of the sound to fill in first.",
    },
    "layer_penalty_factor": {
        "label": "Layer penalty", "min": 0.0, "max": 10.0, "step": 0.1, "default": 5.0,
        "kind": "float",
        "help": "Pushes coarse sound layers to be decided before fine ones.",
    },
    "t_shift": {
        "label": "Time shift", "min": 0.01, "max": 1.0, "step": 0.01, "default": 0.1,
        "kind": "float",
        "help": "Where the decoding spends its effort; smaller favours the noisy start.",
    },
    "pad_duration": {
        "label": "Silence each side (s)", "min": 0.0, "max": 1.0, "step": 0.05,
        "default": 0.1, "kind": "float", "help": "Silence added before and after.",
    },
    "fade_duration": {
        "label": "Fade (s)", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.1,
        "kind": "float", "help": "Fade-in and fade-out length.",
    },
}

#: On/off settings, with the model's defaults.
FLAGS: dict[str, dict[str, Any]] = {
    "denoise": {"label": "Denoise", "default": True,
                "help": "Ask for clean speech even if the reference is noisy."},
    "postprocess_output": {"label": "Tidy output", "default": True,
                           "help": "Remove long silences, fade and pad the edges."},
}

#: The inline tags OmniVoice speaks as sounds (models/omnivoice.py,
#: _NONVERBAL_PATTERN), with what they are.
NONVERBAL: dict[str, str] = {
    "laughter": "laughs", "sigh": "sighs", "surprise-ah": "surprised ah",
    "surprise-oh": "surprised oh", "surprise-wa": "surprised wa", "surprise-yo": "surprised yo",
    "question-ah": "questioning ah", "question-oh": "questioning oh",
    "question-ei": "questioning ei", "question-yi": "questioning yi",
    "question-en": "questioning hm", "confirmation-en": "agreeing mm",
    "dissatisfaction-hnn": "displeased hnn",
}

#: Voice-design attributes (omnivoice/utils/voice_design.py). Accents are
#: English-only in the model: they colour English speech.
DESIGN: dict[str, list[str]] = {
    "gender": ["female", "male"],
    "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "accent": ["", "american accent", "british accent", "australian accent", "canadian accent",
               "indian accent", "chinese accent", "japanese accent", "korean accent",
               "portuguese accent", "russian accent"],
}


def clean_options(raw: dict[str, Any] | None) -> dict[str, Any]:
    """The settings in *raw* that exist, as numbers inside their range.

    Unknown keys are dropped; empty or None means the default; out-of-range
    values are clamped rather than refused, because a slider at its end is
    not an error.
    """
    raw = raw or {}
    out: dict[str, Any] = {}
    for name, spec in ADVANCED.items():
        value = raw.get(name)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec['label']} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{spec['label']} must be a number")
        number = min(max(number, spec["min"]), spec["max"])
        out[name] = int(round(number)) if spec["kind"] == "int" else round(number, 3)
    for name in FLAGS:
        if name in raw and raw[name] is not None:
            out[name] = bool(raw[name])
    return out


def instruct_for(gender: str, age: str, pitch: str, accent: str = "",
                 whisper: bool = False) -> str:
    """The voice-design instruction, from attributes the model knows."""
    chosen = {"gender": gender, "age": age, "pitch": pitch, "accent": accent}
    for key, value in chosen.items():
        if value not in DESIGN[key]:
            allowed = ", ".join(v for v in DESIGN[key] if v) or "none"
            raise ValueError(f"{key} must be one of: {allowed}")
    parts = [gender, age, pitch] + ([accent] if accent else []) + (["whisper"] if whisper else [])
    return ", ".join(parts)
