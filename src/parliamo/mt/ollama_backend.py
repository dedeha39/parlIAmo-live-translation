"""Translation through a locally running Ollama server.

Why this backend exists
-----------------------
Several candidate models were eliminated from the plan on *estimated* VRAM
rather than measured VRAM, and the estimates had already been wrong once by a
factor of four. Testing them properly needed a way to run GGUF models, and
building ``llama-cpp-python`` with CUDA on Windows failed on a missing shared
library. Ollama was already installed, serves GGUF over HTTP, and needs no
build step - so the models can be measured instead of guessed at.

It also does something worth knowing about: when a model does not fit in VRAM,
Ollama **silently offloads layers to the CPU** rather than failing. A 12B model
will therefore appear to "work" while running at a fraction of the speed. That
is exactly the kind of thing an estimate cannot tell you, so this backend
reports how much of the model actually landed on the GPU.

Architectural note: Ollama is a separate process holding its own VRAM. That is
convenient for measurement, but in production it means two processes competing
for one card, and the budget has to account for both.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from .base import MTBackend

log = logging.getLogger(__name__)

#: An address, not "localhost". On Windows "localhost" resolves to ::1 first;
#: Ollama listens on IPv4 only, so every request waited ~2 s for the IPv6
#: attempt to fail before falling back. Measured 2026-09-23: 2274 ms per
#: sentence through "localhost" against 222 ms through 127.0.0.1, for a model
#: that spent 230 ms of it working. ADR 0004's TranslateGemma latency was
#: measured through the same default and carries the same ~2 s.
DEFAULT_HOST = "http://127.0.0.1:11434"

LANGUAGE_NAMES = {
    "tr": "Turkish",
    "it": "Italian",
    "en": "English",
    "fur": "Friulian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

#: Different translation models want different prompts, and using the wrong one
#: costs far more quality than any decoding setting. Selected by substring match
#: on the model name, longest match first.
PROMPT_TEMPLATES: dict[str, str] = {
    # TranslateGemma expects the language pair stated as codes.
    "translategemma": "{src_code}: {text}\n{tgt_code}:",
    # Hunyuan-MT's documented instruction format, for pairs without Chinese.
    # HY-MT1.5 publishes its GGUF as "HY-MT1.5-...", without "hunyuan" in the
    # name, and would have fallen through to the generic prompt.
    "hunyuan": (
        "Translate the following segment into {tgt_name}, without additional "
        "explanation.\n\n{text}"
    ),
    "hy-mt": (
        "Translate the following segment into {tgt_name}, without additional "
        "explanation.\n\n{text}"
    ),
    # MiLMMT follows the common instruction-tuned translation phrasing.
    "milmmt": (
        "Translate the following {src_name} text into {tgt_name}. "
        "Output only the translation.\n\n{text}"
    ),
}

#: Models whose chat template Ollama derives wrongly, sent raw and wrapped by
#: hand. HY-MT1.5's GGUF carries no chat template, and the one Ollama builds
#: for the hunyuan-dense architecture drops ``{{ .Prompt }}`` altogether: the
#: model never saw the text, and every answer was "onse }" after 2.1 s. With
#: the wrapper below the same model answers in 0.23-0.35 s.
RAW_WRAPPERS: dict[str, tuple[str, list[str]]] = {
    "hy-mt": (
        "<｜hy_begin▁of▁sentence｜><｜hy_User｜>{prompt}<｜hy_Assistant｜>",
        ["<｜hy_place▁holder▁no▁2｜>", "<｜hy_end▁of▁sentence｜>", "<｜hy_User｜>"],
    ),
}


def raw_wrapper_for(model: str) -> tuple[str, list[str]] | None:
    lowered = model.lower()
    for key in sorted(RAW_WRAPPERS, key=len, reverse=True):
        if key in lowered:
            return RAW_WRAPPERS[key]
    return None


GENERIC_PROMPT = (
    "Translate the following {src_name} text into {tgt_name}. "
    "Reply with the translation only, no explanation, no quotes.\n\n{text}"
)


def prompt_for(model: str) -> str:
    lowered = model.lower()
    for key in sorted(PROMPT_TEMPLATES, key=len, reverse=True):
        if key in lowered:
            return PROMPT_TEMPLATES[key]
    return GENERIC_PROMPT


class OllamaBackend(MTBackend):
    name = "ollama"

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        source_lang: str = "tr",
        target_lang: str = "it",
        host: str = DEFAULT_HOST,
        temperature: float = 0.0,
        num_predict: int = 512,
        num_ctx: int = 2048,
        keep_alive: str = "5m",
        prompt_template: str | None = None,
        timeout: float = 180.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model, device=device, source_lang=source_lang, target_lang=target_lang,
            temperature=temperature, num_ctx=num_ctx, **kwargs,
        )
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.prompt_template = prompt_template or prompt_for(model)
        self.timeout = timeout
        self.gpu_fraction: float | None = None

    # -- lifecycle --------------------------------------------------------

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach Ollama at {self.host}: {exc}. Start it with 'ollama serve'."
            ) from exc

    def _load(self) -> None:
        # An empty prompt makes Ollama load the model without generating, which
        # is what lets the VRAM measurement attribute the load correctly.
        self._request("/api/generate", {
            "model": self.model, "prompt": "", "keep_alive": self.keep_alive,
        })
        self._report_placement()

    def _report_placement(self) -> None:
        """Say how much of the model actually reached the GPU.

        Ollama offloads to CPU rather than failing when a model does not fit,
        so 'it ran' is not the same as 'it fits'. Without this, a 12B model
        looks like a success that happens to be slow.
        """
        try:
            running = self._request("/api/ps")
        except Exception:  # pragma: no cover
            return
        for entry in running.get("models", []):
            if entry.get("name", "").startswith(self.model.split(":")[0]):
                total = entry.get("size", 0)
                on_gpu = entry.get("size_vram", 0)
                if total:
                    self.gpu_fraction = on_gpu / total
                    log.info(
                        "%s: %.0f%% on GPU (%.2f of %.2f GB)",
                        self.model, 100 * self.gpu_fraction,
                        on_gpu / 1e9, total / 1e9,
                    )
                    if self.gpu_fraction < 0.99:
                        log.warning(
                            "%s does not fit in VRAM - Ollama has offloaded %.0f%% to CPU. "
                            "It will run, but slowly, and the speed figures are not "
                            "comparable with fully-resident models.",
                            self.model, 100 * (1 - self.gpu_fraction),
                        )
                return

    def _unload(self) -> None:
        # Best-effort: the server may already have evicted the model, and a
        # failure to unload must not break a benchmark that is otherwise done.
        with contextlib.suppress(Exception):
            self._request("/api/generate", {
                "model": self.model, "prompt": "", "keep_alive": 0,
            })

    # -- translation ------------------------------------------------------

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        prompt = self.prompt_template.format(
            text=text,
            src_code=source_lang,
            tgt_code=target_lang,
            src_name=LANGUAGE_NAMES.get(source_lang, source_lang),
            tgt_name=LANGUAGE_NAMES.get(target_lang, target_lang),
        )
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            # Deterministic decoding: the same input must give the same
            # output, or a benchmark measures sampling luck.
            "top_k": 1,
            "top_p": 1.0,
            "seed": 0,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        wrapper = raw_wrapper_for(self.model)
        if wrapper is not None:
            template, stops = wrapper
            payload["prompt"] = template.format(prompt=prompt)
            payload["raw"] = True
            options["stop"] = stops
        response = self._request("/api/generate", payload)
        return str(response.get("response", "")).strip()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        if self.gpu_fraction is not None:
            info["gpu_fraction"] = round(self.gpu_fraction, 3)
        return info
