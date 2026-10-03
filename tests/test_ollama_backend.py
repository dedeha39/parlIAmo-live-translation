"""The Ollama translation backend sends each model the prompt it was trained on.

No server is needed: the request is captured instead of sent.
"""

from __future__ import annotations

from parliamo.mt.ollama_backend import OllamaBackend, prompt_for, raw_wrapper_for

HY_MT = "hf.co/tencent/HY-MT1.5-1.8B-GGUF:Q4_K_M"


def _captured(model: str) -> tuple[OllamaBackend, list[dict]]:
    backend = OllamaBackend(model=model, source_lang="tr", target_lang="it")
    sent: list[dict] = []

    def request(path, payload=None):
        sent.append(payload or {})
        return {"response": " Questa non è una questione tecnica. "}

    backend._request = request
    return backend, sent


def test_hy_mt_gets_its_own_instruction_not_the_generic_one() -> None:
    """Its GGUF name has no "hunyuan" in it; it used to fall through to the generic prompt."""
    assert prompt_for(HY_MT).startswith("Translate the following segment into {tgt_name}")


def test_hy_mt_is_sent_raw_with_its_turn_markers() -> None:
    """Ollama's own template for this model dropped the prompt: every answer was "onse }"."""
    backend, sent = _captured(HY_MT)
    out = backend._translate("Bu teknik bir mesele değil.", "tr", "it")
    payload = sent[0]
    assert payload["raw"] is True
    assert payload["prompt"].startswith("<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate")
    assert payload["prompt"].endswith("Bu teknik bir mesele değil.<｜hy_Assistant｜>")
    assert "<｜hy_place▁holder▁no▁2｜>" in payload["options"]["stop"]
    assert out == "Questa non è una questione tecnica."


def test_a_model_whose_template_works_is_left_to_ollama() -> None:
    backend, sent = _captured("hf.co/someone/translategemma-4b-it-Q8_0-GGUF")
    backend._translate("Merhaba.", "tr", "it")
    assert "raw" not in sent[0] and "stop" not in sent[0]["options"]
    assert raw_wrapper_for("phi3:3.8b") is None


def test_the_server_is_addressed_by_ip_not_by_localhost() -> None:
    """"localhost" tried IPv6 first on Windows: ~2 s on every request before IPv4."""
    from parliamo.mt.ollama_backend import DEFAULT_HOST

    assert "localhost" not in DEFAULT_HOST and "127.0.0.1" in DEFAULT_HOST
