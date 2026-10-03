"""NLLB-200 via CTranslate2.

Why this backend exists even though LLMs translate better on tr-it:

**Friulian.** NLLB-200 is the only pretrained translation model that has seen
Friulian (``fur_Latn``) at all. Every other candidate covers 33-140 well-resourced
languages and none of them include it. The Friulian branch of this project has
to run through here, so the backend is needed regardless of how it scores on
Turkish to Italian.

**A CPU option.** At 600M parameters it runs usefully fast on the 13900HX,
which is the lever that frees ~3 GB of VRAM if the ASR stage needs to grow.

**A second implementation.** The ASR layer shipped an abstraction with exactly
one backend behind it, which is a guess about what varies rather than a fact.
Having a sequence-to-sequence model and an instruction-following LLM behind the
same interface is what makes it a real interface.

Licence note: NLLB-200 is **CC-BY-NC**. Fine for a research presentation, a
blocker for anything commercial. See docs/08-model-licences.md.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import MTBackend

log = logging.getLogger(__name__)

#: Neutral codes to NLLB's FLORES-200 tags.
LANG_TAGS = {
    "tr": "tur_Latn",
    "it": "ita_Latn",
    "en": "eng_Latn",
    "fur": "fur_Latn",     # Friulian - the reason this backend exists
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
}

DEFAULT_TOKENIZER = "facebook/nllb-200-distilled-600M"


class CTranslate2NLLBBackend(MTBackend):
    name = "ctranslate2_nllb"

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        source_lang: str = "tr",
        target_lang: str = "it",
        tokenizer: str = DEFAULT_TOKENIZER,
        compute_type: str = "int8_float16",
        beam_size: int = 4,
        max_decoding_length: int = 256,
        cpu_threads: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model, device=device, source_lang=source_lang, target_lang=target_lang,
            compute_type=compute_type, beam_size=beam_size, **kwargs,
        )
        self.tokenizer_id = tokenizer
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.max_decoding_length = max_decoding_length
        self.cpu_threads = cpu_threads
        self._translator: Any = None
        self._tokenizer: Any = None

    # -- lifecycle --------------------------------------------------------

    def _load(self) -> None:
        import ctranslate2
        import transformers

        from ..paths import resolve

        path = resolve(self.model)
        compute = self.compute_type
        if self.device == "cpu" and compute == "int8_float16":
            # int8_float16 is a GPU compute type; on CPU it silently degrades.
            compute = "int8"

        log.info("loading NLLB %s on %s (%s)", path, self.device, compute)
        self._translator = ctranslate2.Translator(
            str(path),
            device=self.device,
            compute_type=compute,
            inter_threads=1,
            intra_threads=self.cpu_threads or 0,
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_id)

    def _unload(self) -> None:
        self._translator = None
        self._tokenizer = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # -- translation ------------------------------------------------------

    @staticmethod
    def tag(lang: str) -> str:
        if lang in LANG_TAGS:
            return LANG_TAGS[lang]
        if "_" in lang:  # already a FLORES tag
            return lang
        raise KeyError(
            f"no NLLB tag for language {lang!r}. Known: {', '.join(sorted(LANG_TAGS))}"
        )

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return self._translate_batch([text], source_lang, target_lang)[0]

    def _translate_pieces(
        self, pieces: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """One batch for all the sentences in a segment, not one call each."""
        return self._translate_batch(pieces, source_lang, target_lang)

    def _translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        self._tokenizer.src_lang = self.tag(source_lang)
        target_tag = self.tag(target_lang)

        sources = [
            self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(t)) for t in texts
        ]
        results = self._translator.translate_batch(
            sources,
            target_prefix=[[target_tag]] * len(sources),
            beam_size=self.beam_size,
            max_decoding_length=self.max_decoding_length,
            # NLLB has no chat behaviour to guard against, so decoding stays
            # plain: no sampling, no repetition tricks.
        )

        out: list[str] = []
        for result in results:
            tokens = result.hypotheses[0]
            if tokens and tokens[0] == target_tag:
                tokens = tokens[1:]  # drop the forced language prefix
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            out.append(self._tokenizer.decode(ids, skip_special_tokens=True).strip())
        return out

    def translate_many(self, texts, source_lang=None, target_lang=None):
        """Batched: CTranslate2 handles a batch far better than a loop."""
        from .base import Translation, sanitise

        if not self._loaded:
            self.load()
        src = source_lang or self.source_lang
        tgt = target_lang or self.target_lang

        import time

        stripped = [t.strip() for t in texts]
        indices = [i for i, t in enumerate(stripped) if t]
        if not indices:
            return [
                Translation(text="", source=t, source_lang=src, target_lang=tgt,
                            backend=f"{self.name}:{self.model}")
                for t in texts
            ]

        t0 = time.perf_counter()
        raw = self._translate_batch([stripped[i] for i in indices], src, tgt)
        elapsed = time.perf_counter() - t0
        per_item = elapsed / len(indices)

        results: list[Translation] = []
        produced = dict(zip(indices, raw, strict=True))
        for i, original in enumerate(texts):
            if i not in produced:
                results.append(
                    Translation(text="", source=original, source_lang=src, target_lang=tgt,
                                backend=f"{self.name}:{self.model}")
                )
                continue
            cleaned, repaired = sanitise(produced[i], stripped[i])
            results.append(
                Translation(
                    text=cleaned, source=stripped[i], source_lang=src, target_lang=tgt,
                    compute_s=per_item, backend=f"{self.name}:{self.model}", repaired=repaired,
                )
            )
        return results
