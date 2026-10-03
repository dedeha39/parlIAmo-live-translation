"""The operator and audience surface.

Two jobs on one local page, because on stage there is one laptop and no time:

* **The audience reads the translation.** With a 10-second sentence the audio
  cannot arrive sooner than the sentence ends, but the text can - provisional
  subtitles land 1.68 s in (ADR 0008). The screen is what keeps a room with a
  speaker while the audio catches up.
* **The operator sees whether it is alive**, and can silence it instantly.

Everything is served from localhost with no external assets. The venue's
internet cannot be trusted, and a page that fetches a font from a CDN is a page
that renders wrong in front of an audience.
"""

from .server import SubtitleServer, UIState

__all__ = ["SubtitleServer", "UIState"]
