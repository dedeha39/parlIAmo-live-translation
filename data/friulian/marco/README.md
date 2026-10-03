# Marco Moroldo — a native Friulian speaker

The recording is not in git (38 MB). Fetch it from Wikimedia Commons:

    https://commons.wikimedia.org/wiki/File:Wikitongues-Friulian-Moroldo.webm

Save it here as `wikitongues-friulian-moroldo.webm`, then:

    python scripts/prepare_native_clips.py data/friulian/marco/wikitongues-friulian-moroldo.webm \
        --speaker "Marco Moroldo" \
        --source "https://commons.wikimedia.org/wiki/File:Wikitongues-Friulian-Moroldo.webm" \
        --licence "CC BY-SA 4.0"

That writes `clips/marco-NNN.wav` and `clips/index.json`, and the Friulian tab
shows a *Marco Moroldo — real Friulian* section with one play button per sentence.

**Licence:** CC BY-SA 4.0 (Wikitongues). Play it, with attribution.
**Not a voice reference.** The licence covers the recording, not the speaker's
voice; this folder is never listed in the Voices tab and never handed to the
synthesiser. The point of the segment is to let the room hear what Friulian
sounds like from someone who speaks it, beside what the pipeline makes of it.
