# The design this UI was built from

`Translation Local UI.dc.html` is the handoff bundle from Claude Design
(claude.ai/design), kept here so the implementation can be checked against its
source rather than against a memory of it.

**Read turn 2, not turn 1.** The file contains both. Turn 1 was drawn before the
designer could read this codebase and describes a different product — a
document-translation studio with model pickers and a history rail. The file says
so itself and keeps it only as history. Turn 2 is the real design:

| | |
|---|---|
| `2a` | operator window, `GET /` — the five tabs |
| `2b` | audience screen, `GET /screen` — projected |
| `2c` | coverage map: every route and control, and where it lives |

The prototype is HTML with `{{ }}` bindings and `<sc-if>` blocks, rendered by
the canvas runtime (`support.js`, not kept — it is React and 69 KB, and nothing
here needs it). The bundle's own instructions are to recreate the *visual
output*, not the prototype's internals, which is what
`src/parliamo/ui/app.html` and `page.html` do.

## Where the implementation departs from it

Three deliberate differences, each because the design was drawn against sample
data and the real thing has to work:

- **The specimen cards are states, not cards.** 2a shows the "starting" and
  "failed" cards side by side to illustrate both. The application renders
  whichever one is true.
- **Every number is live.** The design's figures (11 sentences, 3.66 s, 7292
  MiB, three latency runs) came from this project's measurements; the page reads
  them from `/state`, `/metrics` and `/measurements` rather than hardcoding them.
- **Distinct `<title>`s.** Both windows are open at once on the night, and the
  browser tab is the only thing telling them apart.

The two questions 2c said it could not decide are answered the way the code
already had them: the operator window and the projected screen stay separate
pages, and the latency breakdown stays on the operator's side only.
