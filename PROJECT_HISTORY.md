# TopoForge — Project History & Working Conventions

This file supplements CLAUDE.md's auto-generated structural notes with
context that only exists in the project's history, not in the code
itself. Read this before making changes.

## What this project is, in one paragraph

TopoForge studies whether the physical placement of neurons on a
neuromorphic chip affects what the chip can learn — not just its
energy use, which is what every existing chip-mapping tool optimizes
for. The core finding: placing functionally-different neurons close
together ("interleaved") lets structural plasticity build cross-type
associations that placing them in function-grouped blocks
("segregated") makes much harder or impossible. This has now been
confirmed at multiple scales, with geometry removed entirely (pure
graph locality), and on real spike-encoded human speech data — not
just synthetic patterns.

## Two repositories — know which one you're in

- **`topoforge`** (public, github.com/bowills-srb/topoforge) — the
  main research repo. 37+ numbered experiments in `src/experiments/`,
  the core engine (`engine.py`, `engine_local.py`, `engine_fast.py`,
  `sparse_state.py`, `spatial.py`), the preprint draft, the CLI
  product (`topoforge_run.py`). **This repo needs to stay stable and
  citable** — it's what a collaborator (Henry, at Catalyst
  Neuromorphic) may test against, and what backs the filed provisional
  patent. Don't break working experiments; if you change shared engine
  code, re-run the affected experiments to confirm results still hold.

- **`topoforge-scale`** (private) — a from-scratch rewrite aimed at
  running the same physics at N=500,000 instead of N~1,000-5,000. Has
  its own `ScaleLife` engine, Numba-compiled kernels
  (`numba_kernels.py`, `numba_sparse_state.py`), and its own audit
  suite. This repo is allowed to be messy and in-progress. It only
  merges back into `topoforge` after full equivalence-audit passage.

## The single most important convention: audit before you trust

Every non-trivial change in this project's history has been validated
by comparing new code against known-good output on identical inputs
*before* it was trusted for anything else. This caught real,
consequential bugs — not hypothetical ones. Specific examples:

- Replacing the dense N×N distance matrix with sparse on-demand
  distance computation was proven bit-identical at N=500-5,000 before
  it touched the real engine.
- The Numba-rewritten sparse state (`NumbaSparsePairState`) was
  checked against the original dict-based `SparsePairState` on a
  shared shadow-test (dense matrix as ground truth) before integration.
- A full engine rewrite (`ScaleLife`) was checked against the
  known-good engine on 5 seeds, exact match required, before any
  scale claims were trusted.

**When you change engine code, write or reuse an equivalence test
before believing the new numbers.** Don't skip this because a change
"looks obviously correct" — several bugs tonight looked obviously
correct until checked.

## Known gotchas — bugs that cost real time, now documented so they don't recur

1. **Rewire trigger conditions must be checked for accidental
   always-true logic.** `(t // 40) % 1 == 0` is *always* true (mod 1
   is always 0) — this silently made a rewire step fire on 62% of
   steps instead of once per 40, and elsewhere, nesting the rewire
   block inside a reward `if` block made its trigger condition
   mathematically unreachable. Always sanity-check trigger frequency
   empirically (print a counter), don't just read the condition and
   assume it's right.

2. **Decay rates must match the life length.** `SparsePairState`
   values decay as `rate^steps_elapsed`. A rate of 0.999 is fine for
   an ~800-1,200 step synthetic life but decays to ~0 over an
   ~8,600-step real-data life (0.999^8600 ≈ 0.0002). If a result comes
   back suspiciously exactly zero, check decay-rate-vs-life-length
   arithmetic before assuming the physics is broken.

3. **Reward must be checked frequently relative to the eligibility
   trace's decay rate**, not just once at the end of a long
   stimulus window. The eligibility trace (typically decay=0.90)
   effectively vanishes after ~50-100 steps; checking reward only
   once after a 200-step presentation means the trace has already
   decayed to numerical noise by the time it's used.

4. **When comparing two placement conditions, verify they have
   comparable raw adjacency/opportunity counts before trusting the
   ratio.** An early real-data placement comparison put segregated
   clusters far enough apart that they had 57x fewer adjacent
   cross-type pairs than the interleaved condition — the resulting
   "1,448x" effect was mostly a rigged setup, not a real finding. Fixed
   by placing both conditions in the same disc, same density, differing
   only in whether types are arranged as contiguous wedges (segregated)
   or shuffled (interleaved).

5. **When editing a running script's control flow (e.g., moving a
   rewire block), verify the resulting indentation actually nests
   where intended** — `ast.parse()` catches syntax errors but not
   logical nesting mistakes. Print/count how often a block actually
   executes rather than assuming an edit landed correctly.

6. **If a new file's behavior differs from a known-working reference
   on identical inputs, A/B them directly** — call the same function
   from both contexts on the same data — rather than theorizing about
   what might differ. This isolated harness-vs-physics bugs faster
   than any other technique used tonight.

## Current state (as of this handoff)

- **Provisional patent filed**: USPTO #64/134,008, filed 08/14/2026.
  Four claim areas: factory-dreams commissioning, value-gated
  structural plasticity, spatial consensus anomaly detection,
  interleaved placement for plastic hardware. 12-month window to
  decide on a full utility filing.
- **Preprint**: drafted (`topoforge_preprint.md`), not yet on arXiv.
  arXiv submission needs an endorser in cs.NE — no institutional
  affiliation, so the plan is to ask a cited author (e.g., Anup Das,
  SpiNeMap) once the paper is in strong shape.
- **Henry (Catalyst Neuromorphic)**: potential collaborator, contacted
  re: testing on his N4 hardware. His current chip doesn't have
  structural plasticity in the tapeout (cut before manufacture), so
  the collaboration is currently limited to an inference-only
  comparison he offered to run. Awaiting his results; no fixed
  timeline. A follow-up email was sent; no reply as of this handoff.
- **Most recent completed result**: Exp 37b — the placement effect
  confirmed on real SHD (Spiking Heidelberg Digits) speech data,
  9.31x interleaved-vs-segregated, with adjacency-matched geometry and
  a corrected decay rate. This closed the single biggest gap
  identified in a self-directed skeptical review (the "is this only a
  synthetic-pattern artifact" objection).
- **Open, honestly-unresolved thread**: an "inverted-U" finding in
  Exp 35's timing-jitter sweep (moderate jitter beats both perfect
  synchrony and high jitter) survived three separate mechanism
  hypotheses being tested and refuted. The phenomenon is solid and
  reproducible; the underlying cause is still open.
- **Scale engineering**: `topoforge-scale`'s Phase 1 (dense-matrix
  elimination) and Phase 2 (Numba compilation of the hot loops) are
  both complete and audited. A crossover benchmark shows the
  Numba-compiled engine overtakes the original around N~50,000-70,000
  — right where Phase 3 (the actual N=10K-500K scale ladder) would
  need it. Phase 3 has not yet been run.

## Reference documents worth reading early

- `README.md` (repo root) — the public-facing summary, benchmark
  numbers, quickstart.
- `topoforge_preprint.md` — the fuller scientific write-up, including
  a Limitations section written adversarially (a self-directed
  skeptical-reviewer pass, not just a formality).
- `neuromorphic_sandbox_report.md` and `_addendum.md` — earlier-stage
  research notes, useful for understanding how the core architecture
  (spatial cores, structural plasticity, RPE-driven value gating) was
  derived from first principles across the early experiments.

## What this project's author is like to work with

Self-taught, working nights/weekends around a full-time job. Values
being told directly when something is wrong or risky rather than
being cushioned — the collaboration works best when you push back on
weak claims and flag real problems plainly, the same way you'd flag
them to a colleague. Prefers concrete next steps over open-ended
options when a call needs to be made.
