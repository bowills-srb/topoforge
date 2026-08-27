# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Also read `PROJECT_HISTORY.md`** — it holds context that only exists in the project's history (working conventions, the audit-first discipline, a catalog of bugs that cost real time, and current patent/collaboration status). This file covers structure and commands; that file covers judgment. The most load-bearing points from it are summarized below, but read the original before making changes.

## What this is

A research sandbox (project name "TopoForge") investigating one thesis: **on plastic neuromorphic hardware, where you place neurons determines what the network can learn**, not just communication energy. Placing functionally-different neurons close together ("interleaved") lets structural plasticity build cross-type associations that function-grouped block placement ("segregated") makes much harder or impossible. Existing chip-mapping tools optimize only for energy; the headline result is that segregated placement causes a ~4x learning penalty vs interleaved at zero inference-energy cost — now confirmed at multiple scales, with geometry removed entirely (pure graph locality), and on real spike-encoded speech data.

The repo is organized as a lab notebook, not a product. Each numbered experiment in `src/experiments/` is a self-contained, independently runnable script that tests or refutes one hypothesis. **Git history is the primary record** — commit messages are experiment writeups (what was tested, what was found, what was refuted). Read `git log` before making claims about what is or isn't established; findings are deliberately hedged and several hypotheses have been cleanly refuted.

## Two repositories — know which one you're in

- **`topoforge`** (this repo, public) — the main research repo backing a filed provisional patent and a potential collaborator's testing. **It needs to stay stable and citable.** Don't break working experiments; if you change shared engine code, re-run the affected experiments to confirm results still hold.
- **`topoforge-scale`** (private, separate) — a from-scratch rewrite targeting N=500,000 with a `ScaleLife` engine and Numba kernels. Allowed to be messy; only merges back after full equivalence-audit passage.

## Running things

Everything is plain NumPy (plus `h5py` for the real-data loader). There is no build step and no `requirements.txt`; a `venv/` is checked in. Run from the **repo root** — scripts hardcode `sys.path.insert(0, "src")` and will not resolve imports from other working directories.

```powershell
# The CLI: describe hardware in JSON, get a placement recommendation (~90s)
python src/topoforge_run.py                       # runs default_config()
python src/topoforge_run.py src/example_config.json
python src/topoforge_run.py config.json --custom my_placement.csv   # CSV cols: neuron_id,x,y,cluster_identity

# The headline benchmark (PLB), 10 seeds, full stats
python src/experiments/exp32b_benchmark.py

# Any experiment — each prints its own "Run:" line in its module docstring
python src/experiments/exp37b_v2_real_data.py
```

### Tests / verification

There is no pytest suite. Verification is built into the modules themselves:

- **Core primitives self-test** when run directly: `python src/sparse_state.py`, `python src/spatial.py`, `python src/engine.py`, `python src/engine_fast.py`. These shadow-test the sparse/fast implementations against dense reference matrices and should be run after touching those files.
- **`src/audit.py`** — dense reference harness. Seed-sweeps the five load-bearing claims against pre-registered criteria, printing a `GRANITE` (passed) or `SOFT` (failed) verdict per claim.
- **`src/audit_engine.py`** — the same five claims re-run on the sparse engine as a regression gate (legacy mode). Long-running (hours).

**Audit before you trust — this is the project's single most important convention.** Every non-trivial change here has been validated against known-good output on identical inputs *before* being trusted. This has caught real, consequential bugs, not hypothetical ones (bit-identical sparse-vs-dense distance, shadow-tested Numba state, exact 5-seed engine-rewrite match). When you change engine or state code, write or reuse an equivalence test before believing the new numbers — even if the change "looks obviously correct." The bar for a change: primitive self-tests still pass, and `audit.py` claims stay GRANITE.

## Architecture

The stack is three layers. The trick that makes everything tractable is at the bottom.

**1. Spatial + sparse-state primitives (`src/spatial.py`, `src/sparse_state.py`)**
- `SpatialGrid` — uniform grid bucketing for *exact* k-NN and radius queries (the grid only prunes candidates; distances decide). Rebuilt each epoch after neurons migrate.
- `SparsePairState` — decaying scalars on a dynamic set of `(i, j)` neuron pairs with **lazy exponential decay**: `value(t) = stored * decay^(t - last_touched)`, and `tick()` is O(1). This is what makes N=100K+ tractable instead of storing dense NxN matrices. Three instances are used throughout with different decay constants: **C** = co-firing correlation (~0.95), **E** = eligibility trace (~0.90), **V** = value / reward-attributed structure (~0.999+).

**2. Simulation engine (`src/engine.py` → `engine_local.py` → `engine_fast.py`)**
- `Life` (in `engine.py`) is the base simulation: LIF neurons (leak, threshold, refractory), a fixed input→pattern schedule, structural plasticity (periodically prune "cold" edges and rewire toward high-value candidate pairs), and RPE (reward-prediction-error) learning. Retention rule is pluggable via `rule='corr'|'util'|'rpe'`.
- `LocalLife` restricts plasticity deposits to spatially local pairs within `r_deposit` — this is the mechanism the whole thesis rests on (local rules can't associate neurons placed far apart).
- `FastLife` overrides only `rewire()` with a vectorized array scan; certified to match `LocalLife` exactly.
- `r_grow=None` reproduces legacy global-candidate behavior; `r_grow=<float>` is the honest local-physics mode. Watch for which one an experiment uses.

**3. Experiments & the CLI (`src/experiments/`, `src/topoforge_run.py`)**
- Experiments are numbered roughly chronologically (exp20–exp37b). Many **re-implement the sim loop inline** rather than importing `Life`, so they can vary the physics — do not assume an experiment shares code with the engine; read its top-of-file docstring, which states the hypothesis and the exact run command.
- The three placement strategies (`segregated`, `interleaved`, `random`) are generated in both `topoforge_run.py` and the benchmark experiments. Segregated = each neuron type in a contiguous region; interleaved = types mixed within each core/disc; random = global shuffle.

**Real data (`shd_loader.py`, `shd_data/`)** — `load_shd_samples()` downloads the Spiking Heidelberg Digits dataset directly from Zenke Lab (deliberately avoids `tonic`/`expelliarmus`, which need a C compiler on Windows; only `h5py` is required). Files cache in `shd_data/`. Used by the `exp37*` real-data experiments.

## Gotchas specific to this codebase

These are bugs that have already cost real time. See `PROJECT_HISTORY.md` for the full list with examples.

- **Lazy decay + long runs is a real trap.** Because `SparsePairState` decays continuously as `rate^steps_elapsed`, structure deposited early in a long simulation can decay below measurability before readout: V decay 0.999 over ~8,600 real-data steps ≈ 0.0002. This has produced spurious exact-"zero" results. Keep decay rate and total run length matched; when a result is unexpectedly zero, suspect decay-vs-life-length arithmetic before the physics.
- **Reward must be checked frequently relative to the eligibility trace's decay.** The E trace (decay ~0.90) vanishes after ~50–100 steps; checking reward only once after a 200-step presentation reads numerical noise.
- **Placement comparisons must be adjacency-matched** to be meaningful. Early segregated-vs-interleaved designs gave segregated ~57x fewer adjacent cross-type pairs, so the ratio reflected opportunity count, not arrangement (a bogus "1,448x"). Newer experiments equalize adjacency by placing both conditions in the same disc/density, differing only in contiguous-wedge vs shuffled type assignment.
- **Sanity-check rewire trigger frequency empirically** (print a counter). `(t // 40) % 1 == 0` is always true; nesting a rewire block inside a reward `if` can make its trigger mathematically unreachable. `ast.parse()` catches syntax errors, not logical nesting mistakes.
- `diagnose_exp37*.py` and `diff_check.py` in the repo root are debugging scripts, not experiments — they A/B an experiment's own `run_life()` against a known-good diagnostic loop to isolate harness-vs-physics bugs.
- `archive/` holds superseded Brian2 prototype notebooks; the project moved off Brian2 to pure NumPy (`src/simulation.py`). Don't mine it for current behavior.

## Current project status

- **Provisional patent filed**: USPTO #64/134,008 (08/14/2026). Four claim areas: factory-dreams commissioning, value-gated structural plasticity, spatial consensus anomaly detection, interleaved placement for plastic hardware. 12-month window to decide on a full utility filing.
- **Preprint**: drafted (`topoforge_preprint.md`), not yet on arXiv — arXiv cs.NE needs an endorser and there's no institutional affiliation; plan is to ask a cited author once the paper is strong. (Note: `topoforge_preprint.md` and `neuromorphic_sandbox_addendum.md` are referenced across the docs but are not present in this working tree.)
- **Henry (Catalyst Neuromorphic)** — potential collaborator contacted re: testing on his N4 hardware. His current chip's tapeout dropped structural plasticity, so collaboration is limited to an inference-only comparison he offered to run. Awaiting results; a follow-up email was sent with no reply as of the last handoff. No fixed timeline.
- **Most recent completed result**: Exp 37b — placement effect confirmed on real SHD speech data, 9.31x interleaved-vs-segregated, adjacency-matched, corrected decay. Closed the biggest gap from a self-directed skeptical review (the "synthetic-artifact" objection).
- **Open, unresolved thread**: Exp 35's timing-jitter "inverted-U" (moderate jitter beats both perfect synchrony and high jitter) survived three refuted mechanism hypotheses. Phenomenon is solid and reproducible; cause is still open.
- **Scale engineering** (`topoforge-scale`): Phase 1 (dense-matrix elimination) and Phase 2 (Numba compilation) complete and audited; crossover benchmark shows Numba overtakes the original around N~50K–70K. Phase 3 (the N=10K–500K scale ladder) not yet run.

## Working with this project's author

Self-taught, working nights/weekends around a full-time job. Values being told directly when something is wrong or risky rather than cushioned — push back on weak claims and flag real problems plainly. Prefers concrete next steps over open-ended options when a call needs to be made.
