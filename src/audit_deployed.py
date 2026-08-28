"""The regression gate for the code the PUBLISHED RESULTS actually run.

Why this file exists. An audit sweep (2026-08-27) mapped which code each
audit covers against which code the preprint's numbers flow through, and
found the safety net was over the wrong part of the tree:

  audit.py          imports NOTHING from this project -- it is a self-contained
                    dense reimplementation. It is valuable as an INDEPENDENT
                    replication of the findings, but it cannot regress-test any
                    change to this codebase: it would print GRANITE with the
                    entire engine deleted.
  audit_engine.py   gates engine.Life. No preprint-backing experiment imports
                    the engine at all (verified). It costs hours and guards
                    code no published number depends on.
  the real path     every preprint experiment depends on exactly two project
                    primitives -- SparsePairState and SpatialGrid -- plus its
                    own inline physics loop (exp32b.run_life and friends).
                    Nothing gated that path until this file.

So this gate covers the load-bearing path, and it runs in ~2 minutes rather
than hours, because a gate nobody runs is not a gate.

DESIGN NOTE -- most checks here are exact and environment-independent by
construction (structural properties, brute-force equivalence, a known optimum).
Learning values are NOT: see check [6]. Keep it that way. A gate whose failures
are usually false alarms gets ignored, which is how the exp38 bug survived.

Run: python src/audit_deployed.py        (either interpreter; see [1])
"""
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

FAILURES = []


def check(label, ok, detail=""):
    print("    [{}] {}{}".format("OK" if ok else "FAIL", label,
                                 ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


# ============================================================
# [1] Environment
# ============================================================
def env_banner():
    print("[1] environment")
    print("    python {}".format(sys.version.split()[0]))
    print("    numpy  {}".format(np.__version__))
    print("    NOTE: run_life's rewire step sorts a heavily tie-degenerate score")
    print("    array (V is mostly zero) with np.argsort's default UNSTABLE sort,")
    print("    so which cold edges get replaced depends on sort internals that")
    print("    changed between numpy 1.26 and 2.5. Measured on the PLB, seed 0:")
    print("      numpy 1.26.4 default : topoforge 2955  vlsi 682")
    print("      numpy 2.5.1  default : topoforge 2979  vlsi 724   (6.2% on vlsi)")
    print("      either, kind='stable': topoforge 2978  vlsi 702   (identical)")
    print("    Learning numbers are therefore environment-sensitive; structural")
    print("    checks below are not. See check [6].")


# ============================================================
# [2] SpatialGrid.within at the DEPLOYED configuration
# ============================================================
def check_spatial():
    """The self-test in spatial.py builds the grid with cell_size=2.0, but every
    experiment builds it with cell_size == the query radius (5.0), which makes
    ring = ceil(r/cell) = 1 -- the zero-margin boundary case. Correct, but it
    was the one configuration never tested. Brute-force it on real geometry."""
    from spatial import SpatialGrid
    from exp32b_benchmark import make_placement
    import exp43_coverage_confirmatory as E43
    print("\n[2] SpatialGrid.within exact at deployed config (cell == radius)")

    def brute(coords, i, r):
        d2 = ((coords - coords[i]) ** 2).sum(1)
        return set(np.where((d2 <= r * r) & (np.arange(len(coords)) != i))[0].tolist())

    cases = [("PLB " + s, make_placement(s)[0]) for s in ("topoforge", "vlsi", "random")]
    cases += [("exp43 " + n, E43.make_condition(n)[0]) for n in ("spread15", "clumped30")]
    # points sitting exactly on cell boundaries at exactly the query radius
    cases.append(("adversarial boundary", np.array(
        [[5.0, 5.0], [10.0, 5.0], [0.0, 5.0], [5.0, 10.0], [5.0, 0.0],
         [9.999999, 5.0], [10.000001, 5.0], [7.5, 7.5]])))
    for tag, coords in cases:
        bad = 0
        for cell in (5.0, 2.0):
            g = SpatialGrid(coords, cell)
            for i in range(len(coords)):
                if set(int(x) for x in g.within(i, 5.0)) != brute(coords, i, 5.0):
                    bad += 1
        check("{:<22} exact at cell=5.0 and 2.0".format(tag), bad == 0,
              "" if bad == 0 else "{} mismatched neurons".format(bad))


# ============================================================
# [3] SparsePairState under the DEPLOYED call pattern
# ============================================================
def check_sparse_state():
    """sparse_state.py's shadow test drives the class through deposit_outer(),
    which no experiment calls (they all loop .deposit() over radius neighbours),
    and prunes at the default eps=1e-12 where run_life prunes at 1e-6. Redo the
    shadow test the way the deployed path actually uses it, and cover the
    long-life decay regime that the real-data experiments run in."""
    from sparse_state import SparsePairState
    print("\n[3] SparsePairState vs dense, using run_life's call pattern")

    N, STEPS, R = 120, 300, 5.0

    def shadow(prune_eps):
        """Run dense and sparse side by side. prune_eps=None disables pruning,
        which is the condition under which they must agree EXACTLY."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(N, 2))
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        nbr = [np.where((d2[i] <= R * R) & (np.arange(N) != i))[0] for i in range(N)]
        C_d, E_d, V_d = (np.zeros((N, N)) for _ in range(3))
        C_s, E_s, V_s = (SparsePairState(k) for k in (0.95, 0.90, 0.999))
        for t in range(STEPS):
            f = rng.choice(N, size=int(rng.integers(3, 15)), replace=False)
            fs = set(int(x) for x in f)
            C_d *= 0.95; E_d *= 0.90; V_d *= 0.999
            C_s.tick(); E_s.tick(); V_s.tick()
            for i in f:                              # <- the deployed pattern
                i = int(i)
                for j in nbr[i]:
                    if int(j) in fs:
                        C_d[i, int(j)] += 1.0; E_d[i, int(j)] += 1.0
                        C_s.deposit(i, int(j), 1.0); E_s.deposit(i, int(j), 1.0)
            if t % 20 == 6:                          # RPE deposit, as in run_life
                delta = 0.37
                V_d += delta * E_d
                if prune_eps is not None:
                    E_s.prune_below(prune_eps)
                for key in list(E_s.store.keys()):
                    ev = E_s.get(*key)
                    if ev != 0:
                        V_s.deposit(key[0], key[1], delta * ev)
        err = 0.0
        for i in range(N):
            for j in range(N):
                if i != j:
                    err = max(err, abs(C_d[i, j] - C_s.get(i, j)),
                              abs(V_d[i, j] - V_s.get(i, j)))
        return err, C_s

    # (a) the primitive itself: with pruning off, sparse must be EXACT
    err_exact, C_s = shadow(None)
    check("sparse == dense exactly (no pruning), {} steps".format(STEPS),
          err_exact < 1e-8, "max |dense - sparse| = {:.2e}".format(err_exact))

    # (b) the deployed pattern prunes E at 1e-6, which DISCARDS eligibility the
    #     dense reference still credits. The resulting divergence is expected;
    #     what matters is that it stays at the scale the threshold implies
    #     (~delta * eps per prune) rather than growing into the signal.
    err_pruned, _ = shadow(1e-6)
    check("prune_below(1e-6) divergence stays negligible",
          err_exact < err_pruned < 1e-4,
          "max {:.2e} vs C values O(1) -- expected, not a defect".format(err_pruned))

    si, sj, sv = C_s.get_arrays()
    ga = max((abs(C_s.get(int(a), int(b)) - v) for a, b, v in zip(si, sj, sv)),
             default=0.0)
    check("get_arrays matches get() on all active pairs", ga < 1e-12,
          "max diff {:.2e}".format(ga))

    # the real-data regime: V decay 0.999 over an ~8,600-step life (gotcha #2)
    S = SparsePairState(0.999)
    S.deposit(1, 2, 1.0)
    S.tick(8600)
    want = 0.999 ** 8600
    check("lazy decay exact across an 8,600-step gap", abs(S.get(1, 2) - want) < 1e-15,
          "{:.3e} (this is why long-life results can read as zero)".format(want))


# ============================================================
# [4] SpiNeCluster -- the check that would have caught the exp38 bug
# ============================================================
def check_spinecluster():
    """The partitioner reached the optimal cut on type-SHUFFLED input, which is
    what --sanity-cluster tested, while stalling at the random-partition cut on
    the type-SORTED input make_placement_spinemap actually feeds it. Gate the
    deployed ordering, against a known optimum, with a ratio to random so a
    silent no-op cannot look plausible."""
    from exp32b_benchmark import NC, N, N_CORES, NEURONS_PER_CORE
    from exp38_spinemap_baseline import (
        build_synapse_graph, spinecluster, inter_cluster_cut)
    print("\n[4] SpiNeCluster attains the optimal cut on the DEPLOYED input")
    OPT = {"population": 74250.0, "functional": 139050.0}
    for mode in ("population", "functional"):
        cids = np.repeat(np.arange(NC), N // NC)        # sorted: the deployed path
        W = build_synapse_graph(cids, mode)
        part = spinecluster(W, NEURONS_PER_CORE, N_CORES, seed=0, restarts=2)
        cut = inter_cluster_cut(W, part)
        sizes = np.bincount(part, minlength=N_CORES)
        purity = np.mean([np.bincount(cids[part == c], minlength=NC).max()
                          / NEURONS_PER_CORE for c in range(N_CORES)])
        rc = []
        for s in range(20):
            rp = np.repeat(np.arange(N_CORES), NEURONS_PER_CORE)
            np.random.default_rng(s).shuffle(rp)
            rc.append(inter_cluster_cut(W, rp))
        check("{:<11} cut == theoretical optimum".format(mode),
              abs(cut - OPT[mode]) < 1e-9,
              "cut {:.0f} vs {:.0f}, KL/random {:.4f}".format(cut, OPT[mode],
                                                              cut / np.mean(rc)))
        check("{:<11} clusters balanced".format(mode),
              bool(np.all(sizes == NEURONS_PER_CORE)))
        if mode == "population":
            check("population clusters are type-pure", abs(purity - 1.0) < 1e-9,
                  "purity {:.3f} (0.38 was the stalled partitioner)".format(purity))


# ============================================================
# [5] Placement structure -- deterministic and environment-independent
# ============================================================
def check_placements():
    """Placements are pure geometry: no simulation, no unstable sort. These
    pinned values must hold in any environment."""
    from exp32b_benchmark import make_placement
    from exp38_spinemap_baseline import make_placement_spinemap
    from exp42_fabric_sparsity_crossover import plb_reach, placement_at
    import exp43_coverage_confirmatory as E43
    print("\n[5] placement structure (pinned; environment-independent)")

    for strat, want in (("topoforge", 7.80), ("vlsi", 0.00), ("random", 7.76)):
        c, ci = make_placement(strat)
        check("PLB {:<10} reach".format(strat), abs(plb_reach(c, ci) - want) < 0.01,
              "{:.2f} (want {:.2f})".format(plb_reach(c, ci), want))
    for mode, want in (("population", 6.34), ("functional", 15.13)):
        c, ci = make_placement_spinemap(mode, seed=0)
        check("SpiNeMap {:<10} reach".format(mode), abs(plb_reach(c, ci) - want) < 0.01,
              "{:.2f} (want {:.2f})".format(plb_reach(c, ci), want))
        c2, _ = make_placement_spinemap(mode, seed=0)
        check("SpiNeMap {:<10} deterministic".format(mode), np.array_equal(c, c2))

    # the Exp 42 crossover: population-graph SpiNeMap loses reach entirely
    c, ci = placement_at("spinemap-population", 7.5)
    check("SpiNeMap-pop reach collapses at rho=1.5", plb_reach(c, ci) < 0.02,
          "{:.3f}".format(plb_reach(c, ci)))

    # the Exp 43 design: composition must fix the mediators exactly
    for name, want in (("spread15", (6.00, 0.429, 1.000)),
                       ("dilute30", (6.00, 0.207, 1.000)),
                       ("clumped30", (6.25, 0.216, 0.417))):
        c, ci = E43.make_condition(name)
        m = E43.mediators(c, ci)
        got = (m["count"], m["frac"], m["cover"])
        check("exp43 {:<10} (count, frac, cover)".format(name),
              all(abs(g - w) < 0.005 for g, w in zip(got, want)),
              "({:.2f}, {:.3f}, {:.3f})".format(*got))


# ============================================================
# [6] Learning values -- tolerance band, NOT exact equality
# ============================================================
def check_learning(tol=0.10):
    """These are the only environment-sensitive checks here. The band is wide
    on purpose: the measured cross-version spread is 6.2% on the segregated
    condition. A failure means the physics moved by more than sort-order noise
    can explain; it does NOT mean the numbers are irreproducible in principle
    (kind='stable' makes them identical across versions -- see [1])."""
    from exp32b_benchmark import make_placement, run_life
    print("\n[6] run_life learning, seed 0, +/-{:.0%} band (environment-sensitive)"
          .format(tol))
    for strat, ref in (("topoforge", 2967.0), ("vlsi", 703.0), ("random", 2745.0)):
        t0 = time.time()
        c, ci = make_placement(strat)
        got = run_life(c, ci, 0)["plastic"]["taught"]
        check("{:<10} taught within band of {:.0f}".format(strat, ref),
              abs(got - ref) / ref < tol,
              "{} ({:+.1%}, {:.0f}s)".format(got, (got - ref) / ref, time.time() - t0))


if __name__ == "__main__":
    print("=" * 74)
    print("DEPLOYED-PATH REGRESSION GATE")
    print("=" * 74)
    t0 = time.time()
    env_banner()
    check_spatial()
    check_sparse_state()
    check_spinecluster()
    check_placements()
    if "--fast" not in sys.argv:
        check_learning()
    print("\n" + "=" * 74)
    if FAILURES:
        print("GATE FAILED ({} checks) in {:.0f}s:".format(len(FAILURES), time.time() - t0))
        for f in FAILURES:
            print("    - {}".format(f))
    else:
        print("GATE PASSED in {:.0f}s".format(time.time() - t0))
    print("=" * 74)
    sys.exit(1 if FAILURES else 0)
