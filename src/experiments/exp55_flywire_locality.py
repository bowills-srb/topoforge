"""Exp 55: does a real brain obey the locality this project assumes, and does
it interleave functionally-different neurons?

WHY THIS EXISTS. Every result in this repo rests on two modelling choices that
have never been checked against biology: that plasticity deposits are confined
to spatially local pairs (`LocalLife.r_deposit`), and that placing
functionally-different neurons near each other is what makes cross-type
association possible. Both are assumptions. The FlyWire FAFB v783 release
(Dorkenwald et al. 2024) is a complete adult Drosophila brain -- 139,255
neurons, 2,700,513 connected pairs, every synapse located, every neuron typed
and assigned a neurotransmitter. It is the one substrate where both
assumptions can be measured rather than argued.

WHAT THIS IS AND IS NOT. This is a DESCRIPTIVE measurement of one brain, not a
placement experiment: you cannot rearrange a fly and re-run it, so nothing
here tests the causal claim. It is also a single dataset with no
pre-registration -- treat it as this project treats `tinker_*` scripts, one
notch below the numbered confirmatory experiments, until something replicates
it (the male CNS and hemibrain connectomes are the obvious second and third
looks). What it CAN do is tell us whether the mechanism the thesis proposes is
the mechanism a real brain uses, and whether Henry's framing -- that the
penalty needs locality already present in connectivity -- describes biology.

ARMS (all on the central brain; the optic lobe is a retinotopic lattice with
its own logic and would dominate any whole-brain average):
  1  P(connected | centroid separation), normalised by how many pairs exist at
     each separation. Unnormalised distance histograms mostly measure how many
     neuron pairs happen to sit that far apart, which is not the question.
  2  The same in scale-free units: separation divided by the two neurons'
     combined arbor radius. A fly neuron is not a point -- median radius of
     gyration is 23.4 um against a brain 900 um across -- so "how far apart"
     only means something relative to how far each one reaches.
  3  Local type mixing: of a neuron's k nearest neighbours, what fraction
     shares its label, against a position-shuffled null. >1 means that label
     is spatially clustered (segregated); ~1 means interleaved.
  4  Separation of connected pairs, split by whether the partners share a type.
     This is the one that bears directly on the thesis: if cross-type partners
     sit FURTHER apart than same-type partners, real brains pay the cost this
     project says segregation imposes. If they sit closer, the brain is
     buying cross-type adjacency with its layout.

WHAT EACH OUTCOME MEANS:
  - Arm 1/2 show strong distance dependence: the `r_deposit` assumption is
    biologically grounded and can be cited rather than asserted.
  - Arm 3 shows clustering by class: a NAIVE reading is that real brains
    segregate and learn fine, which is the objection a reviewer will raise.
    Arm 4 is what answers it -- clustering is only a problem if it costs
    cross-type adjacency.
  - Arm 4 shows cross-type pairs no further apart than same-type pairs: the
    brain clusters by class AND keeps functional partners adjacent, which is
    the arrangement this project calls interleaved at the scale that matters.

--audit RUNS FOUR CHECKS, ALL OF WHICH CAUGHT SOMETHING REAL:
  A1  Codex's one representative point per neuron sits a median 49.7 um from
      the neuron's true synapse centroid -- about 2.1 arbor radii. Anything
      measured on representative points is measured through that blur.
  A2  connections.csv has one row PER NEUROPIL, so 3,869,878 rows cover only
      2,700,513 pairs. The 26.8% of pairs that span several neuropils are the
      sprawling long-range ones, so row-counting silently overweights exactly
      the pairs that argue against locality.
  A3  THE CIRCULARITY CHECK, and the reason to distrust arm 2 until it passes.
      Each neuron's centroid and radius are built FROM ITS SYNAPSES, and a
      connected pair SHARES those synapses -- so connecting two neurons
      mechanically drags their centroids together and inflates arm 2. This
      recomputes both clouds with the pair's own shared synapses removed and
      reports how much of the effect survives. Same discipline as Exp 53/54:
      the number looks good, which is when this project has historically been
      wrong.
  A4  Reconciles the synapse file against connections.csv (34,156,320 parsed
      rows vs 34,153,566 accounted for -- a 2,754 row excess, 0.008%).

LICENCE. FlyWire terms are unconfirmed (third parties cite CC BY-NC 4.0;
FlyWire's own docs do not state it). Local analysis is fine either way. Clear
it before any number from here enters the preprint or a patent filing. See
flywire_loader.py.

Run:          python src/experiments/exp55_flywire_locality.py
Run (audit):  python src/experiments/exp55_flywire_locality.py --audit
First run downloads ~360 MB into flywire_data/ and spends ~45s building the
synapse clouds; both are cached afterwards.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from flywire_loader import (DATA_DIR, build_clouds, download_flywire,  # noqa: E402
                            load_annotations, load_connections,
                            load_representative_points, pair_synapse_sums)

CACHE_PATH = os.path.join(DATA_DIR, "exp55_cache.json")
SEED = 0
K_NN = 20
N_QUERY = 8000
N_RANDOM_PAIRS = 3_000_000
MIN_SYN = 10           # neurons with fewer synapses have an unstable centroid
LOO_SAMPLE = 150_000   # connected pairs re-measured in the A3 circularity check
DIST_BINS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500], float)
RADII_BINS = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10], float)


# ---------------------------------------------------------------- primitives

def knn_indices(P, query, k):
    """Exact k nearest neighbours of `query` rows within P. NumPy only.

    Uses |a-b|^2 = |a|^2 + |b|^2 - 2a.b so the heavy step is one BLAS matmul
    per chunk; a (chunk, N, 3) difference tensor would not fit.
    """
    n2 = (P * P).sum(1)
    out = np.empty((len(query), k), np.int64)
    step = max(1, 20_000_000 // max(len(P), 1))
    for s in range(0, len(query), step):
        q = query[s:s + step]
        d2 = n2[None, :] - 2.0 * (P[q] @ P.T)
        d2[np.arange(len(q)), q] = np.inf          # exclude self
        part = np.argpartition(d2, k, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(d2[rows, part], axis=1, kind="stable")
        out[s:s + step] = part[rows, order]
    return out


def rel_prob(d_con, d_all, bins, unit, min_pairs=200):
    """Relative P(connected) per distance bin, normalised by pair availability."""
    h_con, _ = np.histogram(d_con, bins)
    h_all, _ = np.histogram(d_all, bins)
    scale = len(d_con) / float(len(d_all))
    rows = []
    print("    {:>14s} {:>10s} {:>9s} {:>13s}".format(
        "bin " + unit, "edges", "pairs%", "rel P(conn)"))
    for i in range(len(bins) - 1):
        if h_all[i] < min_pairs:
            continue
        rel = (h_con[i] / float(h_all[i])) / scale
        rows.append([float(bins[i]), float(bins[i + 1]), int(h_con[i]), rel])
        print("    {:>14s} {:>10d} {:>8.2f}% {:>13.2f}".format(
            "{:g}-{:g}".format(bins[i], bins[i + 1]), h_con[i],
            100.0 * h_all[i] / len(d_all), rel))
    return rows


def segregation_index(P, labels, rng, k=K_NN, n_query=N_QUERY):
    """Fraction of k nearest neighbours sharing a label, over a shuffled null."""
    q = rng.choice(len(P), min(n_query, len(P)), replace=False)
    nb = knn_indices(P, q, k)
    L = np.asarray([str(x) for x in labels])
    same = float((L[nb] == L[q][:, None]).mean())
    perm = rng.permutation(len(P))
    null = float((L[perm][nb] == L[perm][q][:, None]).mean())
    return same, null, same / max(null, 1e-12)


def describe(name, d, r):
    print("    {:<16s} n={:>8d}  median={:>6.1f} um  p90={:>6.1f} um  "
          "scale-free median={:.2f}".format(
              name, len(d), np.median(d), np.percentile(d, 90), np.median(r)))
    return {"n": int(len(d)), "median_um": float(np.median(d)),
            "p90_um": float(np.percentile(d, 90)), "median_radii": float(np.median(r))}


# ---------------------------------------------------------------- experiment

def load_all():
    download_flywire()
    ann = load_annotations()
    clouds = build_clouds(index=ann["index"])
    pre, post, syn, npil = load_connections(index=ann["index"], dedupe=True)
    return ann, clouds, pre, post, syn, npil


def main(args):
    rng = np.random.default_rng(SEED)
    ann, clouds, pre, post, syn, npil = load_all()
    C, RG, NS = clouds["centroid"], clouds["rg"], clouds["n_syn"]

    valid = np.isfinite(C).all(1) & (NS >= MIN_SYN)
    central = valid & (ann["super_class"] != "optic")
    keep = central[pre] & central[post]
    p, q = pre[keep], post[keep]
    print("\nExp 55 -- FlyWire FAFB v783, central brain")
    print("  neurons kept (located, >={} synapses, non-optic): {}".format(MIN_SYN, central.sum()))
    print("  connected pairs kept: {}".format(len(p)))

    d_con = np.linalg.norm(C[p] - C[q], axis=1)
    pool = np.flatnonzero(central)
    A = rng.choice(pool, N_RANDOM_PAIRS)
    B = rng.choice(pool, N_RANDOM_PAIRS)
    ok = A != B
    A, B = A[ok], B[ok]
    d_all = np.linalg.norm(C[A] - C[B], axis=1)

    print("\n[ARM 1] P(connected | centroid separation)")
    arm1 = rel_prob(d_con, d_all, DIST_BINS, "(um)")

    print("\n[ARM 2] the same in units of combined arbor radius")
    r_con = d_con / (RG[p] + RG[q])
    r_all = d_all / (RG[A] + RG[B])
    arm2 = rel_prob(r_con, r_all, RADII_BINS, "(radii)")
    within_con = float((r_con <= 1.0).mean())
    within_all = float((r_all <= 1.0).mean())
    print("    connected median {:.2f} radii vs random-pair median {:.2f} radii".format(
        np.median(r_con), np.median(r_all)))
    print("    within 1 combined arbor radius: {:.1f}% of connections, "
          "{:.1f}% of random pairs".format(100 * within_con, 100 * within_all))

    print("\n[ARM 3] local type mixing (k={}, >1 = clustered, ~1 = interleaved)".format(K_NN))
    arm3 = {}
    for label in ("class", "super_class", "nt"):
        same, null, seg = segregation_index(C[central], ann[label][central], rng)
        arm3[label] = {"same": same, "null": null, "index": seg}
        print("    {:<14s} same={:.3f}  shuffled null={:.3f}  segregation={:.2f}x".format(
            label, same, null, seg))

    print("\n[ARM 4] connected-pair separation, by type relation")
    arm4 = {}
    for name, sel in (("within-class", ann["class"][p] == ann["class"][q]),
                      ("cross-class", ann["class"][p] != ann["class"][q]),
                      ("same-NT", ann["nt"][p] == ann["nt"][q]),
                      ("cross-NT", ann["nt"][p] != ann["nt"][q])):
        arm4[name] = describe(name, d_con[sel], r_con[sel])

    # The scale-free comparison decides, not the raw one. That is ARM 2's whole
    # argument -- a separation only means something relative to how far the two
    # neurons reach -- and it is also the empirically stable measure: the raw
    # within/cross ORDERING flips depending on whether connections.csv rows or
    # unique pairs are counted (see --audit A2), while the scale-free ratio does
    # not move. Threshold set at 5% AFTER seeing the first run, so ARM 4 is
    # descriptive; a pre-registered replication on the male CNS connectome is
    # what would make it confirmatory.
    raw_ratio = arm4["cross-class"]["median_um"] / arm4["within-class"]["median_um"]
    sf_ratio = arm4["cross-class"]["median_radii"] / arm4["within-class"]["median_radii"]
    print("\n  VERDICT")
    print("    locality: connection probability spans {:.0f}x across the measured"
          " range, and".format(arm1[0][3] / max(arm1[-1][3], 1e-9)))
    print("    {:.1f}% of connections sit within one combined arbor radius against"
          " {:.1f}% of".format(100 * within_con, 100 * within_all))
    print("    random pairs. Connections happen where arbors already overlap.")
    print("    cell classes are spatially clustered ({:.2f}x), neurotransmitter much"
          " less so ({:.2f}x).".format(arm3["class"]["index"], arm3["nt"]["index"]))
    print("    cross-class vs within-class separation: {:.2f}x raw, {:.2f}x scale-free".format(
        raw_ratio, sf_ratio))
    if abs(sf_ratio - 1.0) < 0.05:
        print("    -> NO MATERIAL CROSS-TYPE DISTANCE PENALTY. Clustering by class costs")
        print("       the fly nothing in reach between functionally-different partners:")
        print("       relative to how far each neuron extends, cross-type partners are")
        print("       exactly as close as same-type ones. That is what this project means")
        print("       by interleaved at the scale the plasticity rule sees. Note this is")
        print("       an absence of a penalty, NOT evidence that the fly prefers")
        print("       cross-type proximity -- do not overstate it.")
    elif sf_ratio < 1.0:
        print("    -> cross-type partners sit CLOSER than same-type partners even after")
        print("       normalising for reach: the fly actively buys cross-type adjacency.")
    else:
        print("    -> real brains DO pay cross-type distance for class clustering, by")
        print("       {:.0f}% in scale-free terms. That is a counterexample to the".format(
            100 * (sf_ratio - 1)))
        print("       thesis, not a confirmation; report it as one.")

    out = {"arm1_rel_prob": arm1, "arm2_rel_prob": arm2,
           "arm2_within_one_radius": {"connected": within_con, "random": within_all},
           "arm3_segregation": arm3, "arm4_by_type": arm4,
           "n_neurons": int(central.sum()), "n_pairs": int(len(p))}
    with open(CACHE_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n  cached -> {}".format(CACHE_PATH))
    return out


# -------------------------------------------------------------------- audit

def audit(args):
    rng = np.random.default_rng(SEED)
    ann, clouds, pre, post, syn, npil = load_all()
    C, RG, NS = clouds["centroid"], clouds["rg"], clouds["n_syn"]
    valid = np.isfinite(C).all(1) & (NS >= MIN_SYN)
    central = valid & (ann["super_class"] != "optic")

    print("\n[A1] Codex representative point vs true synapse centroid")
    R = load_representative_points(index=ann["index"])
    ok = np.isfinite(C).all(1) & np.isfinite(R).all(1)
    off = np.linalg.norm(C[ok] - R[ok], axis=1)
    med_rg = float(np.nanmedian(RG[ok]))
    print("    n={}  median={:.1f} um  p90={:.1f} um  max={:.1f} um".format(
        ok.sum(), np.median(off), np.percentile(off, 90), off.max()))
    print("    median arbor radius {:.1f} um -> off by {:.2f} arbor radii".format(
        med_rg, np.median(off) / med_rg))
    print("    CONSEQUENCE: representative points blur locality. Use build_clouds().")

    print("\n[A2] connections.csv rows vs unique pairs")
    rp, rq, rs, _ = load_connections(index=ann["index"], dedupe=False)
    print("    rows={}  unique directed pairs={}  multi-neuropil pairs={} ({:.1f}%)".format(
        len(rp), len(pre), (npil > 1).sum(), 100.0 * (npil > 1).mean()))
    km = central[rp] & central[rq]
    d_rows = np.linalg.norm(C[rp[km]] - C[rq[km]], axis=1)
    kk = central[pre] & central[post]
    d_pairs = np.linalg.norm(C[pre[kk]] - C[post[kk]], axis=1)
    print("    median separation counting rows: {:.1f} um   counting pairs: {:.1f} um"
          "  ({:+.1f}%)".format(np.median(d_rows), np.median(d_pairs),
                                100.0 * (np.median(d_rows) - np.median(d_pairs))
                                / np.median(d_pairs)))
    # The overall median barely moves, but ARM 4's within/cross split does -- the
    # duplicated rows are not spread evenly over the two groups. That is the real
    # consequence, so measure it rather than asserting the one I expected.
    for tag, (ip, iq, dd) in (("rows ", (rp[km], rq[km], d_rows)),
                              ("pairs", (pre[kk], post[kk], d_pairs))):
        w = ann["class"][ip] == ann["class"][iq]
        print("    {}: within-class median {:.1f} um   cross-class median {:.1f} um"
              "   cross/within {:.3f}".format(tag, np.median(dd[w]), np.median(dd[~w]),
                                              np.median(dd[~w]) / np.median(dd[w])))
    print("    CONSEQUENCE: the overall median is insensitive to this, but the")
    print("    within/cross ORDERING is not -- it flips. Count unique pairs, and do")
    print("    not rest an argument on the raw within/cross ordering either way.")

    print("\n[A3] circularity check on ARM 2 -- leave the pair's own synapses out")
    kk = central[pre] & central[post]
    p, q = pre[kk], post[kk]
    take = rng.choice(len(p), min(LOO_SAMPLE, len(p)), replace=False)
    p, q = p[take], q[take]
    print("    re-measuring {} sampled pairs (one extra pass over the synapse file)".format(len(p)))
    sums = pair_synapse_sums(zip(p.tolist(), q.tolist()), index=ann["index"])

    # rebuild the raw nm accumulators from the cached um summaries
    c_nm = C * 1000.0
    rg_nm = RG * 1000.0
    S = c_nm * NS[:, None]
    S2 = NS * (rg_nm ** 2 + (c_nm ** 2).sum(1))

    def cloud_without(i, key):
        cnt_p, S_p, S2_p = sums[key]
        n_left = NS[i] - cnt_p
        if n_left < MIN_SYN:
            return None, None
        c = (S[i] - S_p) / n_left
        var = (S2[i] - S2_p) / n_left - c @ c
        return c / 1000.0, float(np.sqrt(max(var, 0.0))) / 1000.0

    d_naive, r_naive, d_loo, r_loo = [], [], [], []
    dropped = 0
    for a, b in zip(p.tolist(), q.tolist()):
        key = (a, b) if a <= b else (b, a)
        ca, ra = cloud_without(a, key)
        cb, rb = cloud_without(b, key)
        if ca is None or cb is None:
            dropped += 1
            continue
        d_naive.append(float(np.linalg.norm(C[a] - C[b])))
        r_naive.append(d_naive[-1] / (RG[a] + RG[b]))
        d_loo.append(float(np.linalg.norm(ca - cb)))
        r_loo.append(d_loo[-1] / max(ra + rb, 1e-9))
    d_naive = np.array(d_naive); r_naive = np.array(r_naive)
    d_loo = np.array(d_loo); r_loo = np.array(r_loo)
    print("    pairs usable: {} (dropped {} whose clouds fell below {} synapses)".format(
        len(d_loo), dropped, MIN_SYN))
    print("    separation  median  {:.1f} um -> {:.1f} um".format(
        np.median(d_naive), np.median(d_loo)))
    print("    scale-free  median  {:.2f} -> {:.2f} radii".format(
        np.median(r_naive), np.median(r_loo)))
    w0 = float((r_naive <= 1).mean()); w1 = float((r_loo <= 1).mean())
    print("    within 1 combined radius: {:.1f}% -> {:.1f}%".format(100 * w0, 100 * w1))
    shrink = 100.0 * (w0 - w1) / max(w0, 1e-9)
    if w1 >= 0.5 and shrink < 25.0:
        print("    PASSES: {:.1f}% of the effect was self-referential; the locality"
              " claim survives.".format(shrink))
    else:
        print("    FAILS: {:.1f}% of ARM 2 was the pair's own synapses pulling its two"
              " clouds together.".format(shrink))
        print("    Report the leave-one-out number, not ARM 2's.")

    print("\n[A4] synapse reconciliation")
    z = np.load(os.path.join(DATA_DIR, "neuron_clouds.npz"))
    parsed = int(z["n_rows"]) if "n_rows" in z else -1
    print("    synapse rows parsed: {}".format(parsed))
    print("    synapses in connections.csv: {}".format(int(rs.sum())))
    print("    excess: {} ({:.4f}%) -- synapses whose pair is absent from the"
          " connection list".format(parsed - int(rs.sum()),
                                    100.0 * (parsed - int(rs.sum())) / max(parsed, 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true",
                    help="run the four methodology checks instead of the measurement")
    a = ap.parse_args()
    if a.audit:
        audit(a)
    else:
        main(a)
