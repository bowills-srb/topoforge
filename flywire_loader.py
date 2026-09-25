"""flywire_loader.py -- the real thing: a complete adult brain's wiring.

Downloads the FlyWire FAFB v783 connectome (Dorkenwald et al. 2024) directly
from the public Codex storage bucket. Same spirit as `shd_loader.py`: no
third-party connectome package (`fafbseg`, `navis`, `caveclient` all pull
heavy dependency trees), just urllib + gzip + numpy, which this repo already
has. Nothing here needs a compiler and nothing needs a login.

Source:  https://storage.googleapis.com/flywire-data/codex/data/fafb/783/
Portal:  https://codex.flywire.ai
Cite:    Dorkenwald et al., "Neuronal wiring diagram of an adult brain",
         Nature 634, 124-138 (2024);
         Schlegel et al., "Whole-brain annotation and multi-connectome cell
         typing of Drosophila", Nature 634, 139-152 (2024).

LICENCE -- READ BEFORE PUBLISHING. Third parties cite CC BY-NC 4.0
(non-commercial) for FlyWire data, but that is NOT stated in FlyWire's own
Codex documentation and is unconfirmed as of 2026-09-15. A query has been
sent to flywire@princeton.edu. Local analysis is unaffected either way;
clear the terms before any FlyWire-derived number goes into the preprint,
a patent filing, or anything a reader could construe as commercial.

WHAT YOU GET (139,255 neurons, 2,700,513 unique connected pairs):
    classification.csv  root_id, flow, super_class, class, sub_class,
                        hemilineage, side, nerve
    neurons.csv         root_id, nt_type (ACH/GABA/GLUT/DA/SER/OCT) + scores
                        -- a MEASURED excitatory/inhibitory assignment, where
                        this project's simulations draw one at random
    coordinates.csv     one representative point per neuron, nanometres.
                        NOT the soma and NOT the arbor centre: it sits a
                        median 49.7 um (~2.1 arbor radii) away from the
                        neuron's true synapse centroid. Prefer build_clouds().
    connections.csv     pre, post, neuropil, syn_count, nt_type. Filtered at
                        >=5 synapses by Codex. ONE ROW PER NEUROPIL, so a
                        pair spanning several neuropils appears several
                        times -- 3,869,878 rows for 2,700,513 pairs. Use
                        load_connections(dedupe=True) unless you want rows.
    synapse_coordinates.csv  every synapse, 34.2M rows, ~303 MB gzipped.
                        Rows with blank ids are further coordinates for the
                        preceding pair (forward-fill).

Usage:
    from flywire_loader import download_flywire, load_annotations, \
        build_clouds, load_connections
    download_flywire()                      # ~360 MB, cached in flywire_data/
    ann = load_annotations()                # ids + class/side/nt arrays
    clouds = build_clouds()                 # centroid + arbor radius, cached
    pre, post, syn, npil = load_connections(dedupe=True)
"""
import csv
import gzip
import os
import time
import urllib.request

import numpy as np

BASE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
DATA_DIR = "flywire_data"

# name -> approximate gzipped size, for the progress line only
FILES = {
    "classification": 0.9e6,
    "neurons": 1.7e6,
    "coordinates": 5.1e6,
    "connections": 48e6,
    "cell_stats": 2.5e6,
    "synapse_coordinates": 303e6,
}
CORE_FILES = ["classification", "neurons", "coordinates", "connections"]


def _path(name, cache_dir=DATA_DIR):
    return os.path.join(cache_dir, name + ".csv.gz")


def download_flywire(cache_dir=DATA_DIR, files=None):
    """Fetch the Codex exports into `cache_dir`, skipping what is already there.

    `files=None` fetches everything including the 303 MB synapse file. Pass
    `files=CORE_FILES` for the ~56 MB subset that covers annotations,
    positions and the connection list.
    """
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    names = list(FILES) if files is None else list(files)
    out = {}
    for name in names:
        dest = _path(name, cache_dir)
        out[name] = dest
        if os.path.isfile(dest):
            print("  Found cached {}".format(dest))
            continue
        url = "{}/{}.csv.gz".format(BASE_URL, name)
        print("  Downloading {} (~{:.0f} MB) ...".format(url, FILES.get(name, 0) / 1e6))
        tmp = dest + ".part"
        t0 = time.time()
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as f_out:
            while True:
                chunk = response.read(1 << 22)
                if not chunk:
                    break
                f_out.write(chunk)
        os.replace(tmp, dest)
        print("    {:.1f} MB in {:.0f}s".format(os.path.getsize(dest) / 1e6, time.time() - t0))
    return out


def load_annotations(cache_dir=DATA_DIR):
    """Return a dict of per-neuron annotation arrays, all aligned to `ids`.

    `ids` is sorted, so the index of a root_id is stable across calls and
    across the other loaders here.
    """
    ids = []
    with gzip.open(_path("classification", cache_dir), "rt") as fh:
        rows = list(csv.DictReader(fh))
    ids = sorted(r["root_id"] for r in rows)
    index = {k: i for i, k in enumerate(ids)}
    n = len(ids)
    out = {
        "ids": np.array(ids),
        "index": index,
        "super_class": np.array([""] * n, dtype=object),
        "class": np.array([""] * n, dtype=object),
        "sub_class": np.array([""] * n, dtype=object),
        "side": np.array([""] * n, dtype=object),
        "nt": np.array([""] * n, dtype=object),
    }
    for r in rows:
        i = index[r["root_id"]]
        out["super_class"][i] = r["super_class"]
        out["class"][i] = r["class"]
        out["sub_class"][i] = r["sub_class"]
        out["side"][i] = r["side"]
    with gzip.open(_path("neurons", cache_dir), "rt") as fh:
        for r in csv.DictReader(fh):
            i = index.get(r["root_id"])
            if i is not None:
                out["nt"][i] = r["nt_type"]
    return out


def load_representative_points(cache_dir=DATA_DIR, index=None):
    """Codex `coordinates.csv` as an (N, 3) array in MICRONS, NaN where absent.

    Several rows per neuron are averaged. See the module docstring for why you
    probably want build_clouds() instead.
    """
    if index is None:
        index = load_annotations(cache_dir)["index"]
    acc = {}
    with gzip.open(_path("coordinates", cache_dir), "rt") as fh:
        for r in csv.DictReader(fh):
            i = index.get(r["root_id"])
            if i is None:
                continue
            p = r["position"].strip("[] ").split()
            acc.setdefault(i, []).append([float(v) / 1000.0 for v in p])
    P = np.full((len(index), 3), np.nan)
    for i, v in acc.items():
        P[i] = np.mean(v, axis=0)
    return P


def load_connections(cache_dir=DATA_DIR, index=None, dedupe=True, min_syn=0):
    """Connection list as index arrays.

    Returns (pre, post, syn_count, n_neuropils). With `dedupe=True` each
    DIRECTED (pre, post) pair appears exactly once, with `syn_count` summed
    over neuropils and `n_neuropils` recording how many rows were merged.
    With `dedupe=False` you get the raw rows, one per neuropil,
    and `n_neuropils` is all ones -- which will silently overweight pairs whose
    synapses span several neuropils, i.e. the long-range ones.
    """
    if index is None:
        index = load_annotations(cache_dir)["index"]
    agg = {}
    order = []
    with gzip.open(_path("connections", cache_dir), "rt") as fh:
        for r in csv.DictReader(fh):
            a = index.get(r["pre_root_id"])
            b = index.get(r["post_root_id"])
            if a is None or b is None:
                continue
            c = int(r["syn_count"])
            if not dedupe:
                order.append((a, b, c, 1))
                continue
            key = (a, b)
            hit = agg.get(key)
            if hit is None:
                agg[key] = [c, 1]
                order.append(key)
            else:
                hit[0] += c
                hit[1] += 1
    if dedupe:
        pre = np.empty(len(order), np.int32)
        post = np.empty(len(order), np.int32)
        syn = np.empty(len(order), np.int32)
        npil = np.empty(len(order), np.int16)
        for i, key in enumerate(order):
            pre[i], post[i] = key
            syn[i], npil[i] = agg[key]
    else:
        arr = np.array(order, np.int64)
        pre, post, syn, npil = (arr[:, 0].astype(np.int32), arr[:, 1].astype(np.int32),
                                arr[:, 2].astype(np.int32), arr[:, 3].astype(np.int16))
    if min_syn > 0:
        keep = syn >= min_syn
        pre, post, syn, npil = pre[keep], post[keep], syn[keep], npil[keep]
    return pre, post, syn, npil


def _stream_synapses(cache_dir, block=1 << 26):
    """Yield (pre_idx, post_idx, xyz) blocks from synapse_coordinates.csv.gz.

    Forward-fills the blank-id continuation rows. Coordinates stay in the
    file's native nanometres; callers convert. Unresolvable ids come through
    as -1 so the caller decides whether to drop them.
    """
    index = _stream_synapses.index
    with gzip.open(_path("synapse_coordinates", cache_dir), "rb") as fh:
        fh.readline()
        tail = b""
        a = b = -1
        while True:
            blk = fh.read(block)
            if not blk:
                break
            blk = tail + blk
            cut = blk.rfind(b"\n")
            if cut < 0:
                tail = blk
                continue
            tail = blk[cut + 1:]
            lines = blk[:cut].split(b"\n")
            pi = np.empty(len(lines), np.int32)
            qi = np.empty(len(lines), np.int32)
            X = np.empty((len(lines), 3), np.float64)
            k = 0
            for ln in lines:
                p = ln.split(b",")
                if len(p) != 5:
                    continue
                if p[0]:
                    a = index.get(p[0], -1)
                    b = index.get(p[1], -1)
                pi[k] = a
                qi[k] = b
                X[k, 0] = float(p[2])
                X[k, 1] = float(p[3])
                X[k, 2] = float(p[4])
                k += 1
            yield pi[:k], qi[:k], X[:k]


def build_clouds(cache_dir=DATA_DIR, index=None, force=False):
    """Per-neuron synapse-cloud centroid and arbor size, in MICRONS.

    A synapse is a physical contact point on BOTH partners, so each one is
    accumulated onto the pre and the post neuron. Returns a dict with
    `centroid` (N, 3), `rg` (N,) radius of gyration -- the RMS distance of the
    neuron's synapses from its own centroid, i.e. how far the arbor sprawls --
    and `n_syn` (N,). Neurons with no synapses in the file come back NaN.

    Costs one streaming pass over 34.2M rows (~45 s) and is then cached as
    flywire_data/neuron_clouds.npz.
    """
    out_path = os.path.join(cache_dir, "neuron_clouds.npz")
    if os.path.isfile(out_path) and not force:
        z = np.load(out_path)
        return {"centroid": z["centroid"], "rg": z["rg"], "n_syn": z["n_syn"]}
    if index is None:
        index = load_annotations(cache_dir)["index"]
    n = len(index)
    cnt = np.zeros(n, np.int64)
    S = np.zeros((n, 3), np.float64)
    S2 = np.zeros(n, np.float64)
    _stream_synapses.index = {k.encode(): v for k, v in index.items()}
    rows = 0
    t0 = time.time()
    for pi, qi, X in _stream_synapses(cache_dir):
        r2 = (X * X).sum(1)
        for side in (pi, qi):
            m = side >= 0
            s = side[m]
            np.add.at(cnt, s, 1)
            np.add.at(S, s, X[m])
            np.add.at(S2, s, r2[m])
        rows += len(pi)
    print("  {} synapse rows in {:.0f}s".format(rows, time.time() - t0))
    have = cnt > 0
    C = np.full((n, 3), np.nan)
    C[have] = S[have] / cnt[have, None]
    rg = np.full(n, np.nan)
    rg[have] = np.sqrt(np.maximum(S2[have] / cnt[have] - (C[have] ** 2).sum(1), 0.0))
    C /= 1000.0
    rg /= 1000.0
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    np.savez(out_path, centroid=C, rg=rg, n_syn=cnt, n_rows=rows)
    return {"centroid": C, "rg": rg, "n_syn": cnt}


def pair_synapse_sums(pairs, cache_dir=DATA_DIR, index=None):
    """Synapse sums for specific neuron pairs -- the leave-one-out ingredient.

    `pairs` is an iterable of (i, j) index tuples; direction is ignored, so
    synapses from j->i are pooled with i->j. Returns dict keyed by the sorted
    tuple, value (count, sum_xyz (3,), sum_r2), all in NANOMETRES so they can
    be subtracted directly from build_clouds()' pre-conversion accumulators.

    Costs a second full pass over the synapse file. Matching is vectorized on
    a packed int64 key (lo << 20 | hi); a per-row Python loop over 34.2M rows
    would take longer than the whole rest of the analysis.
    """
    if index is None:
        index = load_annotations(cache_dir)["index"]
    seen = set()
    for i, j in pairs:
        seen.add((i, j) if i <= j else (j, i))
    keys = np.array(sorted(seen), np.int64)
    packed = (keys[:, 0] << 20) | keys[:, 1]
    cnt = np.zeros(len(keys), np.int64)
    S = np.zeros((len(keys), 3), np.float64)
    S2 = np.zeros(len(keys), np.float64)
    _stream_synapses.index = {k.encode(): v for k, v in index.items()}
    for pi, qi, X in _stream_synapses(cache_dir):
        ok = (pi >= 0) & (qi >= 0)
        if not ok.any():
            continue
        p = pi[ok].astype(np.int64)
        q = qi[ok].astype(np.int64)
        Xo = X[ok]
        lo = np.minimum(p, q)
        hi = np.maximum(p, q)
        key = (lo << 20) | hi
        at = np.searchsorted(packed, key)
        at[at >= len(packed)] = 0
        hit = packed[at] == key
        if not hit.any():
            continue
        s = at[hit]
        Xh = Xo[hit]
        np.add.at(cnt, s, 1)
        np.add.at(S, s, Xh)
        np.add.at(S2, s, (Xh * Xh).sum(1))
    return {(int(a), int(b)): (cnt[i], S[i], S2[i])
            for i, (a, b) in enumerate(keys)}


if __name__ == "__main__":
    print("FlyWire FAFB v783 loader -- self-test")
    download_flywire(files=CORE_FILES)
    ann = load_annotations()
    print("  neurons: {}".format(len(ann["ids"])))
    from collections import Counter
    print("  super_class: {}".format(Counter(ann["super_class"]).most_common(4)))
    print("  nt_type:     {}".format(Counter(ann["nt"]).most_common(4)))
    pre, post, syn, npil = load_connections(index=ann["index"], dedupe=True)
    print("  unique directed pairs: {}  synapses: {}".format(len(pre), syn.sum()))
    print("  pairs spanning >1 neuropil: {} ({:.1f}%)".format(
        (npil > 1).sum(), 100.0 * (npil > 1).mean()))
    P = load_representative_points(index=ann["index"])
    ok = np.isfinite(P).all(1)
    print("  representative points: {} present, extent(um) {} .. {}".format(
        ok.sum(), np.round(np.nanmin(P, 0), 1), np.round(np.nanmax(P, 0), 1)))
    print("  (build_clouds() needs the 303 MB synapse file; run exp55 for that)")
