"""Fast diagnostic v2: does V ever accumulate, or does it oscillate
and cancel out? Runs a SHORT life (3 samples, not the full 15-epoch
run) with full visibility into Rhat and bridge-mass at EVERY reward
check, instead of only the final endpoint.

Run: python diagnose_exp37b.py
"""
import numpy as np
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid
from shd_loader import load_shd_samples

N_INPUT = 140
N_OUTPUT = 60
N = N_INPUT + N_OUTPUT
INPUT_TYPE, OUTPUT_TYPE = 0, 1

print("Loading real samples...")
all_stimuli, all_labels = load_shd_samples(
    n_samples=200, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=200)
from collections import Counter
counts = Counter(all_labels)
top = [c for c, _ in counts.most_common(2)]
target_samples = [s for s, l in zip(all_stimuli, all_labels) if l == top[0]][:3]
distractor_samples = [s for s, l in zip(all_stimuli, all_labels) if l == top[1]][:3]
print("Target class {}, distractor class {}".format(top[0], top[1]))

# INTERLEAVED placement (the condition with real activity, per the runtime clue)
rng = np.random.default_rng(11)
coords, cids = [], []
all_types = [INPUT_TYPE] * N_INPUT + [OUTPUT_TYPE] * N_OUTPUT
rng.shuffle(all_types)
for t_ in all_types:
    th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 10)
    coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)])
    cids.append(t_)
coords = np.array(coords); cids = np.array(cids)

rng2 = np.random.default_rng(7)
src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
keep = src != dst; src, dst = src[keep], dst[keep]
inhib = rng2.random(N) < 0.20
v = np.zeros(N); refrac = np.zeros(N, dtype=int)
C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
Rhat = np.zeros(2)
g = SpatialGrid(coords, 6.0)
nbr = [g.within(i, 6.0) for i in range(N)]
out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
for s2, d2b in zip(src, dst):
    out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

input_idx = np.where(cids == INPUT_TYPE)[0]
output_idx = np.where(cids == OUTPUT_TYPE)[0]
in_set = set(input_idx.tolist()); out_set = set(output_idx.tolist())

rng3 = np.random.default_rng(0)
combined = [(s, 0) for s in target_samples] + [(s, 1) for s in distractor_samples]
rng3.shuffle(combined)

t = 0
recent_output_fired = False
print("\nTracking Rhat and bridge-mass at every reward check:")
for sample_idx, (sample_arr, class_label) in enumerate(combined):
    T_sample = sample_arr.shape[0]
    for local_t in range(T_sample):
        inp = rng3.uniform(0, 0.01, N)
        inp[input_idx] += sample_arr[local_t] * 0.35
        v_ = v * 0.90 + inp
        fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
        if len(f):
            for fi in f:
                for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
        C.tick(); E.tick(); V.tick()
        if len(f):
            fs = set(int(x) for x in f)
            for i in f:
                i = int(i)
                for j in nbr[i]:
                    if int(j) in fs:
                        C.deposit(i, int(j), 1.0); E.deposit(i, int(j), 1.0)
        v = np.maximum(v_, 0); v[fired] = 0
        refrac[fired] = 3; refrac[refrac > 0] -= 1

        if len(f):
            if len(set(int(x) for x in f) & out_set) > 0:
                recent_output_fired = True

        if t % 20 == 6:
            base = 1.0 if (class_label == 0 and recent_output_fired) else \
                   (0.0 if class_label == 0 else (-0.3 if recent_output_fired else 0.3))
            delta = base - Rhat[class_label]
            n_deposits = 0
            if abs(delta) > 1e-9:
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0:
                        V.deposit(key[0], key[1], delta * ev)
                        n_deposits += 1
            Rhat[class_label] += 0.15 * delta

            bridge_now = 0.0
            for si, di in zip(src, dst):
                si2, di2 = int(si), int(di)
                if si2 in in_set and di2 in out_set:
                    bridge_now += V.get(si2, di2)
                elif di2 in in_set and si2 in out_set:
                    bridge_now += V.get(di2, si2)

            print("  sample{} class{} t={:>4} recent_fired={} base={:+.2f} delta={:+.3f} "
                  "n_deposits={:>4} bridge_mass_now={:+.3f}".format(
                      sample_idx, class_label, t, recent_output_fired, base, delta,
                      n_deposits, bridge_now))
            recent_output_fired = False

        t += 1
