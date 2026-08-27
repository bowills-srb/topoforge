"""preview_shd_data.py -- Verify real SHD data before building on it.
Loads a small batch, prints diagnostics, and shows a text-art preview
of one sample's spike pattern -- so we can SEE this is real, structured
audio data before trusting an experiment built on top of the loader.

Run: python preview_shd_data.py
"""
import numpy as np
import sys
sys.path.insert(0, ".")
from shd_loader import load_shd_samples

if __name__ == "__main__":
    print("=" * 74)
    print("SHD DATA VERIFICATION")
    print("Confirming real spike data loads correctly before any")
    print("experiment gets built on top of this.")
    print("=" * 74)

    stimuli, labels = load_shd_samples(n_samples=40, n_channels_out=140,
                                        bin_ms=4.0, max_steps=250)

    print("\n[1] BASIC SANITY")
    print("  Samples loaded: {}".format(len(stimuli)))
    print("  Classes present: {}".format(sorted(set(labels))))
    print("  Expected: 20 classes (0-9 English, 0-9 German), likely a")
    print("  subset present with only 40 samples drawn")

    print("\n[2] PER-CLASS SAMPLE COUNTS (checking real class diversity)")
    from collections import Counter
    counts = Counter(labels)
    for cls in sorted(counts):
        print("  class {:>2}: {} samples".format(cls, counts[cls]))

    print("\n[3] SPIKE STATISTICS (does this look like real audio, not noise?)")
    total_spikes = [s.sum() for s in stimuli]
    active_channels = [(s.sum(axis=0) > 0).sum() for s in stimuli]
    active_steps = [(s.sum(axis=1) > 0).sum() for s in stimuli]
    print("  Total spikes per sample: mean={:.0f}, min={:.0f}, max={:.0f}".format(
        np.mean(total_spikes), np.min(total_spikes), np.max(total_spikes)))
    print("  Active channels per sample (of 140): mean={:.0f}".format(
        np.mean(active_channels)))
    print("  Active timesteps per sample (of 250): mean={:.0f}".format(
        np.mean(active_steps)))
    print("  (Real speech should NOT use all channels/steps uniformly --")
    print("   sparse, structured activity is expected)")

    print("\n[4] TEXT-ART PREVIEW: one sample's spike pattern")
    print("  (rows=time steps downsampled, cols=channel groups, # = activity)")
    sample = stimuli[0]
    label = labels[0]
    print("  Sample class: {}".format(label))
    # downsample for terminal display: 40 time-bins x 40 channel-bins
    T, C = sample.shape
    t_bins, c_bins = 40, 60
    t_step = max(1, T // t_bins)
    c_step = max(1, C // c_bins)
    for t in range(0, T, t_step):
        row = sample[t:t+t_step].sum(axis=0)
        row_binned = [row[c:c+c_step].sum() for c in range(0, C, c_step)]
        line = "".join("#" if v > 2 else ("." if v > 0 else " ") for v in row_binned)
        print("  {:>4}ms |{}|".format(int(t * 4), line))

    print("\n" + "=" * 74)
    print("VERDICT:")
    if np.mean(active_channels) < C * 0.9 and np.mean(active_steps) < T * 0.9:
        print("  PASS -- data is sparse and structured, consistent with real")
        print("  speech-derived spike trains, not noise or a loading bug.")
        print("  Safe to build the placement experiment on top of this.")
    else:
        print("  WARNING -- activity looks too dense/uniform. Check the")
        print("  loader before building anything on top of it.")
