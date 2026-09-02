"""ssc_loader.py -- Spiking Speech Commands (SSC), the sibling dataset to
SHD from the same Zenke Lab release. Same direct-download-no-tonic approach
as shd_loader.py (h5py only, no C-compiler-requiring `tonic`/`expelliarmus`).

Source: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
Cramer, Stradmann, Schemmel & Zenke (2022), IEEE TNNLS.
700 input channels, 35 word classes (Google Speech Commands re-encoded to
spikes), CC BY 4.0. Uses the VALID split (ssc_valid.h5.gz, ~155MB) rather
than the 1.19GB train split or 323MB test split, purely for download size --
this is a tinkering-grade second-dataset check, not a claim about SSC's
canonical train/test protocol.

HDF5 structure: identical schema to SHD (confirmed via same md5sums.txt
listing at the same BASE_URL) --
    root/spikes/times[]  -- per-sample spike times, in SECONDS despite the
                             lab's docs calling the field "Times (ms)" (see
                             shd_loader.py's note; same correction applied)
    root/spikes/units[]  -- per-sample channel ids, 0-699
    root/labels[]        -- per-sample class label, 0-34

Usage:
    from ssc_loader import load_ssc_samples
    stimuli, labels = load_ssc_samples(n_samples=200, n_channels_out=140,
                                        bin_ms=4.0, max_steps=250)
"""
import numpy as np
import os

from shd_loader import _download_and_gunzip, _get_md5_hashes, BASE_URL


def load_ssc_samples(n_samples=200, n_channels_out=140, bin_ms=4.0,
                      max_steps=250, split="valid", cache_dir="./shd_data",
                      seed=0):
    """Load real SSC samples and convert to TopoForge's stimulus format.
    Mirrors shd_loader.load_shd_samples exactly (same conversion, same
    ms/seconds correction) -- only the source filename and cache-file
    prefix differ.

    Returns:
        stimuli: list of (T, n_channels_out) float arrays. Each entry
            [t, c] is the spike COUNT in channel c during step t.
        labels: list of int class labels (0-34)
    """
    import h5py

    os.makedirs(cache_dir, exist_ok=True)
    converted_cache = os.path.join(
        cache_dir, "ssc_converted_n{}_c{}_b{}_s{}_{}.npz".format(
            n_samples, n_channels_out, bin_ms, max_steps, split))

    if os.path.exists(converted_cache):
        print("Loading cached converted samples from {}".format(converted_cache))
        data = np.load(converted_cache, allow_pickle=True)
        stimuli_loaded = [np.asarray(s, dtype=np.float64) for s in data["stimuli"]]
        return stimuli_loaded, list(data["labels"])

    filename = "ssc_{}.h5.gz".format(split)
    print("Fetching MD5 checksums from official source...")
    md5_hashes = _get_md5_hashes()
    print("Obtaining {} (downloads once, then cached locally)...".format(filename))
    h5_path = _download_and_gunzip(filename, cache_dir, md5_hashes)

    print("Reading HDF5 file...")
    with h5py.File(h5_path, 'r') as f:
        times_all = f['spikes']['times']
        units_all = f['spikes']['units']
        labels_all = f['labels'][:]
        n_total = len(labels_all)
        print("  {} split: {} total samples available".format(split, n_total))

        rng = np.random.default_rng(seed)
        idx_pool = rng.permutation(n_total)[:n_samples]

        n_channels_raw = 700
        pool_factor = max(1, n_channels_raw // n_channels_out)

        stimuli = []
        labels = []
        skipped = 0

        for idx in idx_pool:
            idx = int(idx)
            t_ms = np.array(times_all[idx], dtype=np.float64) * 1000.0
            ch = np.array(units_all[idx], dtype=np.int64)
            label = int(labels_all[idx])

            if len(t_ms) == 0:
                skipped += 1
                continue

            step_idx = (t_ms / bin_ms).astype(np.int64)
            step_idx = np.clip(step_idx, 0, max_steps - 1)
            pooled_ch = np.clip(ch // pool_factor, 0, n_channels_out - 1)

            stim = np.zeros((max_steps, n_channels_out), dtype=np.float64)
            np.add.at(stim, (step_idx, pooled_ch), 1.0)

            stimuli.append(stim)
            labels.append(label)

    if skipped:
        print("Skipped {} empty recordings.".format(skipped))

    np.savez_compressed(converted_cache,
                         stimuli=np.array(stimuli, dtype=object),
                         labels=np.array(labels))
    print("Converted and cached {} samples to {}".format(
        len(stimuli), converted_cache))

    return stimuli, labels


if __name__ == "__main__":
    print("=" * 70)
    print("SSC LOADER SELF-TEST (direct download, no tonic)")
    print("=" * 70)
    stimuli, labels = load_ssc_samples(n_samples=20, n_channels_out=140,
                                        bin_ms=4.0, max_steps=250)
    print("\nLoaded {} samples.".format(len(stimuli)))
    print("Sample shapes: {}".format(stimuli[0].shape))
    print("Label distribution: {}".format(
        {l: labels.count(l) for l in sorted(set(labels))}))
    print("Number of distinct classes seen: {}".format(len(set(labels))))
    print("Mean spikes per sample: {:.1f}".format(
        np.mean([s.sum() for s in stimuli])))
    print("Mean active channels per sample (nonzero at any step): {:.1f}".format(
        np.mean([(s.sum(axis=0) > 0).sum() for s in stimuli])))
