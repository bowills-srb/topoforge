"""shd_loader.py -- Real data, not synthetic patterns. v2: direct download.
Downloads SHD directly from the Zenke Lab (authoritative source), no
`tonic` dependency (which pulls in expelliarmus, requiring a C compiler
Windows doesn't ship with). Just h5py, which installs as a prebuilt
wheel -- no compiler needed.

Source: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
Cramer, Stradmann, Schemmel & Zenke (2022), IEEE TNNLS.
700 input channels, 20 classes (digits 0-9, English + German), CC BY 4.0.

HDF5 structure (confirmed from the lab's own page):
    root/spikes/times[]  -- per-sample spike times, ALREADY in ms
    root/spikes/units[]  -- per-sample channel ids, 0-699
    root/labels[]        -- per-sample class label, 0-19

Install: pip install h5py   (already satisfied if you tried tonic earlier)

Usage:
    from shd_loader import load_shd_samples
    stimuli, labels = load_shd_samples(n_samples=200, n_channels_out=140,
                                        bin_ms=4.0, max_steps=250)
"""
import numpy as np
import os
import gzip
import shutil
import hashlib
import urllib.request

BASE_URL = "https://zenkelab.org/datasets"


def _md5sum(file_path, chunk_size=2**20):
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def _download_and_gunzip(filename, cache_dir, md5_hashes):
    gz_path = os.path.join(cache_dir, filename)
    h5_path = gz_path[:-3]  # strip .gz

    if os.path.isfile(h5_path):
        print("  Found cached {}".format(h5_path))
        return h5_path

    if not os.path.isfile(gz_path):
        url = "{}/{}".format(BASE_URL, filename)
        print("  Downloading {} ...".format(url))
        tmp = gz_path + ".part"
        with urllib.request.urlopen(url) as response, open(tmp, 'wb') as f_out:
            shutil.copyfileobj(response, f_out)
        os.replace(tmp, gz_path)

    expected_md5 = md5_hashes.get(filename)
    if expected_md5:
        actual_md5 = _md5sum(gz_path)
        if actual_md5 != expected_md5:
            raise ValueError(
                "MD5 mismatch for {}: expected {}, got {}. "
                "Delete the file and retry.".format(filename, expected_md5, actual_md5))
        print("  MD5 verified for {}".format(filename))

    print("  Decompressing {} ...".format(gz_path))
    with gzip.open(gz_path, 'rb') as f_in, open(h5_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    return h5_path


def _get_md5_hashes():
    url = "{}/md5sums.txt".format(BASE_URL)
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        hashes = {}
        for line in data.split("\n"):
            parts = line.split()
            if len(parts) == 2:
                hashes[parts[1]] = parts[0]
        return hashes
    except Exception as e:
        print("  WARNING: could not fetch md5sums.txt ({}). "
              "Proceeding without integrity check.".format(e))
        return {}


def load_shd_samples(n_samples=200, n_channels_out=140, bin_ms=4.0,
                      max_steps=250, train=True, cache_dir="./shd_data",
                      seed=0):
    """Load real SHD samples and convert to TopoForge's stimulus format.

    Returns:
        stimuli: list of (T, n_channels_out) float arrays. Each entry
            [t, c] is the spike COUNT in channel c during step t.
        labels: list of int class labels (0-19)
    """
    import h5py

    os.makedirs(cache_dir, exist_ok=True)
    converted_cache = os.path.join(
        cache_dir, "shd_converted_n{}_c{}_b{}_s{}_{}.npz".format(
            n_samples, n_channels_out, bin_ms, max_steps,
            "train" if train else "test"))

    if os.path.exists(converted_cache):
        print("Loading cached converted samples from {}".format(converted_cache))
        data = np.load(converted_cache, allow_pickle=True)
        # explicit float64 cast: the object-dtype array used for storage
        # (defensive against ragged shapes, though samples are actually
        # uniform after fixed-size padding) can round-trip with the wrong
        # dtype through npz save/load -- cast back explicitly to be safe
        stimuli_loaded = [np.asarray(s, dtype=np.float64) for s in data["stimuli"]]
        return stimuli_loaded, list(data["labels"])

    filename = "shd_train.h5.gz" if train else "shd_test.h5.gz"
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
        print("  {} split: {} total samples available".format(
            "train" if train else "test", n_total))

        rng = np.random.default_rng(seed)
        idx_pool = rng.permutation(n_total)[:n_samples]

        n_channels_raw = 700
        pool_factor = max(1, n_channels_raw // n_channels_out)

        stimuli = []
        labels = []
        skipped = 0

        for idx in idx_pool:
            idx = int(idx)
            # NOTE: the Zenke Lab's own example code labels this "Times (ms)"
            # but empirical testing shows the raw values are actually in
            # SECONDS -- without this correction, all spikes collapse into
            # step 0 (values like 0.15s become ~0 when divided by a 4ms bin).
            # Caught by preview_shd_data.py's sanity check before this was
            # used to build anything on top of it.
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
    print("SHD LOADER SELF-TEST (direct download, no tonic)")
    print("=" * 70)
    stimuli, labels = load_shd_samples(n_samples=20, n_channels_out=140,
                                        bin_ms=4.0, max_steps=250)
    print("\nLoaded {} samples.".format(len(stimuli)))
    print("Sample shapes: {}".format(stimuli[0].shape))
    print("Label distribution: {}".format(
        {l: labels.count(l) for l in sorted(set(labels))}))
    print("Mean spikes per sample: {:.1f}".format(
        np.mean([s.sum() for s in stimuli])))
    print("Mean active channels per sample (nonzero at any step): {:.1f}".format(
        np.mean([(s.sum(axis=0) > 0).sum() for s in stimuli])))
