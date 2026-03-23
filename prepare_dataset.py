"""
Dataset preparation: load spectrograms from all five distribution types,
apply Jet colormap to produce RGB images (Section 2.4.1), and split into
train / validation / test sets.

Usage:
    python prepare_dataset.py --output_dir data/
"""

import argparse
import numpy as np
from matplotlib import cm

# Input files (one per distribution type) and class order
FILE_LIST = [
    'spectrograms/square_lattice.npz',
    'spectrograms/poisson_disk.npz',
    'spectrograms/hexagonal_lattice.npz',
    'spectrograms/inhom_random.npz',
    'spectrograms/cluster.npz',
]

CLASS_NAMES = [
    'SquareLattice', 'PoissonDisk', 'HexagonalLattice',
    'InhomRandom', 'Cluster',
]

N_CLASSES = 5
VAL_COUNT = 300
TEST_COUNT = 300


def load_and_split(file_list, seed=42):
    """Load all .npz files and split each class into train/val/test."""
    rng = np.random.RandomState(seed)
    splits = {s: {'specs': [], 'labels': []} for s in ('train', 'val', 'test')}

    for cls, fname in enumerate(file_list):
        specs = np.load(fname)['spectrograms']  # (N, Nfreq, Ntime)
        N = specs.shape[0]

        labels = np.zeros((N, N_CLASSES), dtype=np.float32)
        labels[:, cls] = 1

        idx = rng.permutation(N)
        boundaries = [TEST_COUNT, TEST_COUNT + VAL_COUNT]
        for split_name, sl in zip(('test', 'val', 'train'),
                                   (slice(0, boundaries[0]),
                                    slice(boundaries[0], boundaries[1]),
                                    slice(boundaries[1], None))):
            splits[split_name]['specs'].append(specs[idx[sl]])
            splits[split_name]['labels'].append(labels[idx[sl]])

        print(f"{CLASS_NAMES[cls]}: train {N - boundaries[1]}, "
              f"val {VAL_COUNT}, test {TEST_COUNT}")

    for s in splits:
        splits[s]['specs'] = np.concatenate(splits[s]['specs'])
        splits[s]['labels'] = np.concatenate(splits[s]['labels'])
    return splits


def apply_jet_colormap(gray, vmin=None, vmax=None):
    """Map single-channel dB spectrograms to RGB via Jet colormap."""
    if vmin is None:
        vmin = gray.min()
    if vmax is None:
        vmax = gray.max()
    normed = np.clip((gray - vmin) / (vmax - vmin), 0, 1)
    rgb = cm.jet(normed)[..., :3].astype(np.float32)
    return rgb, vmin, vmax


def main(output_dir):
    splits = load_and_split(FILE_LIST)

    # Shuffle training set
    perm = np.random.permutation(splits['train']['specs'].shape[0])
    splits['train']['specs'] = splits['train']['specs'][perm]
    splits['train']['labels'] = splits['train']['labels'][perm]

    # Apply Jet colormap (normalize using training set range)
    train_rgb, vmin, vmax = apply_jet_colormap(splits['train']['specs'])
    val_rgb, _, _ = apply_jet_colormap(splits['val']['specs'], vmin, vmax)
    test_rgb, _, _ = apply_jet_colormap(splits['test']['specs'], vmin, vmax)

    for name, specs, labels in [('train', train_rgb, splits['train']['labels']),
                                 ('val', val_rgb, splits['val']['labels']),
                                 ('test', test_rgb, splits['test']['labels'])]:
        path = f'{output_dir}/{name}_data.npz'
        np.savez(path, spectrograms=specs, labels=labels)
        print(f"{name}: {specs.shape} -> {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/')
    args = parser.parse_args()
    main(args.output_dir)
