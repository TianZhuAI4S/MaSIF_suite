"""Read legacy numeric surfaces and safe NPZ inputs without TensorFlow or PyMesh."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def mask_features(features, feat_mask):
    """Select original feature columns, preserving the five-feature ordering."""
    if features.shape[-1] != len(feat_mask):
        raise ValueError("Raw feature width must match the original feature mask length")
    return features[..., np.asarray(feat_mask, dtype=bool)]


def flip_binder(data, feat_mask):
    """Reflect theta and all features except original hydrophobicity column 4."""
    result = {key: value.copy() for key, value in data.items()}
    result["theta"] = 2 * np.pi - result["theta"]
    result["features"] *= -1
    # Use original feature identities, including masks with hydrophobicity removed.
    for selected, original in enumerate(np.flatnonzero(feat_mask)):
        if original == 4:
            result["features"][..., selected] *= -1
    return result


def pad_indices(indices, vertices):
    """Pad each neighbor row with its own surface index, as in the TF helper."""
    padded = np.empty((len(indices), vertices), dtype=np.int64)
    for center, row in enumerate(indices):
        row = np.asarray(row, dtype=np.int64)
        if row.ndim != 1 or len(row) > vertices:
            raise ValueError("Neighbor lists must be one-dimensional and fit patch width")
        padded[center] = center
        padded[center, :len(row)] = row
    if (padded < 0).any() or (padded >= len(indices)).any():
        raise ValueError("Neighbor index outside the loaded full surface")
    return padded


def load_surface(directory, pid="p1", feat_mask=(1, 1, 1, 1, 1), trusted_indices=False,
                 require_indices=True):
    """Load p1/p2 arrays; ragged pickled indices require explicit local trust."""
    directory = Path(directory)
    names = dict(features="input_feat", rho="rho_wrt_center", theta="theta_wrt_center", mask="mask")
    data = {key: np.load(directory / f"{pid}_{suffix}.npy", allow_pickle=False)
            for key, suffix in names.items()}
    data["features"] = mask_features(data["features"], feat_mask)
    if require_indices:
        indices = np.load(directory / f"{pid}_list_indices.npy", allow_pickle=trusted_indices,
                          encoding="latin1")
        data["indices"] = pad_indices(indices, data["features"].shape[1])
    label_path = directory / f"{pid}_iface_labels.npy"
    if label_path.exists():
        interface = np.load(label_path, allow_pickle=False)
        data["labels"] = np.stack((interface, 1 - interface), axis=1)
    return data


def load_npz(path):
    """Read primitive arrays from a packet, rejecting pickled object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def tensors(data, device="cpu"):
    """Cast numeric NN inputs to float32 and all indexing arrays to int64."""
    result = {}
    for key, value in data.items():
        if np.asarray(value).dtype.kind not in "biuf":
            continue
        dtype = torch.int64 if key in ("indices", "pos_idx", "neg_idx") else torch.float32
        result[key] = torch.as_tensor(np.asarray(value).copy(), dtype=dtype, device=device)
    return result


def sample_packet(data, config, rng, training=True, batch_size=100):
    """Apply balanced site or 32-patch ligand sampling to a complete NPZ packet."""
    task = config["task"]
    result = dict(data)
    if task == "site":
        labels = result["labels"]
        if labels.ndim != 2 or labels.shape[1] != 2:
            raise ValueError("Site packets require two-column one-hot labels")
        positive, negative = np.flatnonzero(labels[:, 0] == 1), np.flatnonzero(labels[:, 1] == 1)
        if not len(positive) or not len(negative):
            raise ValueError("Site training packets require both positive and negative vertices")
        if training:
            positive, negative = rng.permutation(positive), rng.permutation(negative)
        count = min(len(positive), len(negative))
        if config["n_conv_layers"] == 1:
            count = min(count, batch_size // 2)
            subset = np.concatenate((negative[:count], positive[:count]))
            result = {k: v[subset] if k in ("features", "rho", "theta", "mask", "labels", "indices")
                      else v for k, v in result.items()}
            positive, negative = np.arange(count, 2 * count), np.arange(count)
        result.update(pos_idx=positive[:count], neg_idx=negative[:count])
    elif task == "ligand" and "pocket_labels" in data:
        labels = data["pocket_labels"]
        eligible = [i for i in range(labels.shape[1]) if np.count_nonzero(labels[:, i]) >= 32]
        if not eligible:
            raise ValueError("No ligand pocket has at least 32 points")
        pocket = int(rng.choice(eligible)) if training else eligible[0]
        points = np.flatnonzero(labels[:, pocket])
        selected = rng.choice(points, 32, replace=False) if training else points[:32]
        result = {k: v[selected] for k, v in data.items() if k in ("features", "rho", "theta", "mask")}
        label = int(labels[:, pocket].max()) - 1
        if not 0 <= label < config["n_ligands"]:
            raise ValueError("Ligand labels must be one-based class IDs")
        result["labels"] = np.eye(config["n_ligands"], dtype=np.float32)[label]
    return result


class PacketDataset(Dataset):
    """One protein/pocket/alignment packet per NPZ file with explicit splits."""

    def __init__(self, directory, config, seed=0, training=True, batch_size=100):
        """Filter ineligible surfaces and index bounded alignment batches explicitly."""
        if batch_size < 1 or (config["task"] == "site" and batch_size < 2):
            raise ValueError("batch_size must be positive (at least two for site)")
        directory = Path(directory)
        self.paths = [directory] if directory.is_file() else sorted(directory.glob("*.npz"))
        if not self.paths:
            raise ValueError(f"No numeric NPZ training packets in {directory}")
        self.config, self.seed, self.training = config, seed, training
        self.batch_size, self.epoch = batch_size, 0
        self.entries, self.excluded = [], []
        for path in self.paths:
            data = load_npz(path)
            reason = None
            if config["task"] == "site":
                labels = data["labels"]
                if labels.ndim != 2 or labels.shape[1] != 2:
                    raise ValueError(f"Site labels must have two columns: {path}")
                if not (labels[:, 0] == 1).any() or not (labels[:, 1] == 1).any():
                    reason = "site lacks positive or negative vertices"
            elif config["task"] == "ligand" and "pocket_labels" in data:
                if not (np.count_nonzero(data["pocket_labels"], axis=0) >= 32).any():
                    reason = "no ligand pocket has 32 points"
            if reason:
                self.excluded.append(dict(path=str(path), reason=reason))
                continue
            if config["task"] in ("score", "alignment"):
                count = len(data["features"])
                self.entries.extend((path, start, min(start + batch_size, count))
                                    for start in range(0, count, batch_size))
            else:
                self.entries.append((path, None, None))
        if not self.entries:
            raise ValueError(f"No eligible training records in {directory}: {self.excluded}")

    def __len__(self):
        """Return the number of independent structures or prepared batches."""
        return len(self.entries)

    def __getitem__(self, index):
        """Load and sample reproducibly by epoch and packet index."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        path, start, stop = self.entries[index]
        data = load_npz(path)
        if start is not None:
            # Every alignment is used once; all chunks share the same per-file permutation.
            file_index = self.paths.index(path)
            order = np.arange(len(data["features"]))
            if self.training:
                order = np.random.default_rng(np.random.SeedSequence(
                    [self.seed, self.epoch, file_index])).permutation(order)
            selected = order[start:stop]
            return {key: value[selected] for key, value in data.items() if key in ("features", "labels")}
        return sample_packet(data, self.config, rng,
                             self.training, self.batch_size)


class SearchCacheDataset(Dataset):
    """Sample the original search cache into positive/binder/negative/binder blocks."""

    def __init__(self, directory, config, steps=1000, batch_size=32, seed=0, split="training"):
        """Memory-map original cache arrays and use only the requested split indices."""
        if batch_size < 4 or batch_size % 4 or steps < 1:
            raise ValueError("Search batch size must be positive and divisible by four")
        self.steps, self.count, self.seed, self.epoch = steps, batch_size // 4, seed, 0
        self.config = config
        directory = Path(directory)
        self.groups = {}
        for group in ("pos", "binder", "neg"):
            names = dict(features="input_feat", rho="rho_wrt_center", theta="theta_wrt_center", mask="mask")
            self.groups[group] = {k: np.load(directory / f"{group}_{v}.npy", mmap_mode="r", allow_pickle=False)
                                  for k, v in names.items()}
        self.positive = np.load(directory / f"pos_{split}_idx.npy", allow_pickle=False).astype(np.int64)
        self.negative = np.load(directory / f"neg_{split}_idx.npy", allow_pickle=False).astype(np.int64)
        if min(len(self.positive), len(self.negative)) < self.count:
            raise ValueError("Not enough search pairs for requested batch size")

    def __len__(self):
        """Return the configured number of independently resampled training steps."""
        return self.steps

    def __getitem__(self, index):
        """Match construct_batch's ordering and hydrophobicity-preserving reflection."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        positive = rng.permutation(self.positive)[:self.count]
        negative = rng.permutation(self.negative)[:self.count]
        blocks = []
        for group, selected in (("pos", positive), ("binder", positive), ("neg", negative)):
            data = {k: np.asarray(v[selected]).copy() for k, v in self.groups[group].items()}
            data["features"] = mask_features(data["features"], self.config["feat_mask"])
            if group == "binder":
                data = flip_binder(data, self.config["feat_mask"])
            blocks.append(data)
        blocks.append(blocks[1])
        return {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
