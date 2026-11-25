"""Module 1 - Data acquisition and preprocessing (ModelNet10).

* Downloads ModelNet10 automatically through torch_geometric.datasets.
* Samples 1024 points uniformly on the mesh surface (SamplePoints(1024)).
* Centres each cloud at the origin and rescales it inside the unit sphere.
* Training augmentation: random rotation about the vertical (z) axis + small
  Gaussian noise.

Run stand-alone to download / preprocess once:
    python data/download_and_preprocess.py --root data/ModelNet10
"""
import argparse
import os

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet

NUM_POINTS = 1024


class UnitSphere:
    """Centre the cloud (mean at origin) and scale it so max ||p|| = 1."""

    def __call__(self, data):
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        scale = data.pos.norm(dim=1).max().clamp(min=1e-9)
        data.pos = data.pos / scale
        return data

    def __repr__(self):
        return "UnitSphere()"


class RandomRotateZ:
    """Random rotation around the vertical axis (ModelNet objects are z-up)."""

    def __call__(self, data):
        theta = torch.rand(1).item() * 2 * torch.pi
        c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        data.pos = data.pos @ rot.T
        return data


class GaussianNoise:
    """Additive Gaussian jitter on point coordinates."""

    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma, self.clip = sigma, clip

    def __call__(self, data):
        noise = (torch.randn_like(data.pos) * self.sigma).clamp(-self.clip, self.clip)
        data.pos = data.pos + noise
        return data


def get_datasets(root="data/ModelNet10", name="10", num_points=NUM_POINTS):
    """Return (train_set, test_set). Sampling + normalisation are cached on disk."""
    pre_transform = T.Compose([T.SamplePoints(num_points), UnitSphere()])
    train_aug = T.Compose([RandomRotateZ(), GaussianNoise(0.01, 0.05)])
    train_set = ModelNet(root, name=name, train=True, transform=train_aug,
                         pre_transform=pre_transform)
    test_set = ModelNet(root, name=name, train=False, pre_transform=pre_transform)
    return train_set, test_set


def class_names(dataset):
    """Class names in label order (ModelNet raw folders are sorted alphabetically)."""
    raw = dataset.raw_dir
    if os.path.isdir(raw):
        names = sorted(d for d in os.listdir(raw)
                       if os.path.isdir(os.path.join(raw, d)))
        if len(names) == dataset.num_classes:
            return names
    return [str(i) for i in range(dataset.num_classes)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ModelNet10")
    ap.add_argument("--name", default="10", choices=["10", "40"])
    args = ap.parse_args()
    tr, te = get_datasets(args.root, args.name)
    print(f"ModelNet{args.name}: {len(tr)} train / {len(te)} test objects, "
          f"{tr.num_classes} classes")
    print("Classes:", class_names(tr))
    p = te[0].pos
    print("Sample cloud:", tuple(p.shape), "| max radius =", round(p.norm(dim=1).max().item(), 4),
          "| mean =", [round(v, 4) for v in p.mean(0).tolist()])
