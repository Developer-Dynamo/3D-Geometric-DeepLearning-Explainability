"""Training loop: loss/accuracy tracking, best-weights checkpointing, training curves.

    python train.py --model dgcnn --epochs 100 --batch_size 32
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.download_and_preprocess import get_datasets          # noqa: E402
from models.dgcnn import DGCNN                                  # noqa: E402
from models.pointnet2 import PointNet2                          # noqa: E402
from utils.visualization import plot_training_curves           # noqa: E402


def build_model(name, num_classes, k=20):
    return DGCNN(num_classes, k=k) if name == "dgcnn" else PointNet2(num_classes)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for data in tqdm(loader, leave=False, desc="train" if train else "test"):
            data = data.to(device)
            B = data.num_graphs
            pos = data.pos.view(B, -1, 3)
            logits = model(pos)
            loss = F.cross_entropy(logits, data.y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * B
            correct += (logits.argmax(-1) == data.y).sum().item()
            n += B
    return tot_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dgcnn", choices=["dgcnn", "pointnet2"])
    ap.add_argument("--root", default="data/ModelNet10")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out, exist_ok=True)

    train_set, test_set = get_datasets(args.root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers)

    model = build_model(args.model, train_set.num_classes, args.k).to(device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    hist = {k: [] for k in ("train_loss", "train_acc", "test_loss", "test_acc")}
    best, ckpt = 0.0, os.path.join(args.out, f"{args.model}_best.pt")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, train_loader, device, opt)
        tel, tea = run_epoch(model, test_loader, device)
        sched.step()
        for k_, v in zip(hist, (trl, tra, tel, tea)):
            hist[k_].append(v)
        if tea > best:
            best = tea
            torch.save({"model": args.model, "k": args.k, "state_dict": model.state_dict(),
                        "num_classes": train_set.num_classes, "test_acc": tea, "epoch": ep}, ckpt)
        print(f"Epoch {ep:03d}/{args.epochs} | train loss {trl:.4f} acc {tra:.4f} | "
              f"test loss {tel:.4f} acc {tea:.4f} | best {best:.4f} | {time.time() - t0:.0f}s")
        with open(os.path.join(args.out, f"{args.model}_history.json"), "w") as f:
            json.dump(hist, f)
        plot_training_curves(os.path.join(args.out, "training_curves.png" if args.model == "dgcnn"
                                          else f"training_curves_{args.model}.png"), hist)
    print(f"Done. Best test accuracy {best:.4f}. Weights: {ckpt}")


if __name__ == "__main__":
    main()
