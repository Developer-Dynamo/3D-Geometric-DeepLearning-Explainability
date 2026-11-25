"""Evaluation + explainability figures.

    python evaluate.py --model dgcnn

Outputs (in results/):
  metrics_<model>.json         overall acc, mean per-class acc, inference time per object
  saliency_3d_render_1.png     side-by-side raw vs saliency render (300 DPI)
  saliency_3d_render_2.png
  saliency_data_<n>.pt         coordinates + importance scores
  deletion_test.png            fidelity check of the saliency maps
It also updates the results table in README.md (between the RESULTS markers).
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.download_and_preprocess import class_names, get_datasets      # noqa: E402
from train import build_model                                            # noqa: E402
from utils.explainability import (compute_saliency, deletion_test,       # noqa: E402
                                  save_saliency, smooth_importance)
from utils.visualization import plot_deletion_curve, plot_saliency_comparison  # noqa: E402


@torch.no_grad()
def predict_all(model, loader, device):
    ys, ps = [], []
    for data in loader:
        data = data.to(device)
        ps.append(model(data.pos.view(data.num_graphs, -1, 3)).argmax(-1).cpu())
        ys.append(data.y.cpu())
    return torch.cat(ys), torch.cat(ps)


@torch.no_grad()
def inference_time_ms(model, test_set, device, n=100):
    model.eval()
    sample = [test_set[i].pos.unsqueeze(0).to(device) for i in range(min(n, len(test_set)))]
    for s in sample[:10]:                                   # warm-up
        model(s)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for s in sample:
        model(s)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / len(sample) * 1000


def update_readme(path, table_md):
    if not os.path.exists(path):
        return
    txt = open(path, encoding="utf-8").read()
    new = re.sub(r"(<!-- RESULTS_START -->).*?(<!-- RESULTS_END -->)",
                 lambda m: f"{m.group(1)}\n{table_md}\n{m.group(2)}", txt, flags=re.S)
    open(path, "w", encoding="utf-8").write(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dgcnn", choices=["dgcnn", "pointnet2"])
    ap.add_argument("--root", default="data/ModelNet10")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--smooth", type=int, default=0,
                    help="k for optional neighbour smoothing of saliency maps (0 = off)")
    ap.add_argument("--no_readme", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    _, test_set = get_datasets(args.root)
    names = class_names(test_set)
    ckpt = torch.load(args.weights or os.path.join(args.out, f"{args.model}_best.pt"),
                      map_location=device)
    model = build_model(args.model, ckpt["num_classes"], ckpt.get("k", 20)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---- quantitative metrics -------------------------------------------------
    y, p = predict_all(model, DataLoader(test_set, batch_size=32), device)
    overall = (y == p).float().mean().item()
    per_class = {names[c]: (p[y == c] == c).float().mean().item() for c in range(len(names))
                 if (y == c).any()}
    mean_cls = float(np.mean(list(per_class.values())))
    ms = inference_time_ms(model, test_set, device)
    metrics = {"model": args.model, "device": str(device), "overall_accuracy": overall,
               "mean_class_accuracy": mean_cls, "per_class_accuracy": per_class,
               "inference_ms_per_object": ms, "num_test_objects": len(test_set)}
    print(f"Overall acc {overall:.4f} | mean class acc {mean_cls:.4f} | {ms:.2f} ms/object ({device})")
    for k_, v in per_class.items():
        print(f"  {k_:12s} {v:.4f}")

    # ---- explainability renders -----------------------------------------------
    # pick correctly classified, confidently predicted test objects of 2 different classes
    pos_all = torch.stack([test_set[i].pos for i in range(len(test_set))])
    labels = torch.tensor([int(test_set[i].y) for i in range(len(test_set))])
    parts = [compute_saliency(model, pos_all[i:i + 8].to(device))
             for i in range(0, len(pos_all), 8)]
    sal = {k_: torch.cat([q[k_] for q in parts]) for k_ in ("coords", "importance", "pred", "probs")}
    n_s = sal["coords"].shape[0]
    imp = sal["importance"]
    if args.smooth > 0:
        imp = smooth_importance(sal["coords"], imp, args.smooth)
    conf = sal["probs"].max(-1).values
    ok = (sal["pred"] == labels[:n_s])
    chosen, used = [], set()
    for want in ("chair", "toilet", "table", "bed", "sofa", "monitor"):      # shapes with clear structure
        if want in names:
            c = names.index(want)
            cand = torch.where(ok & (labels[:n_s] == c))[0]
            if len(cand):
                chosen.append(int(cand[conf[cand].argmax()])); used.add(c)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        chosen += [int(i) for i in torch.where(ok)[0][:2 - len(chosen)]]
    for r, i in enumerate(chosen, 1):
        nm = names[int(labels[i])]
        plot_saliency_comparison(os.path.join(args.out, f"saliency_3d_render_{r}.png"),
                                 sal["coords"][i], imp[i],
                                 suptitle=f"{args.model.upper()} saliency - class '{nm}' "
                                          f"(confidence {conf[i]:.2f})")
        save_saliency(os.path.join(args.out, f"saliency_data_{r}.pt"), sal["coords"][i], imp[i],
                      label=nm, pred=names[int(sal["pred"][i])])
        print(f"Saved saliency render {r}: test object #{i} ({nm})")

    # ---- fidelity (deletion) test ---------------------------------------------
    idx = torch.where(ok)[0][:200]
    chunks = []
    for j in range(0, len(idx), 25):
        ii = idx[j:j + 25]
        chunks.append(deletion_test(model, sal["coords"][ii].to(device), imp[ii].to(device)))
    res = {"fractions": chunks[0]["fractions"],
           "salient": np.mean([c["salient"] for c in chunks], 0).tolist(),
           "random": np.mean([c["random"] for c in chunks], 0).tolist()}
    plot_deletion_curve(os.path.join(args.out, "deletion_test.png"), res)
    metrics["deletion_test"] = res
    print("Deletion test (prediction unchanged):")
    for f, s_, r_ in zip(res["fractions"], res["salient"], res["random"]):
        print(f"  remove {f * 100:4.0f}%  salient-first {s_:.3f}  random {r_:.3f}")

    with open(os.path.join(args.out, f"metrics_{args.model}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    table = (f"| Model | Overall accuracy | Mean per-class accuracy | Inference time / object |\n"
             f"|---|---|---|---|\n"
             f"| {args.model.upper()} | {overall * 100:.2f}% | {mean_cls * 100:.2f}% | "
             f"{ms:.2f} ms ({device.type.upper()}) |")
    open(os.path.join(args.out, f"metrics_{args.model}.md"), "w").write(table + "\n")
    if not args.no_readme and args.model == "dgcnn":
        update_readme(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), table)


if __name__ == "__main__":
    main()
