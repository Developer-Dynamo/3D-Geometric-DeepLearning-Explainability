"""Module 3 - Geometric explainability via gradient saliency.

Importance of point p = || d score_c / d (x_p, y_p, z_p) ||_2 where c is the
predicted class and score_c its pre-softmax logit. Scores are min-max
normalised to [0, 1] per object.
"""
import torch


def _minmax(s):
    lo = s.min(dim=-1, keepdim=True).values
    hi = s.max(dim=-1, keepdim=True).values
    return (s - lo) / (hi - lo).clamp(min=1e-12)


def compute_saliency(model, pos, target=None):
    """pos: (B, N, 3). Returns dict with coords, importance (B, N) in [0,1], pred, probs.

    Model is put in eval mode (BatchNorm / Dropout frozen) so each object's
    gradient is independent of the others in the batch.
    """
    was_training = model.training
    model.eval()
    pos = pos.detach().clone().requires_grad_(True)       # gradient on input coordinates
    logits = model(pos)
    probs = logits.softmax(-1).detach()
    pred = logits.argmax(-1) if target is None else target
    score = logits.gather(1, pred.view(-1, 1)).sum()      # sum of independent per-object scores
    grad, = torch.autograd.grad(score, pos)               # (B, N, 3): d score / d (X, Y, Z)
    importance = _minmax(grad.norm(dim=-1))               # Euclidean norm per point
    model.train(was_training)
    return {"coords": pos.detach().cpu(), "importance": importance.detach().cpu(),
            "pred": pred.detach().cpu(), "probs": probs.cpu(), "grad": grad.detach().cpu()}


def smooth_importance(coords, importance, k=16):
    """Optional: average scores over each point's k nearest neighbours (denoises the map)."""
    idx = torch.cdist(coords, coords).topk(k, dim=-1, largest=False).indices
    sm = torch.gather(importance.unsqueeze(1).expand(-1, coords.shape[1], -1), 2, idx).mean(-1)
    return _minmax(sm)


def save_saliency(path, coords, importance, label=None, pred=None):
    """Save coordinates + importance scores (for later visualisation)."""
    torch.save({"coords": coords, "importance": importance, "label": label, "pred": pred}, path)


@torch.no_grad()
def deletion_test(model, pos, importance, fractions=(0.0, 0.1, 0.2, 0.3, 0.5), seed=0):
    """Fidelity check: accuracy-vs-original-prediction after removing the most
    salient points vs the same number of random points.

    If the saliency is meaningful, removing salient points should hurt more.
    Returns {"fractions", "salient", "random"} - fraction of objects whose
    predicted class stays unchanged.
    """
    model.eval()
    B, N, _ = pos.shape
    base = model(pos).argmax(-1)
    g = torch.Generator().manual_seed(seed)
    order_sal = importance.argsort(dim=-1, descending=True)
    order_rnd = torch.stack([torch.randperm(N, generator=g) for _ in range(B)])
    res = {"fractions": list(fractions), "salient": [], "random": []}
    for f in fractions:
        n_keep = N - int(round(f * N))
        for name, order in (("salient", order_sal), ("random", order_rnd)):
            keep = order[:, N - n_keep:] if name == "salient" else order[:, :n_keep]
            kept = torch.gather(pos, 1, keep.to(pos.device).unsqueeze(-1).expand(-1, -1, 3))
            res[name].append((model(kept).argmax(-1) == base).float().mean().item())
    return res
