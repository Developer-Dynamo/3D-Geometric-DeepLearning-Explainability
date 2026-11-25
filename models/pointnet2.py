"""PointNet++ (single-scale grouping) baseline in pure PyTorch.

Set-abstraction levels: farthest point sampling -> k-NN grouping -> shared MLP
-> max pooling. Used as a comparison baseline for DGCNN (`train.py --model pointnet2`).
"""
import torch
import torch.nn as nn


def farthest_point_sample(xyz, npoint):
    """xyz: (B, N, 3) -> indices (B, npoint)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = far
        centroid = xyz[ar, far].unsqueeze(1)
        dist = torch.minimum(dist, ((xyz - centroid) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


def gather(points, idx):
    """points: (B, N, C); idx: (B, S) or (B, S, K) -> gathered points."""
    B = points.shape[0]
    shape = idx.shape
    flat = idx.reshape(B, -1)
    out = torch.gather(points, 1, flat.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return out.reshape(*shape, points.shape[-1])


class SetAbstraction(nn.Module):
    def __init__(self, npoint, k, in_ch, mlp):
        super().__init__()
        self.npoint, self.k = npoint, k
        layers, c = [], in_ch + 3
        for o in mlp:
            layers += [nn.Conv2d(c, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2)]
            c = o
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feats):
        if self.npoint is None:                       # global abstraction
            grouped = xyz if feats is None else torch.cat([xyz, feats], -1)
            x = grouped.permute(0, 2, 1).unsqueeze(-1)           # (B, C, N, 1)
            return None, self.mlp(x).max(2)[0].squeeze(-1)       # (B, C')
        fps = farthest_point_sample(xyz.detach(), self.npoint)
        new_xyz = gather(xyz, fps)                                # (B, S, 3)
        knn = torch.cdist(new_xyz, xyz).topk(self.k, dim=-1, largest=False).indices
        g_xyz = gather(xyz, knn) - new_xyz.unsqueeze(2)           # (B, S, K, 3)
        g = g_xyz if feats is None else torch.cat([g_xyz, gather(feats, knn)], -1)
        out = self.mlp(g.permute(0, 3, 1, 2)).max(-1)[0]          # (B, C', S)
        return new_xyz, out.permute(0, 2, 1)


class PointNet2(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.sa1 = SetAbstraction(512, 32, 0, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 32, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, 256, [256, 512, 1024])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, pos):
        """pos: (B, N, 3) -> logits (B, num_classes)."""
        xyz, f = self.sa1(pos, None)
        xyz, f = self.sa2(xyz, f)
        _, g = self.sa3(xyz, f)
        return self.classifier(g)
