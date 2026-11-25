"""Module 2 - DGCNN built on PyTorch Geometric (DynamicEdgeConv = dynamic k-NN + EdgeConv, k = 20).

Four DynamicEdgeConv blocks (64, 64, 128, 256 channels) with BatchNorm +
LeakyReLU, global max + mean pooling, MLP classifier with Dropout(0.5).

The k-NN graph is recomputed in the *feature space* of every block (dynamic
graph). We compute the k-NN with plain PyTorch so the model runs without extensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_max_pool, global_mean_pool


def knn_graph_dense(x, k, B, N):
    """x: (B*N, F) -> edge_index (2, B*N*k); row 0 = neighbour j, row 1 = centre i."""
    feats = x.view(B, N, -1)
    dist = torch.cdist(feats, feats)                      # (B, N, N)
    idx = dist.topk(k, dim=-1, largest=False).indices     # (B, N, k) (includes self)
    offset = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
    src = (idx + offset).reshape(-1)
    dst = (torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, k)
           + offset).reshape(-1)
    return torch.stack([src, dst], dim=0)


class DynamicEdgeConvTorch(nn.Module):
    """DynamicEdgeConv = k-NN graph rebuilt in the current feature space + PyG `EdgeConv`.

    This is exactly the definition of `torch_geometric.nn.DynamicEdgeConv`, but the k-NN
    step is plain PyTorch, so it works with every PyG version and needs neither
    `torch-cluster` nor `pyg-lib` (recent PyG releases make those mandatory for the
    built-in class).
    """

    def __init__(self, mlp, k=20, aggr="max"):
        super().__init__()
        self.k = k
        self.conv = EdgeConv(mlp, aggr=aggr)   # message = mlp([x_i, x_j - x_i]), max-aggregated

    def forward(self, x, B, N):
        edge_index = knn_graph_dense(x.detach(), self.k, B, N)
        return self.conv(x, edge_index)


def edge_mlp(in_ch, out_ch):
    # EdgeConv feeds [x_i, x_j - x_i] -> 2 * in_ch inputs
    return nn.Sequential(nn.Linear(2 * in_ch, out_ch), nn.BatchNorm1d(out_ch),
                         nn.LeakyReLU(0.2))


class DGCNN(nn.Module):
    def __init__(self, num_classes=10, k=20, dropout=0.5, in_channels=3,
                 dims=(64, 64, 128, 256), emb_dim=1024):
        super().__init__()
        chans = [in_channels, *dims]
        self.convs = nn.ModuleList(
            [DynamicEdgeConvTorch(edge_mlp(chans[i], chans[i + 1]), k=k, aggr="max")
             for i in range(len(dims))])
        self.lin_emb = nn.Sequential(nn.Linear(sum(dims), emb_dim),
                                     nn.BatchNorm1d(emb_dim), nn.LeakyReLU(0.2))
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, pos):
        """pos: (B, N, 3) -> class logits (B, num_classes). Apply softmax for probabilities."""
        B, N, _ = pos.shape
        x = pos.reshape(B * N, -1)
        outs = []
        for conv in self.convs:
            x = conv(x, B, N)
            outs.append(x)
        x = self.lin_emb(torch.cat(outs, dim=1))
        batch = torch.arange(B, device=pos.device).repeat_interleave(N)
        g = torch.cat([global_max_pool(x, batch, size=B),
                       global_mean_pool(x, batch, size=B)], dim=1)
        return self.classifier(g)
