# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops


def build_band_similarity_graph(x: torch.Tensor, threshold=None, normalize=True, eps=1e-6):
    """
    x: [C, T, F] - EEG feature sequence
        Assumes psd_band = x[:,:,133:138], cwt_band = x[:,:,266:271]
    Returns: [C, C] adjacency matrix
    """
    psd_band  = x[:, :, 133:138]     # [C, T, 5]
    cwt_band  = x[:, :, 266:271]     # [C, T, 5]
    band_feat = torch.cat([psd_band, cwt_band], dim=-1)  # [C, T, 10]
    band_mean = band_feat.mean(dim=1)  # [C, 10]

    # Cosine similarity between channels
    normed = F.normalize(band_mean, p=2, dim=1)  # [C, 10]
    adj = (normed @ normed.T).clamp(min=0.0, max=1.0)     # [C, C]

    # Optional: thresholding (if desired)
    if threshold is not None:
        adj = torch.where(adj < threshold, torch.zeros_like(adj), adj)

    # Add self-loops
    adj = adj + eps * torch.eye(adj.size(0), device=x.device)

    # GCN-style normalization
    if normalize:
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5).clamp_min(eps)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt

    return adj




# ─── PCC-based Adjacency Matrix Construction ─────────────────────────────
def build_adj_matrix(
    x: torch.Tensor,
    threshold: float = 0.8,
    eps: float = 1e-6,
    normalize: bool = True
    ) -> torch.Tensor:
    """
    Input: x [C, T, F]
    Output: adj [C, C] (automatically on the same device as x)
    """
    C, T, F = x.shape
    x_flat = x.view(C, -1)                  # [C, T*F]
    x_centered = x_flat - x_flat.mean(1, True)
    std = x_centered.norm(dim=1, keepdim=True) / (x_flat.size(1) ** 0.5)
    std = std.clamp_min(eps)                # Avoid division by zero
    x_norm = x_centered / std               # Zero-mean, unit-variance

    # Pearson correlation coefficient matrix
    corr = (x_norm @ x_norm.T) / x_norm.size(1)
    adj = corr.abs()

    # Thresholding and adding a small self-loop
    adj = torch.where(adj < threshold, 
                      torch.zeros_like(adj), 
                      adj)
    adj = adj + eps * torch.eye(C, device=x.device)

    if normalize:
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5).clamp_min(eps)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt

    return adj


def normalize_adj(adj):
    # adj: [C, C], must be a float-type Tensor
    I = torch.eye(adj.size(0)).to(adj.device)
    adj = adj + I  # Add self-loops
    D = torch.diag(torch.pow(adj.sum(1), -0.5))
    adj_norm = D @ adj @ D
    return adj_norm


# ─── Simple GCN Layer ──────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [C, in_features]
        # adj: [C, C]
        support = self.linear(x)             # shape = [C, out_features]
        out = torch.matmul(adj, support)     # shape = [C, out_features]
        return F.relu(out)


# ─── Main Model: GCN-LSTM ───────────────────────
class GCN_LSTM(nn.Module):
    def __init__(self, in_feat=75, gcn_hidden=128, lstm_hidden=128, num_classes=2):
        super().__init__()
        self.edge_index_knn, _ = add_self_loops(torch.load(r"C:\Users\ROG\Downloads\EEG_nml\edge_index\edge_index_knn.pt"))
        self.gcn_dyn1 = GCNConv(in_feat, gcn_hidden)
        self.dropout_dyn1 = nn.Dropout(p=0.3)
        self.gcn_dyn2 = GCNConv(gcn_hidden, 64)
        self.gcn_knn1 = GCNConv(in_feat, gcn_hidden)
        self.gcn_knn2 = GCNConv(gcn_hidden, 64)

        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=lstm_hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
            # dropout=0.3
        )
        # Channel-wise attention: reduce node features [C, 128] → [C, 1] to get weights
        self.attn_c = nn.Linear(gcn_hidden, 1)
        # Temporal attention: BiLSTM output [B, T, lstm_hidden] → [B, T, 1]
        self.attn_t = nn.Linear(lstm_hidden, 1)
        # Classifier: LSTM output [B, lstm_hidden] → [B, num_classes]
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):  # x: [B, C, T, F]
        B, C, T, _ = x.shape
        device = x.device
        outputs = []

        for b in range(B):
            feats = []  # Store features at each time step
            xb = x[b]                   # [C, T, F]
            xb = xb.to(device)      
            dyn_adj = build_adj_matrix(xb)                     # [C,C]
            edge_index_dyn, _ = dense_to_sparse(dyn_adj)       # [2, E]
            edge_index_dyn, _ = add_self_loops(edge_index_dyn) # Add self-loops

            for t in range(T):
                xt = xb[:, t, :]                     # [C, F]
                h_dyn = F.relu(self.gcn_dyn1(xt, edge_index_dyn))  # [C, gcn_hidden]
                h_dyn = self.dropout_dyn1(h_dyn)     
                h_dyn = F.relu(self.gcn_dyn2(h_dyn, edge_index_dyn))  # [C, 64]

                # KNN graph branch
                edge_index_knn = self.edge_index_knn.to(device)
                h_knn = F.relu(self.gcn_knn1(xt, edge_index_knn))      # [C, gcn_hidden]
                h_knn = F.relu(self.gcn_knn2(h_knn, edge_index_knn))   # [C, 64]

                h = torch.cat([h_dyn, h_knn], dim=1)  # [C, 128]

                # —— Channel-wise Attention Aggregation ——
                weights_c = torch.softmax(self.attn_c(h), dim=0)  # Normalize along channel dimension [C, 1]
                pooled = (weights_c * h).sum(dim=0)               # [128]
                feats.append(pooled)

            seq = torch.stack(feats, dim=0).to(device)   # [T, 128]
            outputs.append(seq)

        x_seq = torch.stack(outputs, dim=0)     # [B, T, 128]
        lstm_out, _ = self.lstm(x_seq)          # [B, T, 128]
        weights_t = torch.softmax(self.attn_t(lstm_out), dim=1)  # Normalize along time dimension
        context = (weights_t * lstm_out).sum(dim=1)               # [B, 128]
        out = self.classifier(context)
        return out
