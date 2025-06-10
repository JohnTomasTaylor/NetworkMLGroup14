import torch
import torch.nn as nn
import torch.nn.functional as F


def build_dyn_adj_batch(x_frame: torch.Tensor, threshold: float = 0.8, eps: float = 1e-6):
    """
    Build a batch of dynamically normalized adjacency matrices: D^{-1/2} A D^{-1/2}

    Args:
        x_frame: Tensor of shape [B, C, F] — B samples, each with C channels and F features
        threshold: Correlation threshold below which edges are removed
        eps: Small epsilon to prevent division by zero

    Returns:
        adj_norm: Tensor of shape [B, C, C], normalized adjacency matrices with self-loops
    """
    B, C, F = x_frame.shape
    device = x_frame.device

    # 1) Zero-mean and unit-variance normalization per channel
    mean = x_frame.mean(dim=2, keepdim=True)               # [B, C, 1]
    x_centered = x_frame - mean                            # [B, C, F]
    std = x_centered.norm(dim=2, keepdim=True) / (F ** 0.5)  # [B, C, 1]
    std = std.clamp_min(eps)
    x_norm = x_centered / std                               # [B, C, F]

    # 2) Compute Pearson correlation coefficient: PCC = (X_norm @ X_norm.T) / F
    corr = torch.bmm(x_norm, x_norm.transpose(1, 2)) / F     # [B, C, C]
    adj = corr.abs()

    # 3) Apply threshold and add small self-loops to ensure connectivity
    mask = (adj > threshold).float()                         # [B, C, C]
    eye = eps * torch.eye(C, device=device).unsqueeze(0)     # [1, C, C] → broadcast to [B, C, C]
    adj = mask * adj + eye                                   # [B, C, C]

    # 4) Normalize: D^{-1/2} A D^{-1/2}
    deg = adj.sum(dim=2)                                      # [B, C]
    deg_inv_sqrt = deg.pow(-0.5).clamp_min(eps)               # [B, C]
    D_inv_sqrt = deg_inv_sqrt.unsqueeze(2) * deg_inv_sqrt.unsqueeze(1)  # [B, C, C]
    adj_norm = D_inv_sqrt * adj                                # [B, C, C]

    return adj_norm  # [B, C, C]


class SingleScale_DenseDynGCN_LSTM(nn.Module):
    """
    Dense GCN + LSTM module for a single scale (batch-wise).
    Input: x of shape [B, C, T, F]
    Output: repr of shape [B, D]
    """

    def __init__(
        self,
        in_feat: int,
        gcn_hidden: int = 128,
        lstm_hidden: int = 128,
        D: int = 128,
        fixed_adj_norm: torch.Tensor = None,
        num_channels: int = 19,
    ):
        super().__init__()
        self.C = num_channels
        self.fixed_adj_norm = fixed_adj_norm  # [C, C], pre-normalized fixed adjacency matrix

        # Two-layer Dense GCN for dynamic graphs
        self.lin_dyn1 = nn.Linear(in_feat, gcn_hidden)
        self.lin_dyn2 = nn.Linear(gcn_hidden, gcn_hidden // 2)

        # Two-layer Dense GCN for fixed (physical) graph if provided
        if fixed_adj_norm is not None:
            self.lin_knn1 = nn.Linear(in_feat, gcn_hidden)
            self.lin_knn2 = nn.Linear(gcn_hidden, gcn_hidden // 2)
        else:
            self.lin_knn1 = None
            self.lin_knn2 = None

        # Channel attention: maps per-node features [C, 2*(gcn_hidden//2)] → [C, 1]
        self.attn_c = nn.Linear(gcn_hidden, 1)
        # Projects pooled feature from 128 → 64
        self.fuse_proj = nn.Linear(gcn_hidden, gcn_hidden // 2)

        # BiLSTM + temporal attention
        self.lstm = nn.LSTM(
            input_size=gcn_hidden // 2,
            hidden_size=lstm_hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_t = nn.Linear(lstm_hidden, 1)

        # Final projection to D dimensions
        self.proj = nn.Linear(lstm_hidden, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor [B, C, T, F]
        Returns: repr [B, D]
        """
        B, C, T, Feture = x.shape
        assert C == self.C, f"Expected {self.C} channels, got {C}"
        device = x.device

        # 1) Flatten time dimension to form [B*T, C, F]
        x_all = x.permute(0, 2, 1, 3).reshape(B * T, C, Feture)  # [B*T, C, F]

        # 2) Compute normalized dynamic adjacency matrix [B*T, C, C]
        adj_norm_dyn = build_dyn_adj_batch(x_all, threshold=0.8)  # [B*T, C, C]

        # 3) Dynamic graph GCN branch
        # 3.1 First layer
        h1 = self.lin_dyn1(x_all)               # [B*T, C, gcn_hidden]
        h1 = F.relu(h1)
        h1 = torch.bmm(adj_norm_dyn, h1)        # [B*T, C, gcn_hidden]
        h1 = F.relu(h1)
        # 3.2 Second layer
        h2 = self.lin_dyn2(h1)                  # [B*T, C, gcn_hidden//2]
        h_dyn = torch.bmm(adj_norm_dyn, h2)     # [B*T, C, gcn_hidden//2]
        h_dyn = F.relu(h_dyn)                   # [B*T, C, gcn_hidden//2]

        # 4) Fixed graph GCN branch (if fixed_adj_norm is provided)
        if self.fixed_adj_norm is not None:
            # fixed_adj_norm: [C, C]
            k1 = self.lin_knn1(x_all)           # [B*T, C, gcn_hidden]
            k1 = F.relu(k1)
            # correct shape for einsum: [C, C] @ [B*T, C, gcn_hidden
            h_knn1 = torch.einsum('ij,bjk->bik', self.fixed_adj_norm, k1)  
            h_knn1 = F.relu(h_knn1)

            k2 = self.lin_knn2(h_knn1)          # [B*T, C, gcn_hidden//2]
            k2 = F.relu(k2)
            h_knn = torch.einsum('ij,bjk->bik', self.fixed_adj_norm, k2)   # [B*T, C, gcn_hidden//2]
            h_knn = F.relu(h_knn)
        else:
            # Else: use zero tensor as placeholder
            h_knn = torch.zeros_like(h_dyn)      # [B*T, C, gcn_hidden//2]

        # 5) Concatenate dynamic + fixed outputs → [B*T, C, gcn_hidden]
        h_cat = torch.cat([h_dyn, h_knn], dim=2)  # [B*T, C, gcn_hidden]

        # 6) Channel-wise attention (batched)
        attn_c = self.attn_c(h_cat)               # [B*T, C, 1]
        w_c = torch.softmax(attn_c, dim=1)        # [B*T, C, 1]
        pooled_BT = (w_c * h_cat).sum(dim=1)       # [B*T, gcn_hidden]

        # 7) Project back to gcn_hidden // 2 → [B*T, gcn_hidden//2]
        h_feat_BT = self.fuse_proj(pooled_BT)      # [B*T, gcn_hidden//2]

        # 8) reshape → [B, T, gcn_hidden//2]
        seq_feats = h_feat_BT.view(B, T, -1)       # [B, T, gcn_hidden//2]

        # 9) LSTM + temporal attention
        lstm_out, _ = self.lstm(seq_feats)         # [B, T, lstm_hidden]
        attn_t = torch.softmax(self.attn_t(lstm_out), dim=1)  # [B, T, 1]
        context = (attn_t * lstm_out).sum(dim=1)    # [B, lstm_hidden]

        # 10) Final projection to D dimensions
        repr = self.proj(context)                  # [B, D]
        return repr


class MultiScale_GCN_LSTM(nn.Module):
    """
    Multi-scale GCN-LSTM model with attention-based fusion across 3 temporal resolutions.
    """

    def __init__(
        self,
        in_feat_05: int,
        in_feat_1: int,
        in_feat_2: int,
        gcn_hidden: int = 128,
        lstm_hidden: int = 128,
        D: int = 128,
        num_classes: int = 2,
        share_weights: bool = False,
        fixed_adj_norm: torch.Tensor = None,
        num_channels: int = 19,
    ):
        super().__init__()
        self.share_weights = share_weights

        if share_weights:
            # Use the same sub-network for all three scales
            shared = SingleScale_DenseDynGCN_LSTM(
                in_feat=in_feat_05,
                gcn_hidden=gcn_hidden,
                lstm_hidden=lstm_hidden,
                D=D,
                fixed_adj_norm=fixed_adj_norm,
                num_channels=num_channels,
            )
            self.branch_05 = shared
            self.branch_1 = shared
            self.branch_2 = shared
        else:
            # Independent branches for each scale
            self.branch_05 = SingleScale_DenseDynGCN_LSTM(
                in_feat=in_feat_05,
                gcn_hidden=gcn_hidden,
                lstm_hidden=lstm_hidden,
                D=D,
                fixed_adj_norm=fixed_adj_norm,
                num_channels=num_channels,
            )
            self.branch_1 = SingleScale_DenseDynGCN_LSTM(
                in_feat=in_feat_1,
                gcn_hidden=gcn_hidden,
                lstm_hidden=lstm_hidden,
                D=D,
                fixed_adj_norm=fixed_adj_norm,
                num_channels=num_channels,
            )
            self.branch_2 = SingleScale_DenseDynGCN_LSTM(
                in_feat=in_feat_2,
                gcn_hidden=gcn_hidden,
                lstm_hidden=lstm_hidden,
                D=D,
                fixed_adj_norm=fixed_adj_norm,
                num_channels=num_channels,
            )

        # Cross-scale attention fusion: input [B, D] × 3 → [B, 3, D]
        self.fuse_attn = nn.Linear(D, 1)
        # Final classifier
        self.classifier = nn.Linear(D, num_classes)

    def forward(self, x05: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x05: Input at 0.5s scale, shape [B, C, T05, F05]
            x1:  Input at 1s scale, shape [B, C, T1, F1]
            x2:  Input at 2s scale, shape [B, C, T2, F2]

        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        r05 = self.branch_05(x05)  # [B, D]
        r1 = self.branch_1(x1)     # [B, D]
        r2 = self.branch_2(x2)     # [B, D]

        # Stack three branch outputs: [B, 3, D]
        stacked = torch.stack([r05, r1, r2], dim=1)  # [B, 3, D]
        # Apply attention across scales: [B, 3, 1]
        attn_scores = self.fuse_attn(stacked)       # [B, 3, 1]
        attn_w = torch.softmax(attn_scores, dim=1)  # [B, 3, 1]
        # Weighted sum fusion: [B, D]
        fused = torch.sum(attn_w * stacked, dim=1)  # [B, D]
        # Final classification
        logits = self.classifier(fused)             # [B, num_classes]
        return logits
