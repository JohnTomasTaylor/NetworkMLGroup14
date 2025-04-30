from torch.nn import TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn as nn
import math
import torch

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    def __init__(self, d_model, max_seq_length=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, feature_dim]
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EEGTransformer(nn.Module):
    """
    Transformer model for EEG binary classification.
    """
    def __init__(
        self,
        input_dim=19,           # Number of EEG channels/features
        d_model=64,             # Embedding dimension
        nhead=8,                # Number of attention heads
        num_encoder_layers=4,   # Number of transformer encoder layers
        dim_feedforward=256,    # Dimension of feedforward network
        dropout=0.1,            # Dropout rate
        max_seq_length=150,     # Maximum sequence length
    ):
        super(EEGTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Optional mask for padding tokens
        Returns:
            Binary classification output
        """
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x