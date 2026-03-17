"""
transformer_model.py
====================
Transformer with:
  - MC-Dropout on all feedforward layers (for uncertainty estimation)
  - Optional personalization via user embedding (ablation-ready)
  - enable_mc_dropout() helper for inference-time uncertainty
"""

try:
    import torch
    import torch.nn as nn

    class TransformerModel(nn.Module):
        def __init__(
            self,
            input_dim:          int   = 6,
            num_users:          int   = 768,
            use_personalization: bool = True,
            seq_len:            int   = 30,
            d_model:            int   = 64,
            nhead:              int   = 4,
            num_layers:         int   = 2,
            dim_feedforward:    int   = 128,
            dropout:            float = 0.20,
            output_dim:         int   = 3,
        ):
            super().__init__()

            self.use_personalization = use_personalization
            self.seq_len = seq_len

            # Feature projection
            self.embedding = nn.Linear(input_dim, d_model)

            # Learnable positional encoding
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # MC-Dropout after pooling
            self.mc_dropout = nn.Dropout(p=dropout)

            # Personalization embedding
            if use_personalization:
                self.user_embedding = nn.Embedding(num_users, 16)
                fc_in = d_model + 16
            else:
                fc_in = d_model

            # Prediction head
            self.fc = nn.Sequential(
                nn.Linear(fc_in, 64),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(64, output_dim),
                nn.Sigmoid(),
            )

        def forward(self, x, user_ids):
            # x: (B, T, input_dim)
            x = self.embedding(x)
            x = x + self.pos_encoder[:, :x.size(1), :]
            x = self.transformer(x)
            h = self.mc_dropout(x.mean(dim=1))   # temporal mean pooling

            if self.use_personalization:
                u = self.user_embedding(user_ids)
                z = torch.cat([h, u], dim=1)
            else:
                z = h

            return self.fc(z)

        def enable_mc_dropout(self):
            """Keep dropout active during inference for MC-Dropout uncertainty."""
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

except ImportError:
    class TransformerModel:
        def __init__(self, **kwargs):
            raise ImportError("PyTorch is required for TransformerModel.")
