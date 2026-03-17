"""
lstm_model.py
=============
LSTM with:
  - MC-Dropout (dropout layers that stay active during inference for uncertainty)
  - Optional personalization via user embedding
  - Proper output head matching 3-step risk horizon
"""

try:
    import torch
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(
            self,
            input_size:  int = 6,
            hidden_size: int = 64,
            num_layers:  int = 2,
            num_users:   int = 768,
            dropout:     float = 0.20,
            output_dim:  int = 3,
        ):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # MC-Dropout: explicit dropout applied after LSTM output
            self.mc_dropout = nn.Dropout(p=dropout)

            self.user_embedding = nn.Embedding(num_users, 16)

            self.fc = nn.Sequential(
                nn.Linear(hidden_size + 16, 64),
                nn.ReLU(),
                nn.Dropout(p=dropout),          # second MC-Dropout layer
                nn.Linear(64, output_dim),
                nn.Sigmoid(),
            )

        def forward(self, x, user_ids):
            _, (h_n, _) = self.lstm(x)
            h = self.mc_dropout(h_n[-1])
            u = self.user_embedding(user_ids)
            z = torch.cat([h, u], dim=1)
            return self.fc(z)

        def enable_mc_dropout(self):
            """Call before uncertainty estimation to keep dropout active."""
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

except ImportError:
    # Fallback stub so the rest of the codebase can import without torch
    class LSTMModel:
        def __init__(self, **kwargs):
            raise ImportError("PyTorch is required for LSTMModel.")
