# lstm_model.py - LSTM 與 Transformer 股價預測模型定義

import math
import torch
import torch.nn as nn


# ── LSTM 模型 ──────────────────────────────────────────────────────────────────

class StockLSTM(nn.Module):
    """雙層 LSTM 股價預測模型，輸入多特徵時序，輸出下一日收盤價。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播：x shape (batch, seq_len, features) -> (batch, 1)。"""
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)            # (batch, seq_len, hidden_size)
        out     = self.dropout(out[:, -1, :])  # 取最後一個 timestep
        out     = self.fc(out)           # (batch, output_size)
        return out


# ── Transformer 模型 ───────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    """Sinusoidal 位置編碼，注入序列位置資訊。"""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                     # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """將位置編碼加到輸入嵌入上。"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class StockTransformer(nn.Module):
    """Transformer Encoder 股價預測模型，含 sinusoidal 位置編碼。"""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_encoder_layers,
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播：x shape (batch, seq_len, features) -> (batch, 1)。"""
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.pos_enc(x)              # 加位置編碼
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        x = x[:, -1, :]                  # 取最後一個 timestep
        x = self.fc(x)                   # (batch, output_size)
        return x


# ── 模型工廠 ───────────────────────────────────────────────────────────────────

def get_model(model_type: str, input_size: int, config, output_size: int = 1) -> nn.Module:
    """依 model_type 建立並回傳對應的預測模型實例（'lstm' 或 'transformer'）。

    output_size: 輸出維度。1=單點預測（預設）；3=分位數迴歸（q10/q50/q90）。
    """
    model_type = model_type.lower()
    if model_type == "lstm":
        return StockLSTM(
            input_size  = input_size,
            hidden_size = config.HIDDEN_SIZE,
            num_layers  = config.NUM_LAYERS,
            dropout     = config.DROPOUT,
            output_size = output_size,
        )
    elif model_type == "transformer":
        return StockTransformer(
            input_size  = input_size,
            dropout     = config.DROPOUT,
            output_size = output_size,
        )
    else:
        raise ValueError(f"未知的模型類型：{model_type}，請選擇 'lstm' 或 'transformer'。")


# ── 快速測試 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as cfg

    batch, seq_len, features = 8, 20, 16
    dummy = torch.randn(batch, seq_len, features)

    lstm_model = get_model("lstm", features, cfg)
    tf_model   = get_model("transformer", features, cfg)

    print("StockLSTM        output:", lstm_model(dummy).shape)
    print("StockTransformer output:", tf_model(dummy).shape)

    total_lstm = sum(p.numel() for p in lstm_model.parameters())
    total_tf   = sum(p.numel() for p in tf_model.parameters())
    print(f"LSTM 參數量：{total_lstm:,}　Transformer 參數量：{total_tf:,}")
