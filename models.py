"""
models.py — SAR (Self-Attention RNN) Phase Picker Architecture
==============================================================
Salton Sea Geothermal Region — Standalone Version
AI-PAL: Zhou et al. (2025), JGR Solid Earth
doi: 10.1029/2025JB031294

Architecture:
    Input → Bi-directional GRU → Multihead Self-Attention → Linear + Softmax → Output

Config (Salton Sea):
    Window length  : 30s      (S-P times up to ~15s in geothermal field)
    Frequency band : 2–45 Hz  (avoids geothermal pump noise < 2 Hz)
    RNN steps      : 296      (= (30 - 0.5) / 0.1 + 1)
    Sample rate    : 100 Hz
"""

import torch
import torch.nn as nn

# ── Model Config ──────────────────────────────────────────────────────────────
SAMP_RATE     = 100       # Hz
WIN_LEN       = 30        # seconds
NUM_CHN       = 3         # components (E, N, Z)
STEP_LEN      = 0.5       # seconds per RNN frame
STEP_STRIDE   = 0.1       # seconds between frames
NUM_STEPS     = int((WIN_LEN - STEP_LEN) / STEP_STRIDE) + 1   # 296
INPUT_SIZE    = int(STEP_LEN * SAMP_RATE) * NUM_CHN            # 150
RNN_HIDDEN    = 128
RNN_LAYERS    = 2
NUM_ATT_HEADS = 4
FREQ_BAND     = [2, 45]   # Hz
WIN_STRIDE    = 15        # seconds — sliding window stride during picking


# ── SAR Model ─────────────────────────────────────────────────────────────────

class SAR(nn.Module):
    """
    Self-Attention RNN for seismic phase picking.

    Input  : (batch, NUM_STEPS, INPUT_SIZE) — framed 3-component waveform
    Output : (batch, NUM_STEPS, 3)          — per-frame [Noise, P, S] logits

    Architecture:
        1. Bi-directional GRU   — sequential context (forward + backward)
           2 layers, hidden=128, bidirectional → 256-dim output per step
        2. Multihead Attention  — long-range P↔S dependency
           4 heads, embed_dim=256, self-attention (Q=K=V)
        3. Linear + (Softmax applied externally during inference)
           256 → 3 classes: 0=Noise, 1=P-wave, 2=S-wave
    """
    def __init__(self):
        super(SAR, self).__init__()

        # Bi-directional GRU
        self.gru = nn.GRU(
            input_size  = INPUT_SIZE,
            hidden_size = RNN_HIDDEN,
            num_layers  = RNN_LAYERS,
            bidirectional = True,
            batch_first = True
        )

        # Multihead Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim  = 2 * RNN_HIDDEN,   # 256 = 2 × 128
            num_heads  = NUM_ATT_HEADS,     # 4
            batch_first = True
        )

        # Classifier
        self.fc = nn.Linear(2 * RNN_HIDDEN, 3)

    def forward(self, x):
        """
        x: (batch, NUM_STEPS, INPUT_SIZE)
        returns: (batch, NUM_STEPS, 3) logits
        """
        x, _ = self.gru(x)                        # BiGRU
        x, _ = self.attention(x, x, x)            # Self-attention (Q=K=V)
        return self.fc(x)                          # Linear classifier


def load_model(ckpt_path, device=None):
    """
    Load trained SAR model from checkpoint.

    Args:
        ckpt_path : path to .ckpt file (e.g. '8700_17-319.ckpt')
        device    : torch.device — auto-detected if None

    Returns:
        model : SAR model in eval mode
        device: device being used
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f'Device: CUDA ({torch.cuda.get_device_name(0)})')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Device: Apple MPS')
        else:
            device = torch.device('cpu')
            print('Device: CPU')

    model = SAR().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f'Model loaded: {ckpt_path}')
    print(f'Parameters : {sum(p.numel() for p in model.parameters()):,}')
    return model, device


if __name__ == '__main__':
    # Quick test — load model and print summary
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else '8700_17-319.ckpt'
    model, device = load_model(ckpt)
    # Test forward pass
    x = torch.randn(2, NUM_STEPS, INPUT_SIZE).to(device)
    out = model(x)
    print(f'Input shape : {x.shape}')
    print(f'Output shape: {out.shape}')
    print('Model OK ✅')
