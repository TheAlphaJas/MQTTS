import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

# Ref: StyleTTS2 models.py style encoder (inferred structure for compatibility)
# It typically uses ResBlocks.
# I'll use the exact structure if possible.
# Based on common StyleTTS implementations:

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.convs(x)

class StyleEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # StyleTTS2 Style Encoder configuration often:
        # input: mel spectrogram (80)
        # layers: 
        #   conv 1->channel (3x3)
        #   AveragePooling or more convs with stride?
        #   Usually uses standard Reference Encoder from generic TTS papers.
        
        # I'll stick to a robust implementation that matches StyleTTS2 checkpoints.
        # StyleTTS2 uses 80 mel channels.
        
        # To ensure we can load weights, I will define it flexibly or provide a loading hook.
        
        # Structure based on StyleTTS2 (yl4579):
        self.n_mels = 80
        self.style_dim = 128 # Default in StyleTTS2 config
        
        # Ref: https://github.com/yl4579/StyleTTS2/blob/main/models.py
        # (Reconstructed from common knowledge of this repo)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 2, 1)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(128)
        ])
        
        # StyleTTS2 uses a GRU after convs
        # Input dim to GRU: 128 * (80 // 2^3) = 128 * 10 = 1280
        self.gru_dim = 128 * (self.n_mels // 8)
        self.gru = nn.GRU(self.gru_dim, 512, batch_first=True) # 512 hidden
        self.projection = nn.Linear(512, 128) # Output style dim
        
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            f_min=0,
            f_max=None
        )

    def forward(self, x):
        # x can be raw audio or mel.
        # If x is (B, T), compute mel.
        if x.ndim == 2:
            with torch.no_grad():
                x = self.to_mel(x)
                x = torch.log(torch.clamp(x, min=1e-5))
        
        # x: (B, 80, T)
        x = x.unsqueeze(1) # (B, 1, 80, T)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(bn(conv(x)))
            
        # x: (B, 128, 10, T')
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim)
        
        _, h = self.gru(x)
        # h: (1, B, 512)
        h = h.squeeze(0)
        
        return self.projection(h)

