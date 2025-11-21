import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchaudio

class DownsampleRes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Stride 2 depthwise conv
        self.conv = spectral_norm(nn.Conv2d(dim, dim, 3, 2, 1, groups=dim))
        
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 2 if downsample else 1, 1))
        
        # Checkpoint indicates conv1x1 has no bias
        self.conv1x1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)) if in_channels != out_channels else None
        
        self.downsample = downsample
        if downsample:
            # DownsampleRes uses in_channels (applied before conv1x1 or on input)
            self.downsample_res = DownsampleRes(in_channels)
            
    def forward(self, x):
        sc = x
        
        # Apply downsample first (on in_channels)
        if self.downsample:
            sc = self.downsample_res(sc)
            
        # Then apply projection if needed
        if self.conv1x1 is not None:
            sc = self.conv1x1(sc)
        
        h = F.leaky_relu(x, 0.2)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
             
        return h + sc

class StyleEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        self.shared = nn.ModuleList()
        
        # Block 0: Conv2d(1, 64)
        self.shared.append(spectral_norm(nn.Conv2d(1, 64, 3, 1, 1)))
        
        # Block 1: 64 -> 128
        self.shared.append(ResBlock(64, 128, downsample=True))
        
        # Block 2: 128 -> 256
        self.shared.append(ResBlock(128, 256, downsample=True))
        
        # Block 3: 256 -> 512
        self.shared.append(ResBlock(256, 512, downsample=True))
        
        # Block 4: 512 -> 512 (Downsample, but no conv1x1 as in=out)
        self.shared.append(ResBlock(512, 512, downsample=True))
        
        # Block 5: Identity (placeholder for missing module.shared.5)
        self.shared.append(nn.Identity())
        
        # Block 6: Conv2d(512, 512, 5, 1, 0)
        self.shared.append(spectral_norm(nn.Conv2d(512, 512, 5, 1, 0)))
        
        # Unshared
        self.unshared = nn.Linear(512, 128)
        
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
        # x: (B, T) audio
        with torch.no_grad():
             x = self.to_mel(x)
             x = torch.log(torch.clamp(x, min=1e-5))
        
        # Check for minimum length (need 80 frames for 4 downsamples + 5x5 conv)
        # 80 / 16 = 5.
        if x.size(-1) < 80:
            pad_size = 80 - x.size(-1)
            x = F.pad(x, (0, pad_size))
            
        # x: (B, 80, T)
        x = x.unsqueeze(1) # (B, 1, 80, T)
        
        for layer in self.shared:
            if isinstance(layer, ResBlock):
                x = layer(x)
            elif isinstance(layer, nn.Identity):
                pass
            else:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                     x = F.leaky_relu(x, 0.2)

        # x: (B, 512, 1, T')
        x = x.flatten(2) # (B, 512, T')
        x = x.mean(dim=2) # Global average pooling over time
        
        x = self.unshared(x)
        return x
