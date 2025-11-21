import torch.nn as nn
from quantizer.env import AttrDict
from quantizer.models import Generator, Quantizer, Encoder
import torch
import json

class Vocoder(nn.Module):
    def __init__(self, config_path, ckpt_path, with_encoder=False):
        super(Vocoder, self).__init__()
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        
        if ckpt_path is not None:
            print(f"Initializing Vocoder from {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            self.generator.load_state_dict(ckpt['generator'])
            self.quantizer.load_state_dict(ckpt['quantizer'])
            if with_encoder:
                self.encoder = Encoder(self.h)
                self.encoder.load_state_dict(ckpt['encoder'])
        else:
            print("Initializing Vocoder with random weights (expecting load later)")
            if with_encoder:
                self.encoder = Encoder(self.h)

    def forward(self, x, spkr):
        return self.generator(self.quantizer.embed(x), spkr)

    def encode(self, x):
        batch_size = x.size(0)
        c = self.encoder(x.unsqueeze(1))
        q, loss_q, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        return torch.stack(c, -1) #N, T, 4
