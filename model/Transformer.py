import torch
import torch.nn as nn
import torch.nn.functional as F

from Encoder.T_ec import Encoder
from Decoder.T_dc import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, enc_out, src_mask, trg_mask)
        out = self.out(dec_out)
        return out