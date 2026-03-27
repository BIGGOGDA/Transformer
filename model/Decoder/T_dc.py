from ..modules import *

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.norm_3 = NormLayer(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)


    def forward(self, x, e_outputs, src_mask, trg_mask):
        shortcut = x
        x2 = self.norm_1(x)
        x = shortcut + self.dropout_1(self.attn_1(x2,x2,x2,trg_mask))
        x2 = self.norm_2(x)
        x = shortcut + self.dropout_2(self.attn_2(x2,e_outputs,e_outputs,src_mask))
        x2 = self.norm_3(x)
        x = shortcut + self.dropout_3(self.ff(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = NormLayer(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)