from ..modules import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        shortcut = x
        x2 = self.norm_1(x)
        x = shortcut + self.dropout_1(self.attn(x2,x2,x2,mask=mask))
        x2 = self.norm_2(x)
        x = shortcut + self.dropout_2(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(self,vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = NormLayer(d_model)

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

