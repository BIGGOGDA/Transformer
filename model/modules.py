import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model,max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len,d_model)

        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos, i]= math.sin(pos / (10000 ** (2*i / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** (2*(i+1) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1) #获取张量x在第1维度（即第二个维度）上的大小

        x = x + self.pe[:, :seq_len]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads = 8, dropout = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(d_model, d_model)

    def attn(self,q,k,v,d_k,mask = None,dropout=None):
        scores = torch.matmul(q,k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim = -1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask = None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = self.attn(q,k,v,k,mask=mask,dropout=self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(-1, self.num_heads * self.d_k)
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class NormLayer(nn.Module):
    def __init__(self,d_model,eps=1e-6):
        super().__init__()
        self.size = d_model
        # 层归一化包含两个可以学习的参数
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True)+self.eps)+self.bias
        return norm



