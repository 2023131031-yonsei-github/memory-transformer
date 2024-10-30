import torch
import torch.nn.functional as F
from torch import nn

import math
from typing import Optional, Tuple

from dataclasses import dataclass

@dataclass
class MemoryArgs:
    memory_size: int = 16
    dim: int = 2048
    hdim: int = 4096
    # mem_cycle_len: int = 32
    summary_len: int = 8
    stride: int = 8

class ConstQAtt(nn.Module):
    def __init__(self, num_tokens: int, dim: int, hdim: int):
        super().__init__()
        self.dim = dim
        self.hdim = hdim

        self.wk = nn.Linear(dim, hdim, bias=False)
        self.wv = nn.Linear(dim, hdim, bias=False)

        self.q = nn.Parameter(
            torch.normal(
                torch.zeros((num_tokens, hdim)), 
                torch.ones((num_tokens, hdim)),
            )
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xk, xv = self.wk(x), self.wv(x) # (bs, seqlen, hdim)
        scores = torch.matmul(self.q, xk.transpose(1,2)) / math.sqrt(self.hdim) # (bs, num_tokens, seqlen)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(self.q)
        output = torch.matmul(scores, xv) # (bs, num_tokens, hdim)
        return output

class ConstKVAtt(nn.Module):
    def __init__(self, num_kv, dim, hdim):
        super().__init__()
        self.dim = dim
        self.hdim = hdim
        
        self.wq = nn.Linear(dim, hdim, bias=False)

        self.k = nn.Parameter(
            torch.normal(
                torch.zeros((num_kv, hdim)), 
                torch.ones((num_kv, hdim)),
            )
        )
        self.v = nn.Parameter(
            torch.normal(
                torch.zeros((num_kv, hdim)), 
                torch.ones((num_kv, hdim)),
            )
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x) # (bs, seqlen, hdim)
        scores = torch.matmul(xq, self.k.transpose(1,2)) / self.hdim # (bs, seqlen, num_kv)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, self.v) #(bs, seqlen, hdim)
        return output

class MultiHeadSelfAtt(nn.Module):
    def __init__(self, dim, num_heads, hdim = None):
        super.__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.hdim = hdim if hdim is not None else dim // num_heads

        self.wq = nn.Linear(dim, num_heads * hdim, bias = False)
        self.wk = nn.Linear(dim, num_heads * hdim, bias = False)
        self.wv = nn.Linear(dim, num_heads * hdim, bias = False)

        self.wo = nn.Linear(num_heads * hdim, dim)

    def forward(
        self,
        x : torch.Tensor,
        mask : Optional[torch.Tensor]
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.hdim).transpose(1,2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.hdim).transpose(1,2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.hdim).transpose(1,2)

        scores = torch.matmul(xq, xk.transpose(2,3)) / math.sqrt(self.hdim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv).transpose(1,2).view(bsz, seqlen, -1)
        
        return self.wo(output)

class SelfAtt(nn.Module):
    def __init__(self, dim, hdim):
        super().__init__()
        self.dim = dim
        self.hdim = hdim

        self.wq = nn.Linear(dim, hdim, bias = False)
        self.wk = nn.Linear(dim, hdim, bias = False)
        self.wv = nn.Linear(dim, dim, bias = False)

    def forward(
        self, 
        x : torch.Tensor,
        mask : Optional[torch.Tensor]
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        scores = torch.matmul(xq, xk.transpose(1,2)) / math.sqrt(self.hdim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, xv)

        return output

class MultiHeadQAtt(nn.Module):
    pass

class MultiHeadKVAtt(nn.Module):
    pass

class Memory(nn.Module):
    def __init__(self, dim, memory_size, memory_dim, summary_len, stride = None):
        super().__init__()
        self.dim = dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.summary_len = summary_len
        self.stride = stride if stride is not None else summary_len

        self.ws = nn.Linear(dim * summary_len, memory_dim, bias = False)
        self.attention = SelfAtt(memory_dim, memory_dim)
        self.q_att = ConstQAtt(memory_size, memory_dim, memory_dim)
        self.memory_norm = nn.RMSNorm((memory_size, memory_dim))
        self.h_norm = nn.RMSNorm((memory_size, memory_dim))

        self.lin_head = nn.Linear(memory_dim, dim * summary_len)

    def forward(
        self, 
        x: torch.Tensor, 
        h: torch.Tensor,
    ):
        #Positional Embedding placed outside of the module
        bsz, seqlen, _ = x.shape
        conv_seqlen = seqlen + self.stride - (seqlen - self.summary_len) % self.stride
        max_summary_len = (((conv_seqlen - self.summary_len) // self.stride ) + 1) * self.summary_len

        seq = torch.zeros((bsz, conv_seqlen, self.dim))
        for k, t in enumerate(x):
            seq[k,:len(t)] = t

        summary = torch.zeros((bsz, max_summary_len, self.dim))
        for i in range(((conv_seqlen - self.summary_len) // self.stride) + 1):
            summary[:, i*self.summary_len:(i+1)*self.summary_len] = seq[:, i*self.stride: i*self.stride + self.summary_len]
        

        summary = summary.view(bsz, max_summary_len // self.summary_len, self.dim * self.summary_len)
        summary = self.ws(summary)
        att_summary = self.attention(summary, mask=None)
        memory = self.q_att(summary, mask=None)
        h = h + self.memory_norm(memory)
        return self.h_norm(h)

    def init_hidden(self, bsz):
        return torch.zeros((bsz, self.memory_size, self.memory_dim))

    def memory_len(self):
        return self.memory_size * self.summary_len

    def read(
        self,
        h: torch.Tensor,
    ):
        bsz, memory_size, _ = h.shape
        return self.lin_head(h).view(bsz, memory_size * self.summary_len, self.dim)

class MultiHeadMemory(nn.Module):
    def __init__(self, max_seq_len, n_heads, dim, memory_size, memory_dim, summary_len, stride = None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.dim = dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.summary_len = summary_len
        self.stride = stride
        
        self.heads = torch.nn.ModuleList()
        for head_id in range(n_heads):
            self.heads.append(Memory(dim, memory_size, memory_dim, summary_len, stride))

        self.pos_embeddings = torch.nn.Parameter(torch.randn(max_seq_len, dim))
        
    def forward(self, x, hs):
        bsz, seqlen, _ = x.shape
        assert seqlen < self.max_seq_len
        emblen, embdim = self.pos_embeddings.shape
        emb = self.pos_embeddings[None, :, :].expand((bsz, emblen, embdim))
        x = x + emb[:, :seqlen, :]
        new_hs = []
        for i in range(len(self.heads)):
            new_hs.append(self.heads[i](x,hs[:,i]))
            i = i + 1
        new_hs = torch.stack(new_hs,dim=1)
        return new_hs
    
    def init_hidden(self, bsz):
        return torch.zeros((bsz, self.n_heads, self.memory_size, self.memory_dim))
    
    def memory_len(self):
        return self.memory_size * self.summary_len

    def read(self, hs):
        bsz, _, _, _ = hs.shape
        out = torch.zeros((bsz, self.memory_size * self.summary_len, self.dim))
        for i in range(len(self.heads)):
            out = out + self.heads[i].read(hs[:,i])
        return out


        
