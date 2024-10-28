import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import VocabParallelEmbedding

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, rank, alpha, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.w_q = torch.randn((args.n_heads * self.head_dim, args.dim)).cuda()
        self.w_k = torch.randn((self.n_kv_heads * self.head_dim, args.dim)).cuda()
        self.w_v = torch.randn((self.n_kv_heads * self.head_dim, args.dim)).cuda()
        self.w_o = torch.randn((args.dim, args.n_heads * self.head_dim)).cuda()

        self.alpha = alpha
        self.rank = rank

        self.lora_wq_a = nn.Linear(args.dim, rank, bias=False)
        self.lora_wq_b = nn.Linear(rank, args.n_heads * self.head_dim, bias=False)
        self.lora_wk_a = nn.Linear(args.dim, rank, bias=False)
        self.lora_wk_b = nn.Linear(rank, self.n_kv_heads * self.head_dim, bias=False)
        self.lora_wv_a = nn.Linear(args.dim, rank, bias=False)
        self.lora_wv_b = nn.Linear(rank, self.n_kv_heads * self.head_dim, bias=False)
        self.lora_wo_a = nn.Linear(args.n_heads * self.head_dim, rank, bias=False)
        self.lora_wo_b = nn.Linear(rank, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def load_weights(self, weights: dict, prefix: str = ""):
        self.w_q = weights[prefix + "wq.weight"].detach().cuda()
        self.w_k = weights[prefix + "wk.weight"].detach().cuda()
        self.w_v = weights[prefix + "wv.weight"].detach().cuda()
        self.w_o = weights[prefix + "wo.weight"].detach().cuda()

    def clear_cache(self):
        self.cache_k = torch.zeros(self.cache_k.shape).cuda()
        self.cache_v = torch.zeros(self.cache_v.shape).cuda()

    def init_cache(self):
        self.cache_k = torch.zeros(
            (
                self.args.max_batch_size,
                self.args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                self.args.max_batch_size,
                self.args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq = F.linear(x, self.w_q) + (self.alpha / self.rank) * self.lora_wq_b(self.lora_wq_a(x))
        xk = F.linear(x, self.w_k) + (self.alpha / self.rank) * self.lora_wk_b(self.lora_wk_a(x))
        xv = F.linear(x, self.w_v) + (self.alpha / self.rank) * self.lora_wv_b(self.lora_wv_a(x))

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return F.linear(output, self.w_o) + (self.alpha / self.rank) * self.lora_wo_b(self.lora_wo_a(x))


class FeedForward(nn.Module):
    def __init__(
        self,
        rank: int,
        alpha,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_1 = torch.randn((hidden_dim, dim)).cuda()
        self.w_2 = torch.randn((dim, hidden_dim)).cuda()
        self.w_3 = torch.randn((hidden_dim, dim)).cuda()

        self.alpha = alpha
        self.rank = rank
        
        self.lora_w1_a = nn.Linear(dim, rank, bias=False)
        self.lora_w1_b = nn.Linear(rank, hidden_dim, bias=False)
        self.lora_w2_a = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_w2_b = nn.Linear(rank, dim, bias=False)
        self.lora_w3_a = nn.Linear(dim, rank, bias=False)
        self.lora_w3_b = nn.Linear(rank, hidden_dim, bias=False)

    def load_weights(self, weights: dict, prefix: str = ""):
        self.w_1 = weights[prefix + "w1.weight"].detach().cuda()
        self.w_2 = weights[prefix + "w2.weight"].detach().cuda()
        self.w_3 = weights[prefix + "w3.weight"].detach().cuda()
        
    def forward(self, x):
        out = F.linear(x, self.w_1) + (self.alpha / self.rank) * self.lora_w1_b(self.lora_w1_a(x))
        out = F.silu(out)
        out = out * (F.linear(x, self.w_3) + (self.alpha / self.rank) * self.lora_w3_b(self.lora_w3_a(x)))
        out = F.linear(out, self.w_2) + (self.alpha / self.rank) * self.lora_w2_b(self.lora_w2_a(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, rank: int, alpha, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(rank, alpha, args)
        self.feed_forward = FeedForward(
            rank,
            alpha,
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def load_weights(self, weights: dict, prefix: str = ""):
        attention_prefix = prefix + "attention."
        feed_forward_prefix = prefix + "feed_forward."

        self.attention.load_weights(weights, attention_prefix)
        self.feed_forward.load_weights(weights, feed_forward_prefix)

    def clear_cache(self):
        self.attention.clear_cache()

    def init_cache(self):
        self.attention.init_cache()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, rank: int, alpha, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, rank, alpha, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.rank = rank
        self.alpha = alpha

        self.w_output = torch.randn(params.vocab_size, params.dim).cuda()
        self.lora_output_a = nn.Linear(params.dim, self.rank, bias=False)
        self.lora_output_b = nn.Linear(self.rank, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        self.tok_embeddings.requires_grad_(False)

    def load_weights(self, weights: dict, prefix: str = ""):
        self.load_state_dict(weights, strict=False)
        
        self.w_output = weights[prefix + "output.weight"].detach().cuda()
        for i in range(len(self.layers)):
            layer_prefix = prefix + "layers." + str(i) + "."
            self.layers[i].load_weights(weights, layer_prefix)
    
    def clear_cache(self):
        for layer in self.layers:
            h = layer.clear_cache()
    
    def init_cache(self):
        for layer in self.layers:
            h = layer.init_cache()
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = F.linear(h, self.w_output) + (self.alpha / self.rank) * self.lora_output_b(self.lora_output_a(h))
        output = output.float()
        return output

def forward_no_embeddings(model, h: torch.Tensor, start_pos: int):
    _bsz, seqlen, _ = h.shape
    model.freqs_cis = model.freqs_cis.to(h.device)
    freqs_cis = model.freqs_cis[start_pos : start_pos + seqlen]

    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)

        mask = torch.triu(mask, diagonal=1)

        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack(
            [torch.zeros((seqlen, start_pos), device=h.device), mask]
        ).type_as(h)

    for layer in model.layers:
        h = layer(h, start_pos, freqs_cis, mask)
    h = model.norm(h)
    output = F.linear(h, model.w_output) + (model.alpha / model.rank) * model.lora_output_b(model.lora_output_a(h))
    output = output.float()
    return output
