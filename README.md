#Edit llama/model.py
#class Attention(nn.Module):
    def clear_cache(self):
        self.cache_k = torch.zeros(self.cache_k.shape)
        self.cache_v = torch.zeros(self.cache_v.shape)

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

#class TransformerBlock(nn.Module):
    def clear_cache(self):
        self.attention.clear_cache()

    def init_cache(self):
        self.attention.init_cache()

#class Transformer(nn.Module):
    def clear_cache(self):
        for layer in self.layers:
            h = layer.clear_cache()
            
    def init_cache(self):
        for layer in self.layers:
            h = layer.init_cache()