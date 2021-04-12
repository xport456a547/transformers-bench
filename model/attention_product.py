import torch
import torch.nn as nn
import math
from numpy import prod

try:
    from pykeops.torch import LazyTensor
    #from pykeops import clean_pykeops
    #clean_pykeops() 
except:
    import logging
    logging.info("pykeops not installed")

class BaseAttentionProduct(nn.Module):

    def __init__(self, config, return_attention_probs=False):
        """
        Compute attention: softmax(Q @ K.T) @ V
        """
        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.return_attention_probs = return_attention_probs

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask
            del attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        context_layer = self.dropout(attention_probs) @ value_layer

        if not self.return_attention_probs:
            return context_layer
        return (context_layer, attention_probs)


class BaseAttentionProductWithLSE(nn.Module):

    def __init__(self, config, return_logsumexp=False):
        """
        Compute attention: softmax(Q @ K.T) @ V
        """
        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.return_logsumexp = return_logsumexp

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d) 

        del query_layer
        del key_layer

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        del attention_mask

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        context_layer = self.dropout(attention_probs) @ value_layer

        if not self.return_logsumexp:
            return context_layer
        return (context_layer, torch.logsumexp(attention_scores, dim=-1, keepdim=True))


class KernelAttentionProduct(nn.Module):

    def __init__(self, config, normalize=True):
        """
        Compute attention: Φ(Q) @ (Φ(K).T @ V)
        """
        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.normalize = normalize

    def forward(self, query_layer, key_layer, value_layer, **karwgs):
        
        # Normalize
        if self.normalize:
            query_layer = query_layer / (
                query_layer @ key_layer.sum(-2, keepdim=True).transpose(-1, -2) + 10e-6
            )

        # Apply dropout here because we dont compute a score matrix
        context_layer = query_layer @ self.dropout(key_layer.transpose(-1, -2) @ value_layer)

        return context_layer


class KeopsAttentionProduct(nn.Module):
    """
    Keops attention for full attention and efficient attention with low memory print
    Don't support head mask
    """
    def __init__(self, config):

        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        # Expect (..., t, d) shape for query, key, value
        # Expect (..., t) for mask (default: (n, 1, 1, t))
        d = query_layer.shape[-1]
        t = query_layer.shape[-2]
        sizes = query_layer.size()

        reduction_dim = len(query_layer.size()) - 1

        q = LazyTensor( query_layer.unsqueeze(-2).contiguous() / math.sqrt(d) )
        k = LazyTensor( key_layer.unsqueeze(-3).contiguous() )
        v = LazyTensor( value_layer.unsqueeze(-3).contiguous() )

        attention_scores = (q | k)

        if attention_mask is not None:
            mask = LazyTensor( attention_mask.unsqueeze(-1) )
            attention_scores = attention_scores + mask

        return (attention_scores).sumsoftmaxweight(v, dim=reduction_dim).reshape(*sizes)


class KeopsBlockAttentionProduct(nn.Module):
    """
    Keops attention for full attention and efficient attention with low memory print
    Don't support head mask
    """
    def __init__(self, config, window=None):

        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.local = True if window is not None else False

        self.ranges = None
        self.window = window
        self.offset = window // 2 if window is not None else None

        self.num_attention_heads = config.num_attention_heads

    def set_ranges(self, sizes, device):

        t, d = sizes[-2:]
        batch = prod(sizes[:-1])

        assert t % self.stride == 0

        w = self.stride
        n_blocks = t // w
        batch_blocks = batch // w 

        if self.ranges is None or self.ranges[1].shape[0] != batch//w:
            slices_i = torch.arange(batch_blocks).to(device).int() + 1

            idx_i = torch.arange(n_blocks).to(device).int()
            ranges_i = torch.cat([idx_i.unsqueeze(-1)*w, idx_i.unsqueeze(-1)*w + w], dim=-1).clamp(0, t)
            ranges_i = torch.cat([ranges_i + i*t for i in range(batch//t)])

            ranges_j = torch.cat([idx_i.unsqueeze(-1)*w, idx_i.unsqueeze(-1)*w + self.window_size], dim=-1) - self.offset
            ranges_j = ranges_j.clamp(0, t)
            ranges_j = torch.cat([ranges_j + i*t for i in range(batch//t)])

            self.ranges = (ranges_i, slices_i, ranges_j)*2
        
    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        # Expect (..., t, d) shape for query, key, value
        # Expect (..., t) for mask (default: (n, 1, 1, t))
        d = query_layer.shape[-1]
        t = query_layer.shape[-2]
        sizes = query_layer.size()

        if self.local:
            query_layer = query_layer.reshape(-1, d)
            key_layer = key_layer.reshape(-1, d)
            value_layer = value_layer.reshape(-1, d)
            if attention_mask is not None:
                attention_mask = attention_mask.expand(*sizes[:-2], 1, -1).reshape(1, -1)

        reduction_dim = len(query_layer.size()) - 1

        # Compute dropout here as it cannot be done on the score matrix
        query_layer = self.dropout(query_layer)
        key_layer = self.dropout(key_layer)

        q = LazyTensor( query_layer.unsqueeze(-2).contiguous() / math.sqrt(d) )
        k = LazyTensor( key_layer.unsqueeze(-3).contiguous() )
        v = LazyTensor( value_layer.unsqueeze(-3).contiguous() )

        attention_scores = (q | k)

        if attention_mask is not None:
            mask = LazyTensor( attention_mask.unsqueeze(-1) )
            attention_scores = attention_scores + mask

        if self.local:
            self.set_ranges(sizes, query_layer.device)
            attention_scores.ranges = self.ranges

        return attention_scores.sumsoftmaxweight(v, dim=reduction_dim).reshape(*sizes)


class BlockAttentionProduct(nn.Module):

    def __init__(self, config):
        """
        Compute block or overlapping blocks attention products
        """
        super().__init__()
 
        self.block_size = config.block_size

        assert config.sequence_len % self.block_size == 0

        self.attention = BaseAttentionProduct(config)

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None, global_key=None, global_value=None, global_mask=None):

        # expect (..., t, d) shape
        initial_shape = query_layer.size()

        key_layer = self.chunk(key_layer)
        value_layer = self.chunk(value_layer)
        attention_mask = self.chunk(attention_mask.transpose(-1, -2)).transpose(-1, -2)

        # Add global tokens
        key_layer = self.add_global_tokens(key_layer, global_key)
        value_layer = self.add_global_tokens(value_layer, global_value)
        attention_mask = self.add_global_tokens(attention_mask, global_mask, dim=-1)

        context_layer = self.attention(
            query_layer=self.chunk(query_layer), 
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask
            )

        return context_layer.reshape(*initial_shape)

    def chunk(self, x):
        t, d = x.size()[-2:]
        return x.reshape(*x.size()[:-2], -1, self.block_size, d)

    def dechunk(self, x, initial_shape):
        return x.reshape(*initial_shape)

    def add_global_tokens(self, x, x_global, dim=-2):
        if x_global is not None:
            n, h, b, t, d = x.size()
            x_global = x_global.unsqueeze(-3).expand(-1, -1, b, -1, -1)
            return torch.cat([x, x_global], dim=dim)
        return x


class BlockLocalAttentionProduct(nn.Module):

    def __init__(self, config, chunk_size=None, return_logsumexp=False):
        """
        Compute block or overlapping blocks attention products
        """
        super().__init__()
 
        self.chunk_size = chunk_size
        self.return_logsumexp = return_logsumexp
        self.circular = config.circular 

        if chunk_size is None:
            self.chunk_size = config.chunk_size

        assert config.sequence_len % self.chunk_size == 0
        assert self.chunk_size % 2 == 0

        self.n_blocks = config.sequence_len // self.chunk_size * 2

        if config.keops and not return_logsumexp:
            self.attention = KeopsAttentionProduct(config) 
        else:
            self.attention = BaseAttentionProductWithLSE(config, return_logsumexp=return_logsumexp)

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None, global_key=None, global_value=None, global_mask=None):
        
        # Input batch, heads, length, hidden_size
        n, h, t, d = query_layer.size()
        query_layer = self.chunk(query_layer)
        
        if self.circular:
            key_layer, value_layer, attention_mask = self.reshape_inputs_circular(key_layer, value_layer, attention_mask)
        else:
            key_layer, value_layer, attention_mask = self.reshape_inputs_with_pad(key_layer, value_layer, attention_mask)
        
        # Add global tokens
        key_layer = self.add_global_tokens(key_layer, global_key)
        value_layer = self.add_global_tokens(value_layer, global_value)
        attention_mask = self.add_global_tokens(attention_mask, global_mask, dim=-1)
        
        # expect (..., t, d) shape
        if self.return_logsumexp:
            (context_layer, logsumexp) = self.attention(
                query_layer=query_layer, 
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask
                )

            return (context_layer.reshape(n, h, t, d), logsumexp.reshape(n, h, -1, 1))
        
        context_layer = self.attention(
                query_layer=query_layer, 
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask
                )

        return context_layer.reshape(n, h, t, d)
    
    def reshape_inputs_circular(self, key_layer, value_layer, attention_mask):

        w = key_layer.size()[-2] // self.n_blocks * 2

        key_layer = self.build_blocks_circular(key_layer, w)
        value_layer = self.build_blocks_circular(value_layer, w)
        attention_mask = self.build_blocks_circular(attention_mask.transpose(-1, -2), w).transpose(-1, -2)

        return key_layer, value_layer, attention_mask

    def reshape_inputs_with_pad(self, key_layer, value_layer, attention_mask):
        
        n, h, t, d = key_layer.size()
        mask_size = attention_mask.size()
        w = t // self.n_blocks * 2
        
        pad = torch.zeros(n, h, w//2, d, device=key_layer.device)
        pad_mask = torch.zeros(*mask_size[:2], 1, w//2, device=key_layer.device) - 10000.

        key_layer = self.build_blocks_with_pad(key_layer, pad, w, dim=-2)
        value_layer = self.build_blocks_with_pad(value_layer, pad, w, dim=-2)
        attention_mask = self.build_blocks_with_pad(attention_mask, pad_mask, w, dim=-1)

        return key_layer, value_layer, attention_mask

    def build_blocks_circular(self, x, w):
        return torch.cat([x[..., -w//2:, :], x, x[..., :w//2, :]], dim=-2).unfold(-2, int(w*1.5), w//2).transpose(-1, -2)

    def build_blocks_circular_old(self, x, dim=-2):
        return torch.cat([torch.roll(x, -1, dims=-3), x, torch.roll(x, 1, dims=-3)], dim=dim)

    def build_blocks_with_pad(self, x, pad, w, dim=-2):
        if dim != -1:
            return torch.cat([pad, x, pad], dim=dim).unfold(dim, int(w*1.5), w//2).transpose(-1, -2)
        return torch.cat([pad, x, pad], dim=dim).unfold(dim, int(w*1.5), w//2).transpose(-2, -3)

    def add_global_tokens(self, x, x_global, dim=-2):
        if x_global is not None:
            n, h, b, t, d = x.size()
            x_global = x_global.unsqueeze(-3).expand(-1, -1, b, -1, -1)
            return torch.cat([x, x_global], dim=dim)
        return x

    def chunk(self, x):
        t, d = x.size()[-2:]
        return x.reshape(*x.size()[:-2], self.n_blocks, -1, d)


class BlockGlobalAttentionProduct(nn.Module):

    def __init__(self, config, chunk_size=None, topk_chunk_size=None):
        """
        Compute block or overlapping blocks attention products
        """
        super().__init__()
 
        self.chunk_size = chunk_size
        self.topk_chunk_size = topk_chunk_size

        if chunk_size is None:
            self.chunk_size = config.chunk_size

        if topk_chunk_size is None:
            self.topk_chunk_size = 4*config.topk

        self.n_blocks = config.sequence_len // self.chunk_size * 2
        self.topk_n_blocks = config.sequence_len // self.topk_chunk_size * 2

        self.factor = self.n_blocks // self.topk_n_blocks
        self.circular = config.circular

        if config.keops:
            self.attention = KeopsAttentionProduct(config)
        else:
            self.attention = BaseAttentionProduct(config)

        assert config.sequence_len % self.chunk_size == 0
        assert self.chunk_size % 2 == 0

        assert config.sequence_len % self.topk_chunk_size == 0
        assert self.topk_chunk_size % 2 == 0

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None, local_idx=None, global_idx=None, global_key=None, global_value=None, global_mask=None):
        
        # Input batch, ..., heads, length, hidden_size
        n, h, t, d = query_layer.size()
        initial_shape = query_layer.size()

        query_layer = self.chunk(query_layer, self.n_blocks)
        attention_mask = attention_mask.expand(n, h, 1, t)

        key_layer_local, value_layer_local, attention_mask_local = self.build_context(
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask, 
            idx=local_idx, 
            n_blocks=self.n_blocks
            )

        key_layer_global, value_layer_global, attention_mask_global = self.build_context(
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask, 
            idx=global_idx, 
            n_blocks=self.topk_n_blocks
            )

        # Add global tokens
        key_layer_global = self.add_global_tokens(key_layer_global, global_key)
        value_layer_global = self.add_global_tokens(value_layer_global, global_value)
        attention_mask_global = self.add_global_tokens(attention_mask_global, global_mask, dim=-1)

        n, h, b, t, d = query_layer.size()
        query_layer = query_layer.reshape(n, h, -1, self.factor, t, d)
        n, h, b, f, t, d = query_layer.size()

        key_layer = torch.cat([key_layer_local.reshape(n, h, b, f, -1, d), key_layer_global.unsqueeze(-3).expand(n, h, -1, f, -1, d)], dim=-2)
        del key_layer_local
        del key_layer_global

        value_layer = torch.cat([value_layer_local.reshape(n, h, b, f, -1, d), value_layer_global.unsqueeze(-3).expand(n, h, -1, f, -1, d)], dim=-2)
        del value_layer_local
        del value_layer_global

        attention_mask = torch.cat([attention_mask_local.reshape(n, h, b, f, 1, -1), attention_mask_global.unsqueeze(-3).expand(n, h, -1, f, 1, -1)], dim=-1)
        del attention_mask_local
        del attention_mask_global

        # expect (..., t, d) shape
        context_layer = self.attention(
            query_layer=query_layer, 
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask=attention_mask
            )

        context_layer = context_layer.reshape(*initial_shape)
        return context_layer

    def build_context(self, key_layer, value_layer, attention_mask, idx, n_blocks):

        d = key_layer.size()[-1]
        
        attention_mask = attention_mask.gather(dim=-1, index=idx.transpose(-1, -2))
        idx = idx.expand(-1, -1, -1, d)
        key_layer = key_layer.gather(dim=-2, index=idx)
        value_layer = value_layer.gather(dim=-2, index=idx)

        if self.circular:
            return self.reshape_inputs_circular(key_layer, value_layer, attention_mask, n_blocks)
        return self.reshape_inputs_with_pad(key_layer, value_layer, attention_mask, n_blocks)
    
    def reshape_inputs_circular(self, key_layer, value_layer, attention_mask, n_blocks):

        w = key_layer.size()[-2] // n_blocks * 2

        key_layer = self.build_blocks_circular(key_layer, w)
        value_layer = self.build_blocks_circular(value_layer, w)
        attention_mask = self.build_blocks_circular(attention_mask.transpose(-1, -2), w).transpose(-1, -2)

        return key_layer, value_layer, attention_mask

    def reshape_inputs_with_pad(self, key_layer, value_layer, attention_mask, n_blocks):
        
        n, h, t, d = key_layer.size()
        mask_size = attention_mask.size()
        w = t // n_blocks * 2
        
        pad = torch.zeros(n, h, w//2, d, device=key_layer.device)
        pad_mask = torch.zeros(*mask_size[:2], 1, w//2, device=key_layer.device) - 10000.

        key_layer = self.build_blocks_with_pad(key_layer, pad, w, dim=-2)
        value_layer = self.build_blocks_with_pad(value_layer, pad, w, dim=-2)
        attention_mask = self.build_blocks_with_pad(attention_mask, pad_mask, w, dim=-1)

        return key_layer, value_layer, attention_mask

    def build_blocks_circular(self, x, w):
        return torch.cat([x[..., -w//2:, :], x, x[..., :w//2, :]], dim=-2).unfold(-2, int(w*1.5), w//2).transpose(-1, -2)

    def build_blocks_circular_old(self, x, dim=-2):
        return torch.cat([torch.roll(x, -1, dims=-3), x, torch.roll(x, 1, dims=-3)], dim=dim)

    def build_blocks_with_pad(self, x, pad, w, dim=-2):
        if dim != -1:
            return torch.cat([pad, x, pad], dim=dim).unfold(dim, int(w*1.5), w//2).transpose(-1, -2)
        return torch.cat([pad, x, pad], dim=dim).unfold(dim, int(w*1.5), w//2).transpose(-2, -3)

    def add_global_tokens(self, x, x_global, dim=-2):
        if x_global is not None:
            n, h, b, t, d = x.size()
            x_global = x_global.unsqueeze(-3).expand(-1, -1, b, -1, -1)
            return torch.cat([x, x_global], dim=dim)
        return x

    def chunk(self, x, n_blocks):
        t, d = x.size()[-2:]
        return x.reshape(*x.size()[:-2], n_blocks, -1, d)

