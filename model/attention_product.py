import torch
import torch.nn as nn
import math
from numpy import prod

try:
    from pykeops.torch import LazyTensor
except:
    import logging
    logging.info("pykeops not installed")

class BaseAttentionProduct(nn.Module):

    def __init__(self, config):
        """
        Compute attention: softmax(Q @ K.T) @ V
        """
        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(d)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = attention_probs @ value_layer

        return context_layer


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
    def __init__(self, config, window=None, stride=1):

        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.local = True if window is not None else False

        self.ranges = None
        self.window = window
        self.stride = stride
        self.offset = window // 2 if window is not None else None

        self.num_attention_heads = config.num_attention_heads

    def set_ranges(self, sizes, device):
        t, d = sizes[-2:]
        batch = prod(sizes[:-1])

        assert t % self.stride == 0
        w = self.stride
        n_blocks = t // w

        if self.ranges is None or self.ranges[1].shape[0] != batch//w:
            slices_i = torch.arange(batch//w).to(device).int() + 1

            idx_i = torch.arange(n_blocks).to(device).int()
            ranges_i = torch.cat([idx_i.unsqueeze(-1)*w, idx_i.unsqueeze(-1)*w + w], dim=-1).clamp(0, t)
            ranges_i = torch.cat([ranges_i + i*t for i in range(batch//t)])

            ranges_j = torch.cat([idx_i.unsqueeze(-1)*w - self.offset + (w % 2), idx_i.unsqueeze(-1)*w + self.offset], dim=-1).clamp(0, t)
            ranges_j = torch.cat([ranges_j + i*t for i in range(batch//t)])

            self.ranges = (ranges_i, slices_i, ranges_j)*2

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        # Expect (..., t, d) shape for query, key, value
        # Expect (..., t) for mask (default: (n, 1, 1, t))
        d = query_layer.shape[-1]
        t = query_layer.shape[-2]
        sizes = query_layer.size()

        if self.local:
            if self.window == self.stride:

                self.local = False
                query_layer = query_layer.reshape(*sizes[:-2], -1, self.window, d)
                key_layer = key_layer.reshape(*sizes[:-2], -1, self.window, d)
                value_layer = value_layer.reshape(*sizes[:-2], -1, self.window, d)
                if attention_mask is not None:
                    attention_mask = attention_mask.reshape(*attention_mask.size()[:-1], -1, self.window)

            else:

                query_layer = query_layer.reshape(-1, d)
                key_layer = key_layer.reshape(-1, d)
                value_layer = value_layer.reshape(-1, d)
                if attention_mask is not None:
                    attention_mask = attention_mask.expand(*sizes[:-2], 1, -1).reshape(1, -1)

        reduction_dim = len(query_layer.size()) - 1

        # Compute dropout here as it cannot be done on the score matrix
        query_layer = self.dropout(query_layer)
        key_layer = self.dropout(key_layer)

        context_layer = self.compute_product(
            query_layer=query_layer/math.sqrt(d), 
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask, 
            dim=reduction_dim, 
            sizes=sizes
            )

        return context_layer
    
    def compute_product(self, query_layer, key_layer, value_layer, attention_mask, dim, sizes):
        
        q = LazyTensor( query_layer.unsqueeze(-2).contiguous() )
        k = LazyTensor( key_layer.unsqueeze(-3).contiguous() )
        v = LazyTensor( value_layer.unsqueeze(-3).contiguous() )

        attention_scores = (q | k)

        if attention_mask is not None:
            mask = LazyTensor( attention_mask.unsqueeze(-1) )
            attention_scores = attention_scores + mask

        if self.local:
            self.set_ranges(sizes, query_layer.device)
            attention_scores.ranges = self.ranges

        return attention_scores.sumsoftmaxweight(v, dim=dim).reshape(*sizes)

class BlockLocalAttentionProduct(nn.Module):

    def __init__(self, config, overlap=False):
        """
        Compute block or overlapping blocks attention products
        """
        super().__init__()
 
        self.chunk_size = config.chunk_size

        assert config.sequence_len % self.chunk_size == 0

        if overlap:
            assert self.chunk_size % 2 == 0
            self.w = self.chunk_size // 2

        self.overlap = overlap

        try:
            if config.keops:
                self.attention = KeopsAttentionProduct(config)
            else:
                self.attention = BaseAttentionProduct(config)
        except:
            self.attention = BaseAttentionProduct(config)

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):

        # expect (..., t, d) shape
        context_layer = self.block_forward(
            query_layer=query_layer, 
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask
            )

        if self.overlap:
            # Split the matrix
            bot, top, context_layer = context_layer[:,:,:self.w], context_layer[:,:,-self.w:], context_layer[:,:,self.w:-self.w]
            
            # Average overlappings and compute them
            context_layer = 0.5*context_layer 
            context_layer += 0.5*self.block_forward(
                query_layer=query_layer, 
                key_layer=key_layer, 
                value_layer=value_layer, 
                attention_mask=attention_mask, 
                reduce=True
                )

            # Rebuild matrix
            context_layer = torch.cat([bot, context_layer, top], dim=-2)

        return context_layer

    def block_forward(self, query_layer, key_layer, value_layer, attention_mask, reduce=False):
        """
        Chunk the input before computing the attention
        """

        if reduce:
            query_layer = query_layer[...,self.w:-self.w,:]
            key_layer = key_layer[...,self.w:-self.w,:]
            value_layer = value_layer[...,self.w:-self.w,:]
            attention_mask = attention_mask[...,self.w:-self.w]

        initial_shape = query_layer.size()

        context_layer = self.attention(
            query_layer=self.chunk(query_layer), 
            key_layer=self.chunk(key_layer), 
            value_layer=self.chunk(value_layer), 
            attention_mask=self.chunk(attention_mask.transpose(-1, -2)).transpose(-1, -2)
            )

        context_layer = self.dechunk(context_layer, initial_shape)
        
        return context_layer

    def chunk(self, x):
        t, d = x.size()[-2:]
        return x.reshape(*x.size()[:-2], -1, self.chunk_size, d)

    def dechunk(self, x, initial_shape):
        return x.reshape(*initial_shape)