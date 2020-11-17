import torch
import torch.nn as nn
import math

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
    def __init__(self, config):

        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        # Expect (..., t, d) shape for query, key, value
        # Expect (..., t) for mask (default: (n, 1, 1, t))
        reduction_dim = len(query_layer.size()) - 1

        # Compute dropout here as it cannot be done on the score matrix
        query_layer = self.dropout(query_layer)
        key_layer = self.dropout(key_layer)

        q = LazyTensor( query_layer.unsqueeze(-2).contiguous() / math.sqrt(query_layer.shape[-1]) )
        k = LazyTensor( key_layer.unsqueeze(-3).contiguous() )
        v = LazyTensor( value_layer.unsqueeze(-3).contiguous() )

        if attention_mask is not None:
            mask = LazyTensor( attention_mask.unsqueeze(-1) )
            return ((q*k).sum(dim=-1) + mask).sumsoftmaxweight(v, dim=reduction_dim)

        return (q*k).sum(dim=-1).sumsoftmaxweight(v, dim=reduction_dim)


class BlockLocalAttentionProduct(nn.Module):

    def __init__(self, config, overlap=False):
        """
        Compute bloc or overlapping blocs attention products
        """
        super().__init__()
 
        self.chunk_size = config.chunk_size

        assert config.sequence_len % self.chunk_size == 0

        if overlap:
            assert self.chunk_size % 2 == 0
            self.w = self.chunk_size // 2

        self.overlap = overlap
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

