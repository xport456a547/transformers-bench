import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers.models.big_bird.modeling_big_bird import BigBirdSelfAttention, BigBirdBlockSparseAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from transformers.models.reformer.modeling_reformer import LSHSelfAttention as ReformerLSHSelfAttention
from transformers.models.reformer.modeling_reformer import LocalSelfAttention as ReformerLocalSelfAttention

try:
    from fast_transformers.attention.reformer_attention import ReformerAttention
    from fast_transformers.masking import FullMask, LengthMask
except:
    import logging
    logging.info("pytorch-fast-transformers is not installed")

from model.attention_product import *


class BaseSelfAttention(nn.Module):
    
    def init_modules(self, config):
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def reshape_output(self, context_layer):
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_context_layer_shape)

    def project_QKV(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        return query_layer, key_layer, value_layer


class KernelSelfAttention(BaseSelfAttention):
    """
    Adapted from "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    https://arxiv.org/abs/2006.16236
    Can't return attention_probs
    Doesn't support headmask
    """

    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.act = lambda x: F.elu(x) + 1
        self.attention = KernelAttentionProduct(config, normalize=True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.transpose(-1, -2)
            query_layer = query_layer + attention_mask
            key_layer = key_layer + attention_mask
    
        context_layer = self.attention(
            query_layer=self.act(query_layer), 
            key_layer=self.act(key_layer), 
            value_layer=value_layer
            )

        context_layer = self.reshape_output(context_layer)

        return (context_layer,)


class LinformerSelfAttention(BaseSelfAttention):
    """
    Adapted from "Linformer: Self-Attention with Linear Complexity"
    https://arxiv.org/abs/2006.04768
    """

    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.E = nn.Linear(
            config.sequence_len, config.projection_length, bias=config.projection_bias
            )
        self.F = nn.Linear(
            config.sequence_len, config.projection_length, bias=config.projection_bias
            )

        self.attention = BaseAttentionProduct(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        if attention_mask is not None:
            # Custom masking for Linformer as we cannot mask properly the projected sequence
            attention_mask = attention_mask.transpose(-1, -2)
            attention_mask = (attention_mask / 10000) + 1
            
            query_layer = query_layer * attention_mask
            key_layer = key_layer * attention_mask

        context_layer = self.attention(
            query_layer=query_layer, 
            key_layer=self.E(key_layer.transpose(-1, -2)).transpose(-1, -2), 
            value_layer=self.F(value_layer.transpose(-1, -2)).transpose(-1, -2), 
            attention_mask=None
            )

        context_layer = self.reshape_output(context_layer)
        outputs = (context_layer,)
        return outputs


class AvgPoolingSelfAttention(BaseSelfAttention):

    def __init__(self, config):
        """
        Attention using average pooling
        Replace the product (n, T, d).(n, d, T).(n, T, d) by (n, T, d) (n, d, t) (n, t, d) with T > t
        """
        super().__init__()

        self.init_modules(config)
        self.pooling = nn.AvgPool1d(config.kernel_size, config.stride)
        self.attention = BaseAttentionProduct(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        pooled_hidden_states = self.pooling(
            hidden_states.transpose(-1, -2)).transpose(-1, -2)

        query_layer = self.query(hidden_states)
        key_layer = self.key(pooled_hidden_states)
        value_layer = self.value(pooled_hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            # We compute a pooled mask as the score matrix is (seq_len, reduced_seq_len)
            attention_mask = self.pooling(attention_mask.squeeze(-2)).unsqueeze(-2)
            attention_mask[attention_mask != 0] = -10000

        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)

        context_layer = self.reshape_output(context_layer)

        outputs = (context_layer, )
        return outputs


class MaxPoolingSelfAttention(BaseSelfAttention):
    """
    Attention using max pooling
    Replace the product (n, T, d).(n, d, T).(n, T, d) by (n, T, d) (n, d, t) (n, t, d) with T > t
    """

    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.pooling = nn.MaxPool1d(config.kernel_size, config.stride)
        self.attention = BaseAttentionProduct(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        n, h, t, d = query_layer.size()

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            key_layer = key_layer + attention_mask.transpose(-1, -2)
            value_layer = value_layer + attention_mask.transpose(-1, -2)

        key_layer = self.pooling(key_layer.transpose(-1, -2).reshape(-1, d, t)).reshape(n, h, d, -1).transpose(-1, -2)
        value_layer = self.pooling(value_layer.transpose(-1, -2).reshape(-1, d, t)).reshape(n, h, d, -1).transpose(-1, -2)

        context_layer = self.attention(
            query_layer=query_layer, 
            key_layer=key_layer, 
            value_layer=value_layer, 
            )

        context_layer = self.reshape_output(context_layer)
        outputs = (context_layer, )
        return outputs


class CosineSelfAttention(BaseSelfAttention):
    """
    Attention using (cosine + 1) as score
    Replace (Q @ K) @ V by Q @ (K @ V) to reduce complexity
    """

    def __init__(self, config):
        super().__init__()
        self.init_modules(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.transpose(-1, -2)
            attention_mask = (attention_mask / 10000) + 1

            query_layer = query_layer * attention_mask
            key_layer = key_layer * attention_mask

        # Compute Q/norm(Q) and K/norm(K)
        query_layer = query_layer / (query_layer.norm(dim=-1, keepdim=True) + 10e-6)
        key_layer = key_layer / (key_layer.norm(dim=-1, keepdim=True) + 10e-6)

        # Compute unormalized output with dropout
        context_layer = self.dropout(key_layer.transpose(-1, -2) @ value_layer)
        context_layer = query_layer @ context_layer + value_layer.sum(dim=-2, keepdim=True)

        # Normalisation
        normalization = (
            query_layer @ key_layer.sum(-2, keepdim=True).transpose(-1, -2) + 10e-6
        )

        if attention_mask is not None:
            normalization = normalization + attention_mask.sum(dim=-2, keepdim=True)
        else:
            normalization = normalization + context_layer.size(-2)
            
        context_layer = context_layer / normalization
        context_layer = self.reshape_output(context_layer)
        return (context_layer,)


class EfficientSelfAttention(BaseSelfAttention):
    """
    Adapted from "Efficient Attention: Attention with Linear Complexities"
    https://arxiv.org/abs/1812.01243
    Can't return attention_probs
    Doesn't support headmask
    """

    def __init__(self, config):
        super().__init__()
        self.init_modules(config)
        self.attention = KernelAttentionProduct(config, normalize=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.transpose(-1, -2)
            query_layer = query_layer + attention_mask
            key_layer = key_layer + attention_mask

        context_layer = self.attention(
            query_layer=torch.softmax(query_layer, dim=-2), 
            key_layer=torch.softmax(key_layer, dim=-2), 
            value_layer=value_layer
            )

        context_layer = self.reshape_output(context_layer)
        return (context_layer,)


class LongformerSelfAttention(LongformerSelfAttention):
    """
    Modified module to be compatible with Roberta
    We use the same attention_window for all layers
    See: "Longformer: The Long-Document Transformer"
    https://arxiv.org/abs/2004.05150
    """

    def __init__(self, config):
        super().__init__(config=config, layer_id=0)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        ):

        """
        # Fix HuggingFace dogshit arguments that change every release
        # Call parent forward pass to handle longformer
        attention_mask = attention_mask.squeeze(1).squeeze(1)
        is_index_masked = attention_mask.bool()

        # Manually set <s> as global
        attention_mask[:, 0] = 10000

        is_index_global_attn = torch.zeros_like(attention_mask)
        is_index_global_attn[:, 0] = 1
        is_index_global_attn = is_index_global_attn.bool()
        """

        attention_mask = attention_mask, is_index_masked, is_index_global_attn
        output = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=True
        )
        
        return (output[0], )


class LocalSelfAttention(ReformerLocalSelfAttention):
    """
    Compute local attention with a given window size
    Based on HuggingFace Reformer code
    """
    def __init__(self, config):
        super().__init__(config=config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False
        ):

        attention_mask = attention_mask.squeeze(1).squeeze(1)
        attention_mask = (attention_mask / 10000) + 1

        context_layer = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_buckets_states=None,
            use_cache=False,
            output_attentions=False,
            )

        return (context_layer[0], )


class BlockSelfAttention(BaseSelfAttention):
    """
    Block attention for local attention
    Don't support head mask
    """

    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.attention = BlockAttentionProduct(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        context_layer = self.attention(
            query_layer=query_layer, 
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask
            )

        context_layer = self.reshape_output(context_layer)
        return (context_layer, )


class BlockLocalSelfAttention(BaseSelfAttention):
    '''
    Compute local attention with overlapping blocks
    '''

    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.local_attention = BlockLocalAttentionProduct(config, chunk_size=config.chunk_size)
        self.use_global = config.use_global
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        n, h, t, d = query_layer.size()
        attention_mask = attention_mask.expand(n, h, 1, t)

        if self.use_global:
            context_layer = self.local_attention(
                query_layer=query_layer, 
                key_layer=key_layer, 
                value_layer=value_layer, 
                attention_mask=attention_mask,
                global_key=key_layer[:,:,0].unsqueeze(-2),
                global_value=value_layer[:,:,0].unsqueeze(-2),
                global_mask=torch.zeros(n, h, 1, 1, device=attention_mask.device)
                )

            bos = torch.softmax(query_layer[...,0,:].unsqueeze(-2) @ key_layer.transpose(-1, -2), dim=-1) @ value_layer
            context_layer[...,0,:] = bos.squeeze(-2)

        else:
            context_layer = self.local_attention(
                query_layer=query_layer, 
                key_layer=key_layer, 
                value_layer=value_layer, 
                attention_mask=attention_mask,
                )

        context_layer = self.reshape_output(context_layer)

        return (context_layer,)


class BlockGlobalSelfAttention(BaseSelfAttention):
    '''
    Compute local attention with overlapping blocs
    Use global attention for tokens with highest norm
    '''
    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.topk = config.topk
        self.chunk_size = self.topk*config.factor

        self.local_attention = BlockLocalAttentionProduct(config, return_logsumexp=True)
        self.global_attention = BlockLocalAttentionProduct(config, chunk_size=self.chunk_size, return_logsumexp=True)
    
    def get_indexes(self, x, mask):
        
        x = self.chunk(x, self.chunk_size)

        n, h, b, t, d = x.size()
        mask = mask.reshape(n, 1, b, 1, t)

        if mask is not None:
            mask = ~mask.transpose(-1, -2).bool()

        x = x.pow(2).sum(dim=-1, keepdim=True)*mask
        #x[...,0,:] += 10000
        idx = x.argsort(dim=-2)

        global_idx = idx[...,-self.topk:,:]
        local_idx = idx[...,:-self.topk,:]

        pad = (torch.arange(b, device=idx.device)*t).reshape(1, 1, b, 1, 1)
        global_idx = global_idx.gather(dim=-2, index=global_idx.argsort(dim=-2)) + pad
        local_idx = local_idx.gather(dim=-2, index=local_idx.argsort(dim=-2)) + pad

        return global_idx.reshape(n, h, -1, 1), local_idx.reshape(n, h, -1, 1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        n, h, t, d = key_layer.size()

        # <s> token
        bos = query_layer[:,:,0].unsqueeze(-2) @ key_layer.transpose(-1, -2) + attention_mask
        bos = torch.softmax(bos, dim=-1) @ value_layer

        # Get indexes
        global_idx, local_idx = self.get_indexes(key_layer + self.query.bias.reshape(1, h, 1, d), attention_mask)

        # Get masks
        attention_mask = attention_mask.expand(n, h, 1, t)
        attention_mask_global = attention_mask.gather(dim=-1, index=global_idx.transpose(-1, -2))
        attention_mask = attention_mask.gather(dim=-1, index=local_idx.transpose(-1, -2))

        # Compute local attention
        local_idx = local_idx.expand(n, h, -1, d)
        (context_layer, sumexp) = self.local_attention(
            query_layer, 
            key_layer.gather(dim=-2, index=local_idx), 
            value_layer.gather(dim=-2, index=local_idx), 
            attention_mask,
            global_key=key_layer[:,:,0].unsqueeze(-2),
            global_value=value_layer[:,:,0].unsqueeze(-2),
            global_mask=torch.zeros(n, h, 1, 1, device=local_idx.device)
            )
        del local_idx

        # Compute attention with global tokens
        global_idx = global_idx.expand(n, h, -1, d)
        (context_layer_global, sumexp_global) = self.global_attention(
            query_layer, 
            key_layer.gather(dim=-2, index=global_idx), 
            value_layer.gather(dim=-2, index=global_idx), 
            attention_mask_global)

        del global_idx

        # Merge branches
        probs = 1/(1 + (sumexp_global - sumexp).exp()).detach()
        context_layer = context_layer_global + probs*(context_layer - context_layer_global)

        context_layer[...,0,:] = bos.squeeze(-2)
        context_layer = self.reshape_output(context_layer)

        return (context_layer,)

    def chunk(self, x, chunk_size):
        n, h, t, d = x.size()
        return x.reshape(n, h, -1, chunk_size, d)


class BlockGlobalSelfAttentionMerged(BaseSelfAttention):
    '''
    Compute local attention with overlapping blocs
    Use global attention for tokens with highest norm
    '''
    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.topk = config.topk
        self.chunk_size = self.topk*config.factor

        self.attention = BlockGlobalAttentionProduct(config, chunk_size=config.chunk_size, topk_chunk_size=self.chunk_size)
    
    def get_global_index(self, x, mask):
        
        x = self.chunk(x, self.chunk_size)

        n, h, b, t, d = x.size()
        mask = mask.reshape(n, 1, b, 1, t)

        if mask is not None:
            mask = ~mask.transpose(-1, -2).bool()

        x = x.norm(dim=-1, keepdim=True)*mask
        #x[...,0,:] += 10000
        idx = x.argsort(dim=-2)
        
        global_idx = idx[...,-self.topk:,:]
        local_idx = idx[...,:-self.topk,:]

        pad = (torch.arange(b, device=idx.device)*t).reshape(1, 1, b, 1, 1)
        global_idx = global_idx.gather(dim=-2, index=global_idx.argsort(dim=-2)) + pad
        local_idx = local_idx.gather(dim=-2, index=local_idx.argsort(dim=-2)) + pad

        return global_idx.reshape(n, h, -1, 1), local_idx.reshape(n, h, -1, 1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        n, h, t, d = key_layer.size()

        global_idx, local_idx = self.get_global_index(key_layer + self.query.bias.reshape(1, h, 1, d), attention_mask)

        context_layer = self.attention(
            query_layer, 
            key_layer, 
            value_layer, 
            attention_mask,
            local_idx,
            global_idx,
            global_key=key_layer[:,:,0].unsqueeze(-2),
            global_value=value_layer[:,:,0].unsqueeze(-2),
            global_mask=torch.zeros(n, h, 1, 1, device=local_idx.device)
            )

        # <s> token
        bos = query_layer[:,:,0].unsqueeze(-2) @ key_layer.transpose(-1, -2) + attention_mask
        bos = torch.softmax(bos, dim=-1) @ value_layer
        context_layer[...,0,:] = bos.squeeze(-2)
        
        context_layer = self.reshape_output(context_layer)

        return (context_layer,)

    def chunk(self, x, chunk_size):
        n, h, t, d = x.size()
        return x.reshape(n, h, -1, chunk_size, d)


class BigBirdBlockSparseAttention(BigBirdBlockSparseAttention):
    """
    Modified module to be compatible with Roberta
    We use the same attention_window for all layers
    See: "Longformer: The Long-Document Transformer"
    https://arxiv.org/abs/2004.05150
    """

    def __init__(self, config):
        super().__init__(config=config, seed=config.seed)
        self.block_size = config.block_size
        self.max_seqlen = config.sequence_len

    def create_masks_for_block_sparse_attn(self, attention_mask, block_size):
        """
        Computes mask
        See: https://github.com/huggingface/transformers/blob/master/src/transformers/models/big_bird/modeling_big_bird.py
        In the original implementation, the mask is computed once for all layers. 
        Here we compute a different random mask each layer since it is a lot easier to implement from HF library
        """
        batch_size, seq_length = attention_mask.size()

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            
            exp_blocked_to_pad = torch.cat(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2
            )
            band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask

        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):
        
        """
        # Change {-10000, 0} mask to {0, 1}
        attention_mask = (1 - attention_mask.bool().float()).squeeze(1).squeeze(1)

        # Compute BigBird mask
        blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(attention_mask, self.block_size)
        """
        blocked_encoder_mask, band_mask, from_mask, to_mask = attention_mask
        return super().forward(
            hidden_states,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=blocked_encoder_mask,
            to_blocked_mask=blocked_encoder_mask,
            output_attentions=output_attentions,
        )


class LSHSelfAttention(ReformerLSHSelfAttention):

    def __init__(self, config):
        super().__init__(config=config)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.query_key = self.query

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False
        ):

        attention_mask = attention_mask.squeeze(1).squeeze(1)
        attention_mask = (attention_mask / 10000) + 1

        context_layer = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            num_hashes=self.num_hashes,
            buckets=None,
            past_buckets_states=None,
            use_cache=False,
            output_attentions=False,
            )

        return (context_layer[0], )


class LSHFTSelfAttention(BaseSelfAttention):
    """
    Adapted from "Reformer: The Efficient Transformer"
    https://arxiv.org/abs/2001.04451

    Use https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/reformer_attention.py
    Can't return attention_probs
    Doesn't support headmask
    """

    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.reformer = ReformerAttention(
            chunk_size=config.chunk_size, 
            bits=config.bits, 
            rounds=config.rounds, 
            attention_dropout=config.hidden_dropout_prob
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

        # Change mask behavior to be compatible with https://github.com/idiap/fast-transformers
        attention_mask = (attention_mask.squeeze(1).squeeze(1) + 10000) / 10000
        attention_mask = FullMask(mask=attention_mask.bool())
        attention_length = LengthMask(attention_mask.lengths, max_len=t)

        context_layer = self.reformer(
            queries=query_layer, 
            keys=key_layer, 
            values=value_layer, 
            attn_mask=attention_mask, 
            query_lengths=attention_length, 
            key_lengths=attention_length
            )

        context_layer = self.reshape_output(context_layer.transpose(1, 2))

        return (context_layer,)
    

class KeopsSelfAttention(BaseSelfAttention):
    """
    Keops attention for full attention
    Don't support head mask
    """

    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.attention = KeopsAttentionProduct(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)

        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)
        context_layer = self.reshape_output(context_layer)

        return (context_layer,)
