import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers.modeling_longformer import LongformerSelfAttention
from transformers.modeling_roberta import RobertaSelfAttention


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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if attention_mask is not None:
            attention_mask = attention_mask.transpose(-1, -2)
            query_layer = query_layer + attention_mask
            key_layer = key_layer + attention_mask

        # Apply dropout here because we dont compute a score matrix
        query_layer = self.dropout(self.act(query_layer))
        key_layer = self.dropout(self.act(key_layer))

        # Normalize
        query_layer = query_layer / (
            query_layer @ key_layer.sum(-2, keepdim=True).transpose(-1, -2) + 10e-6
        )
        context_layer = query_layer @ (key_layer.transpose(-1, -2) @ value_layer)

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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            # Custom masking for Linformer as we cannot mask properly the projected sequence
            attention_mask = attention_mask.transpose(-1, -2)
            attention_mask[attention_mask != 0] = 1
            attention_mask = 1 - attention_mask
            query_layer = query_layer * attention_mask
            key_layer = key_layer * attention_mask

        key_layer = self.E(key_layer.transpose(-1, -2))
        value_layer = self.F(value_layer.transpose(-1, -2)).transpose(-1, -2)

        attention_scores = query_layer @ key_layer
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = attention_probs @ value_layer

        context_layer = self.reshape_output(context_layer)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):

        pooled_hidden_states = self.pooling(hidden_states.transpose(-1, -2)).transpose(
            -1, -2
        )

        query_layer = self.query(hidden_states)
        key_layer = self.key(pooled_hidden_states)
        value_layer = self.value(pooled_hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            # We compute a pooled mask as the score matrix is (seq_len, reduced_seq_len)
            attention_mask = self.pooling(attention_mask.squeeze(-2)).unsqueeze(-2)
            attention_mask[attention_mask != 0] = -10000

            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = attention_probs @ value_layer

        context_layer = self.reshape_output(context_layer)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            # Mask: (n, 1, 1, t) -> (n, t, 1)

            attention_mask = attention_mask.transpose(-1, -2).squeeze(1)
            attention_mask[attention_mask != 0] = 1
            attention_mask = 1.0 - attention_mask
            key_layer = (key_layer * attention_mask).transpose(-1, -2)
            value_layer = (value_layer * attention_mask).transpose(-1, -2)

        key_layer = self.transpose_for_scores(self.pooling(key_layer).transpose(-1, -2))
        value_layer = self.transpose_for_scores(
            self.pooling(value_layer).transpose(-1, -2)
        )

        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = attention_probs @ value_layer

        context_layer = self.reshape_output(context_layer)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
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
        output_attentions=False,
    ):

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Apply dropout here because we dont compute a score matrix
        query_layer = self.dropout(query_layer)
        key_layer = self.dropout(key_layer)

        if attention_mask is not None:
            attention_mask = attention_mask.transpose(-1, -2)
            attention_mask[attention_mask != 0] = 1
            attention_mask = 1 - attention_mask

            query_layer = query_layer * attention_mask
            key_layer = key_layer * attention_mask

        # Compute Q/norm(Q) and K/norm(K)
        query_layer = query_layer / (query_layer.norm(dim=-1, keepdim=True) + 10e-6)
        key_layer = key_layer / (key_layer.norm(dim=-1, keepdim=True) + 10e-6)

        # Compute unormalized output
        context_layer = query_layer @ (
            key_layer.transpose(-1, -2) @ value_layer
        ) + value_layer.sum(dim=-2, keepdim=True)

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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Apply dropout here because we dont compute a score matrix
        query_layer = self.dropout(query_layer)
        key_layer = self.dropout(key_layer)

        if attention_mask is not None:
            attention_mask = attention_mask.transpose(-1, -2)
            query_layer = query_layer + attention_mask
            key_layer = key_layer + attention_mask

        query_layer = torch.softmax(query_layer, dim=-2)
        key_layer = torch.softmax(key_layer, dim=-2)

        # Compute normalized output
        context_layer = query_layer @ (key_layer.transpose(-1, -2) @ value_layer)

        context_layer = self.reshape_output(context_layer)
        return (context_layer,)


class LongformerSelfAttention_(LongformerSelfAttention):
    """
    Modified module to be compatible with Roberta
    We use the same attention_window for all layers
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

        # Call parent forward pass which doesnt support **kwargs
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )


class BlockSelfAttention(BaseSelfAttention):
    """
    Block attention for local attention
    Don't support head mask
    """

    def __init__(self, config):
        super().__init__()

        self.init_modules(config)
        self.chunk_size = config.chunk_size

        assert config.sequence_len % self.chunk_size == 0
        self.n_chunks = config.sequence_len // self.chunk_size

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):

        query_layer = self.transpose_for_scores(hidden_states)
        key_layer = self.transpose_for_scores(hidden_states)
        value_layer = self.transpose_for_scores(hidden_states)

        query_layer = self.reshape_chunks(query_layer, dim=-2)
        key_layer = self.reshape_chunks(key_layer, dim=-2)
        value_layer = self.reshape_chunks(value_layer, dim=-2)
        attention_mask = self.reshape_chunks(attention_mask, dim=-1)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = attention_probs @ value_layer

        context_layer = self.reverse_chunks(context_layer, dim=-2)
        context_layer = self.reshape_output(context_layer)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs

    def reshape_chunks(self, x, dim):
        size = list(x.size())
        size[dim] = size[dim] // self.n_chunks
        size[0] *= self.n_chunks
        return x.reshape(*size)

    def reverse_chunks(self, x, dim):
        size = list(x.size())
        size[dim] *= self.n_chunks
        size[0] = size[0] // self.n_chunks
        return x.reshape(*size)