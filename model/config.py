from transformers.modeling_roberta import RobertaConfig
from transformers.modeling_bert import BertConfig


class RobertaConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        **kwargs
        ):
        """Constructs RobertaConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
            )


class KernelConfig(RobertaConfig):

    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs KernelConfig."""
        super().__init__(**kwargs)


class LinformerConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        sequence_len=512,
        projection_length=128,
        projection_bias=False,
        **kwargs
        ):
        """Constructs LinformerConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
            )

        self.sequence_len = sequence_len
        self.projection_length = projection_length
        self.projection_bias = projection_bias


class AvgPoolingConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        kernel_size=8,
        stride=4,
        **kwargs
        ):
        """Constructs AvgPoolingConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
            )

        self.kernel_size = kernel_size
        self.stride = stride


class MaxPoolingConfig(AvgPoolingConfig):

    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs MaxPoolingConfig."""
        super().__init__(**kwargs)


class CosineConfig(RobertaConfig):

    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs CosineConfig."""
        super().__init__(**kwargs)


class EfficientConfig(RobertaConfig):

    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs EfficientConfig."""
        super().__init__(**kwargs)


class LongformerConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        attention_window=128,
        **kwargs
        ):
        """Constructs LongformerConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
            )

        # We keep the same window for all layers
        self.attention_window = [attention_window]


class LocalConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        local_attn_chunk_length=128,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        is_decoder=False,
        **kwargs
        ):
        """Constructs LocalConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
        )

        self.local_attn_chunk_length = local_attn_chunk_length
        self.local_num_chunks_before = local_num_chunks_before
        self.local_num_chunks_after = local_num_chunks_after
        self.is_decoder = is_decoder
        self.local_attention_probs_dropout_prob = self.attention_probs_dropout_prob
        self.attention_head_size = self.hidden_size // self.num_attention_heads
  

class BlockConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        sequence_len=512,
        chunk_size=16,
        **kwargs
        ):
        """Constructs BlockConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
            )

        self.sequence_len = sequence_len
        self.chunk_size = chunk_size


class BlockLocalConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        attention_window=128,
        **kwargs
        ):
        """Constructs BlockLocalConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
            )

        # We keep the same window for all layers
        self.attention_window = [attention_window]


class BlockGlobalConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        attention_window=128,
        topk=128,
        **kwargs
        ):
        """Constructs BlockGlobalConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
            )

        # We keep the same window for all layers
        self.attention_window = [attention_window]
        self.topk = topk


class LSHConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        lsh_attn_chunk_length=128,
        num_hashes=4,
        num_buckets=128,
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        hash_seed=None,
        is_decoder=False,
        sequence_len=512,
        **kwargs
        ):
        """Constructs LSHConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
        )

        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.num_hashes = num_hashes
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.hash_seed = hash_seed
        self.is_decoder = is_decoder
        self.lsh_attention_probs_dropout_prob = self.attention_probs_dropout_prob
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.sequence_len = sequence_len

        self.num_buckets = num_buckets
        if self.num_buckets <= 0:
            self.num_buckets = self.get_num_buckets()

    def get_num_buckets(self):
        # `num_buckets` should be set to 2 * sequence_length // chunk_length as recommended in paper
        num_buckets_pow_2 = (2 * (self.sequence_len // self.lsh_attn_chunk_length)).bit_length() - 1
        # make sure buckets are power of 2
        num_buckets = 2 ** num_buckets_pow_2

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = 2 * max(
            int((self.max_position_embeddings // self.lsh_attn_chunk_length) ** (0.5)),
            self.lsh_attn_chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]

        return num_buckets


class LSHFTConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        sequence_len=512,
        chunk_size=16,
        bits=8,
        rounds=4,
        **kwargs
            ):
        """Constructs LSHFTConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
            )

        self.sequence_len = sequence_len
        self.chunk_size = chunk_size
        self.bits = bits
        self.rounds = rounds


class KeopsConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        **kwargs
        ):
        """Constructs KeopsConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
        )


      