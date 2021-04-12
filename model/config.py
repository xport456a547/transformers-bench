from transformers.models.roberta.modeling_roberta import RobertaConfig
from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.big_bird.modeling_big_bird import BigBirdConfig


class RobertaConfig(BertConfig):

    model = "roberta"
    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        sequence_len=512,
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

        self.sequence_len = sequence_len

class KernelConfig(RobertaConfig):

    model = "kernel"
    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs KernelConfig."""
        super().__init__(**kwargs)


class LinformerConfig(RobertaConfig):

    model = "linformer"
    model_type = "roberta"

    def __init__(
        self,
        sequence_len=512,
        projection_length=128,
        projection_bias=False,
        **kwargs
        ):
        """Constructs LinformerConfig."""
        super().__init__(**kwargs)

        self.sequence_len = sequence_len
        self.projection_length = projection_length
        self.projection_bias = projection_bias


class AvgPoolingConfig(RobertaConfig):

    model = "avgpooling"
    model_type = "roberta"

    def __init__(
        self,
        kernel_size=8,
        stride=4,
        **kwargs
        ):
        """Constructs AvgPoolingConfig."""
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.stride = stride


class MaxPoolingConfig(AvgPoolingConfig):

    model = "maxpooling"
    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs MaxPoolingConfig."""
        super().__init__(**kwargs)


class CosineConfig(RobertaConfig):

    model = "cosine"
    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs CosineConfig."""
        super().__init__(**kwargs)


class EfficientConfig(RobertaConfig):

    model = "efficient"
    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs EfficientConfig."""
        super().__init__(**kwargs)


class LongformerConfig(RobertaConfig):

    model = "longformer"
    model_type = "roberta"

    def __init__(
        self,
        attention_window=128,
        **kwargs
        ):
        """Constructs LongformerConfig."""
        super().__init__(**kwargs)

        # We keep the same window for all layers
        self.attention_window = [attention_window]


class LocalConfig(RobertaConfig):

    model = "local"
    model_type = "roberta"

    def __init__(
        self,
        local_attn_chunk_length=128,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        is_decoder=False,
        **kwargs
        ):
        """Constructs LocalConfig."""
        super().__init__(**kwargs)

        self.local_attn_chunk_length = local_attn_chunk_length
        self.local_num_chunks_before = local_num_chunks_before
        self.local_num_chunks_after = local_num_chunks_after
        self.is_decoder = is_decoder
        self.local_attention_probs_dropout_prob = self.attention_probs_dropout_prob
        self.attention_head_size = self.hidden_size // self.num_attention_heads
  

class BlockConfig(RobertaConfig):

    model = "block"
    model_type = "roberta"

    def __init__(
        self,
        chunk_size=128,
        use_global=True,
        **kwargs
        ):
        """Constructs BlockConfig."""
        super().__init__(**kwargs)

        self.chunk_size = chunk_size
        self.use_global = use_global


class BlockLocalConfig(RobertaConfig):

    model = "block-local"
    model_type = "roberta"

    def __init__(
        self,
        chunk_size=128,
        use_global=True,
        **kwargs
        ):
        """Constructs BlockLocalConfig."""
        super().__init__(**kwargs)

        self.chunk_size = chunk_size
        self.use_global = use_global

class BlockGlobalConfig(RobertaConfig):

    model = "block-global"
    model_type = "roberta"

    def __init__(
        self,
        chunk_size=128,
        topk=128,
        keops=False,
        **kwargs
        ):
        """Constructs BlockGlobalConfig."""
        super().__init__(**kwargs)

        # We keep the same window for all layers
        self.chunk_size = chunk_size
        self.topk = topk
        self.keops = keops

class BigBirdConfig(BigBirdConfig):

    model = "bigbird"
    model_type = "roberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
        block_size=64,
        num_random_blocks=2,
        seed=0, 
        attention_type="block_sparse",
        **kwargs
        ):
        """Constructs BigBirdConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            block_size=block_size,
            num_random_blocks=num_random_blocks,
            seed=seed, 
            attention_type=attention_type,
            **kwargs
            )

        # We keep the same window for all layers
        self.block_size = block_size
        self.attention_type = attention_type
        self.num_random_blocks = num_random_blocks
        self.seed = seed
        

class LSHConfig(RobertaConfig):

    model = "lsh"
    model_type = "roberta"

    def __init__(
        self,
        lsh_attn_chunk_length=128,
        num_hashes=4,
        num_buckets=128,
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        hash_seed=None,
        is_decoder=False,
        **kwargs
        ):
        """Constructs LSHConfig."""
        super().__init__(**kwargs)

        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.num_hashes = num_hashes
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.hash_seed = hash_seed
        self.is_decoder = is_decoder
        self.lsh_attention_probs_dropout_prob = self.attention_probs_dropout_prob
        self.attention_head_size = self.hidden_size // self.num_attention_heads

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


class LSHFTConfig(RobertaConfig):

    model = "lsh-ft"
    model_type = "roberta"

    def __init__(
        self,
        sequence_len=512,
        chunk_size=16,
        bits=8,
        rounds=4,
        **kwargs
            ):
        """Constructs LSHFTConfig."""
        super().__init__(**kwargs)

        self.sequence_len = sequence_len
        self.chunk_size = chunk_size
        self.bits = bits
        self.rounds = rounds


class KeopsConfig(RobertaConfig):

    model = "keops"
    model_type = "roberta"

    def __init__(self, **kwargs):
        """Constructs KeopsConfig."""
        super().__init__(**kwargs)


      