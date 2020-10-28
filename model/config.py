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


class ReformerConfig(BertConfig):

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
        self.bits = bits
        self.rounds = rounds