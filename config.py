import json

from transformers.modeling_roberta import RobertaConfig
from transformers.modeling_bert import BertConfig
from modeling import *
from attention import *
import transformers.modeling_roberta

class RobertaConfig(BertConfig):

    model_type = "roberta"

    def __init__(
        self, 
        pad_token_id=1, 
        bos_token_id=0, 
        eos_token_id=2, 
        type_vocab_size=1,
        **kwargs):
        """Constructs LinformerConfig."""
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
        **kwargs):
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
        **kwargs):
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
        attention_window_size=512, 
        **kwargs):
        """Constructs LongformerConfig."""
        super().__init__(
            pad_token_id=pad_token_id, 
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id, 
            type_vocab_size=type_vocab_size,
            **kwargs
            )

        self.attention_window_size = attention_window_size

class ModelBuilder(object):

    def __init__(self, path_to_config, tokenizer):
        self.config = json.load(open(path_to_config, "r"))
        self.config["max_position_embeddings"] += 2
        self.config["vocab_size"] = len(tokenizer)

        self.model_type = self.config["model"]
        self.from_pretrained = self.config["from_pretrained"]

        assert self.model_type in ["roberta", "kernel", "linformer", "avgpooling", "maxpooling", "efficient", "longformer"]
        assert self.model_type != "longformer", "Not implemented yet"

    def get_model(self):
        config = self.get_config()
        model = RobertaForMaskedLM(config)

        if self.config["from_pretrained"]:
            model = model.from_pretrained("roberta-base", config=config)

        return model

    def get_config(self):
        if self.model_type == "roberta":
            config = RobertaConfig(**self.config)
             
        elif self.model_type == "kernel":
            config = KernelConfig(**self.config)
            transformers.modeling_roberta.RobertaSelfAttention = KernelSelfAttention

        elif self.model_type == "linformer":
            config = LinformerConfig(**self.config)
            transformers.modeling_roberta.RobertaSelfAttention = LinformerSelfAttention

        elif self.model_type == "avgpooling":
            config = AvgPoolingConfig(**self.config)
            transformers.modeling_roberta.RobertaSelfAttention = AvgPoolingSelfAttention

        elif self.model_type == "maxpooling":
            config = MaxPoolingConfig(**self.config)
            transformers.modeling_roberta.RobertaSelfAttention = MaxPoolingSelfAttention

        elif self.model_type == "efficient":
            config = EfficientConfig(**self.config)
            transformers.modeling_roberta.RobertaSelfAttention = EfficientSelfAttention

        return config