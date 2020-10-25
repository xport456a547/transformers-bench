import json
import os
from attention import *
from config import *
from modeling import *
import transformers.modeling_roberta

MODELS = {
    "roberta": (RobertaConfig, RobertaSelfAttention),
    "kernel": (KernelConfig, KernelSelfAttention),
    "linformer": (LinformerConfig, LinformerSelfAttention),
    "avgpooling": (AvgPoolingConfig, AvgPoolingSelfAttention),
    "maxpooling": (MaxPoolingConfig, MaxPoolingSelfAttention),
    "efficient": (EfficientConfig, EfficientSelfAttention),
    "longformer": (LongformerConfig, LongformerSelfAttention_),
    "block": (BlockConfig, BlockSelfAttention)
}


class ModelBuilder(object):

    def __init__(self, path_to_config=None, from_checkpoint=None, vocab_size=50265):

        path_to_config = None if path_to_config == "" else path_to_config
        from_checkpoint = None if from_checkpoint == "" else from_checkpoint
        assert path_to_config is not None or from_checkpoint is not None

        # Load a checkpoint folder if available
        if from_checkpoint is not None:
            assert os.path.isdir(from_checkpoint)
            self.config = json.load(open(from_checkpoint+"/config.json", "r"))
            self.from_pretrained = from_checkpoint

        # Else load model cfg
        else:
            assert os.path.isfile(path_to_config)
            self.config = json.load(open(path_to_config, "r"))
            self.config["max_position_embeddings"] += 2
            self.config["vocab_size"] = vocab_size
            self.from_pretrained = "roberta-base" if self.config["from_pretrained_roberta"] else None

        self.model_type = self.config["model"]

        assert self.model_type in MODELS

    def get_model(self):
        config = self.get_config()
        model = RobertaForMaskedLM(config)

        if self.from_pretrained is not None:
            model = model.from_pretrained(self.from_pretrained, config=config)

        return model

    def get_config(self):
        (config, attention) = MODELS[self.model_type]
        config = config(**self.config)
        transformers.modeling_roberta.RobertaSelfAttention = attention
        return config
