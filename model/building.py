import json
import os
from model.attention import *
from model.modeling import *
import transformers.models.roberta.modeling_roberta
from model.config import *
import logging

MODELS = {
    "roberta": (RobertaConfig, RobertaSelfAttention),           
    "kernel": (KernelConfig, KernelSelfAttention),
    "linformer": (LinformerConfig, LinformerSelfAttention),
    "avgpooling": (AvgPoolingConfig, AvgPoolingSelfAttention),
    "maxpooling": (MaxPoolingConfig, MaxPoolingSelfAttention),
    "cosine": (CosineConfig, CosineSelfAttention),
    "efficient": (EfficientConfig, EfficientSelfAttention),
    "longformer": (LongformerConfig, LongformerSelfAttention_),
    "local": (LocalConfig, LocalSelfAttention), #pytorch-fast-transformer local attention implementation
    "block": (BlockConfig, BlockSelfAttention),
    "block-local": (BlockLocalConfig, BlockLocalSelfAttention),
    "block-global": (BlockGlobalConfig, BlockGlobalSelfAttention),
    "block-global-merged": (BlockGlobalConfig, BlockGlobalSelfAttentionMerged),
    "lsh": (LSHConfig, LSHSelfAttention),        #huggingface  LSH
    "lsh-ft": (LSHFTConfig, LSHFTSelfAttention), #pytorch-fast-transformer LSH
    "keops": (KeopsConfig, KeopsSelfAttention),
}


class ModelBuilder(object):

    def __init__(self, path_to_config=None, from_checkpoint=None, vocab_size=50265):

        path_to_config = None if path_to_config == "" else path_to_config
        from_checkpoint = None if from_checkpoint == "" else from_checkpoint
        assert path_to_config is not None or from_checkpoint is not None

        self.from_checkpoint = from_checkpoint

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

        if config.sequence_len > config.max_position_embeddings - 2:
            model, config = self.set_positional_embeddings(model, config)

        if self.from_checkpoint is None:
            self.init_global_params(model)

        return model, config

    def get_config(self):
        """
        Get config and its associated model
        """
        (config, attention) = MODELS[self.model_type]
        config = config(**self.config)
        transformers.models.roberta.modeling_roberta.RobertaSelfAttention = attention
        return config

    def set_positional_embeddings(self, model, config):
        """
        Handle longer sequences and duplicate positional embedding
        """
        logging.info("Duplicating positional embedding")

        current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
        max_pos = config.sequence_len + 2

        assert max_pos > current_max_pos
        config.max_position_embeddings = max_pos
        current_max_pos -= 2

        chunks = config.sequence_len // current_max_pos
        last_chunk = config.sequence_len % current_max_pos

        current_position_embeddings_weight = model.roberta.embeddings.position_embeddings.weight.clone()

        new_position_embedding_weight = torch.cat(
            [current_position_embeddings_weight[:2]]
            + [current_position_embeddings_weight[2:] for _ in range(chunks)] 
            + [current_position_embeddings_weight[2:last_chunk+2]], dim=0)

        model.roberta.embeddings.position_embeddings.weight.data = new_position_embedding_weight
        model.roberta.embeddings.position_ids = torch.arange(max_pos, device=model.roberta.embeddings.position_ids.device).unsqueeze(0)
        return model, config

    def init_global_params(self, model):
        """
        Init global projections with local projections if it exists
        """
        for i, layer in enumerate(model.roberta.encoder.layer):

            if hasattr(layer.attention.self, 'query_global') and hasattr(layer.attention.self, 'query'):
                layer.attention.self.query_global.weight.data = layer.attention.self.query.weight.data.clone()
                layer.attention.self.query_global.bias.data = layer.attention.self.query.bias.data.clone()

            if hasattr(layer.attention.self, 'key_global') and hasattr(layer.attention.self, 'key'):
                layer.attention.self.key_global.weight.data = layer.attention.self.key.weight.data.clone()
                layer.attention.self.key_global.bias.data = layer.attention.self.key.bias.data.clone()

            if hasattr(layer.attention.self, 'value_global') and hasattr(layer.attention.self, 'value'):
                layer.attention.self.value_global.weight.data = layer.attention.self.value.weight.data.clone()
                layer.attention.self.value_global.bias.data = layer.attention.self.value.bias.data.clone()


        