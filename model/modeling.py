from transformers.models.roberta.modeling_roberta import *
from transformers.activations import ACT2FN
 
class RobertaEncoder_(RobertaEncoder):

    def __init__(self, config):
        super().__init__(config)
        
        self.model_name = config.model
        if self.model_name == "bigbird":
            self.block_size = config.block_size

    def create_masks_for_block_sparse_attn(self, attention_mask, block_size):
        """
        Computes mask for BigBird
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

        return (blocked_encoder_mask, band_mask, from_mask, to_mask)

    def create_mask_for_longformer(self, attention_mask):
        # Fix HuggingFace dogshit mask
        is_index_masked = attention_mask.bool()

        # Manually set <s> as global
        attention_mask[:, 0] = 10000
        is_index_global_attn = torch.zeros_like(attention_mask)
        is_index_global_attn[:, 0] = 1

        return (attention_mask, is_index_masked, is_index_global_attn.bool())

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
        ):

        # Override mask behavior
        if self.model_name == "bigbird":
            attention_mask = (1 - attention_mask.bool().float()).squeeze(1).squeeze(1)
            attention_mask = self.create_masks_for_block_sparse_attn(attention_mask, self.block_size)

        elif self.model_name == "longformer":
            attention_mask = attention_mask.squeeze(1).squeeze(1)
            attention_mask = self.create_mask_for_longformer(attention_mask)

        return super().forward(
            hidden_states, 
            attention_mask=attention_mask, 
            head_mask=head_mask, 
            encoder_hidden_states=encoder_attention_mask, 
            encoder_attention_mask=encoder_attention_mask, 
            past_key_values=past_key_values, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict
            )


class RobertaForMaskedLM(RobertaPreTrainedModel):

    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.roberta.encoder = RobertaEncoder_(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
        ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        prediction_scores = torch.argmax(prediction_scores, dim=-1, keepdim=True)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )