from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    MistralConfig,
    MistralModel,
    MistralForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    
    """
    LlavaMetaModel 
        is a mixin that adds the vision side of LLaVA:
        it builds the vision tower (image encoder) and the projector that maps
        image patch embeddings into the LLM hidden size so they can be inserted
        into the text token stream.
    MistralModel
        provides the core language model (token embeddings + decoder stack).
        LlavaMetaModel only adds vision modules; it does not implement the LM itself.
    """
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config) # == super().__init__(config)


# LLaVA architecture using Mistral as the text backbone.
# Image
#  ↓
# Vision encoder (CLIP)
#  ↓
# Image feature vectors
#  ↓
# Projection layer (mm_projector)
#  ↓
# Mapped into Mistral embedding space
#  ↓
# Mistral processes everything together
#  ↓
# Text answer

# Mistral is the language brain.
# LLaVA is the multimodal wrapper that lets images talk to that brain.

class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config):
        # "skip MistralForCausalLM.__init__" and call the next __init__ in the MRO:
        # however we still have all methods of ALL the parent classes
        # So the intent is: keep Mistral behavior, replace Mistral’s internal model construction.
        super(MistralForCausalLM, self).__init__(config)   
        # LlavaMistralForCausalLM -> MistralForCausalLM -> MistralPreTrainedModel -> PreTrainedModel.__init__ (-> nn.Module.__init__)                                   
       
        self.model = LlavaMistralModel(config) # combine the vision module Llave (tower+projector) and the Mistral Language Model

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # maps hidden states → vocab logits, used in MistralForCausalLM.forward to produce token logits

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[str] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None) # at first is None
        attention_mask = kwargs.pop("attention_mask", None) # at first is None
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
