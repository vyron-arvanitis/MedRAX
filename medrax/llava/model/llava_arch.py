#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import os
from glob import glob

import torch

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from medrax.llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config) # call next class MistralModel (builds LM)

        if hasattr(config, "mm_vision_tower"):
            # image encoder: take image
            
            # Vision tower = the image encoder part of the system.
            # It uses a pre-trained CLIP vision model to convert an input image into feature vectors.
            # Concretely: the image is split into patches and the model outputs a sequence of patch embeddings
            # (optionally with a CLS token). These embeddings are later projected into the LLM embedding space.
            # Note: delay_load=True means the weights are not loaded yet; preprocessing happens before this call.
            self.vision_tower = build_vision_tower(config, delay_load=True) # NOTE: [revisit]
  
            # Projector = the bridge from vision features to the LLM embedding space.
            # It maps CLIP output size (mm_hidden_size) → LLM hidden size, so image tokens can be inserted
            # alongside text tokens. Implemented as Linear or MLP depending on mm_projector_type.
            self.mm_projector = build_vision_projector(config)


    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, embed_tokens=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        # add additional configs for segtok
        self.config.feature_outs = model_args.feature_outs
        self.config.img_size = model_args.img_size
        self.config.vision_backbone = model_args.vision_backbone
        self.config.segtok_posembed = model_args.segtok_posembed

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # Initialize last layer in mm_projector with weight=0 and bias=mean(embed_tokens)
        if embed_tokens is not None:
            embed_tokens_weight = embed_tokens.weight.data
            self.mm_projector[-1].weight.data.zero_()
            self.mm_projector[-1].bias.data.copy_(embed_tokens_weight.mean(dim=0))

        if pretrain_mm_mlp_adapter is not None:

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))

            # also load additional learnable parameters during feature alignment
            checkpoint_folder = os.path.dirname(pretrain_mm_mlp_adapter)
            ckpts = glob(f"{checkpoint_folder}/checkpoint-*", recursive=False)
            if len(ckpts) > 0:
                vision_module_weights = torch.load(
                    f"{ckpts[-1]}/mm_projector.bin", map_location="cpu"
                )
                model_dict = get_w(vision_module_weights, "vision_tower")
                print(f"Loading vision module weights from {ckpts[-1]}/mm_projector.bin")
                # print keys in model_dict
                print(f"Loaded keys: {model_dict.keys()}")
                self.vision_tower.load_state_dict(model_dict, strict=False)


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        """
        Build LM-ready tensors by replacing IMAGE_TOKEN_INDEX placeholders with image embeddings.

        Shape legend:
        - B: batch size
        - T: padded text length before multimodal expansion
        - D: LM hidden size
        - V: vision token count per image after vision tower + projector
        - n_i: number of images for sample i

        Expected inputs:
        - input_ids: [B, T], contains IMAGE_TOKEN_INDEX markers.
        - attention_mask: [B, T] or None.
        - position_ids: [B, T] or None.
        - labels: [B, T] or None.
        - images:
          1) tensor [B, C, H, W] (one image per sample), or
          2) tensor [B, N, C, H, W] / list of length B with each item [n_i, C, H, W].

        Returns:
        - input_ids: None when multimodal embeddings are produced (caller should use inputs_embeds).
        - position_ids: [B, T_mm] or None.
        - attention_mask: [B, T_mm] or None.
        - past_key_values: passthrough.
        - inputs_embeds: [B, T_mm, D] or None.
        - labels: [B, T_mm] or None.

        T_mm is the per-batch max sequence length after replacing each image placeholder.
        For one sample with k image placeholders and text chunks [t_0, ..., t_k], final length is:
        sum(t_j) + sum(v_j), where v_j is the inserted image block length for placeholder j.

        Concrete example: how IMAGE_TOKEN_INDEX expands sequence length.

        Legend (and where each value is used/computed):
        - T: original unpadded token length, including placeholders (Step 3, after unpadding).
        - k: number of IMAGE_TOKEN_INDEX placeholders (Step 4: num_images).
        - t_j: j-th text chunk length between placeholders (Step 4: cur_input_ids_noim/split_sizes).
        - V: vision tokens inserted per placeholder (Step 1 output, inserted in Step 4).

        Example (single sample):
        We have k = 2 placeholders and text chunk lengths [4, 3, 1]. The unpadded token
        sequence (Step 3 view) is:

            [t t t t  <img>  t t t  <img>  t]

        Counts before replacement (Step 4 logic):
        - text tokens = 4 + 3 + 1 = 8
        - placeholders = k = 2
        - original unpadded length: T = 8 + 2 = 10

        Placeholder replacement (Step 4):
        Each <img> token is removed and replaced by a vision block of length V.
        If V = 576 for both placeholders:

            final_len = 4 + 576 + 3 + 576 + 1 = 1160
                    = sum(t_j) + k * V
                    = (4 + 3 + 1) + 2 * 576
                    = 8 + 1152
                    = 1160

        General formula per sample (Step 4):
            final_len = sum(t_j) + sum(v_j)

        If each placeholder uses the same V (Step 4):
            final_len = sum(t_j) + k * V

        Post-processing:
        - Step 5 may truncate final_len to tokenizer_model_max_length.
        - Step 6 pads all samples in the batch to T_mm = max(final_len_i).
        """
        ###################
        # Step 0 - doing early-exit checks for text-only or cache-only decoding
        ###################
        vision_tower = self.get_vision_tower()
        # Skip multimodal expansion when model/images are missing or generation is on one new token.
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # Cached generation step: align mask/position to cache length + current token.
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        ###################
        # Step 1 - doing image encoding and normalizing image feature layout
        ###################
        # Encode images and normalize to:
        # - tensor [B, V, D] for one-image-per-sample input, or
        # - list length B with each item [n_i * V, D] for multi-image input.
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)

        ###################
        # Step 2 - doing guard checks for unsupported IM_START/IM_END path
        ###################
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        ###################
        # Step 3 - doing optional-tensor normalization and text unpadding
        ###################
        # Save original optionals so we can restore None in outputs.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove text padding first so each sample becomes a variable-length 1D sequence.
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        ###################
        # Step 4 - doing per-sample placeholder replacement with image features
        ###################
        new_input_embeds = []
        new_labels = []
        # Points to the next image feature block to consume.
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # Keep a shared concat path by appending an empty [0, D] slice.
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                # Keep image/text batch alignment even when this sample has no placeholder token.
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )

            # Text-only chunks between placeholders: [t_0], [t_1], ..., [t_num_images].
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                # Interleave [text_i] and [image_i] so each placeholder is replaced.
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        ###################
        # Step 5 - doing truncation after multimodal expansion
        ###################
        # Image insertion can expand length; respect tokenizer max length if configured.
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        ###################
        # Step 6 - doing batch re-padding of multimodal embeddings, labels, and masks
        ###################
        # Re-pad to [B, T_mm, ...] for the LM.
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        ###################
        # Step 7 - doing output finalization and restoring original None semantics
        ###################
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

