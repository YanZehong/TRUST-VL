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

import math
import re
import time
import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_misinformation_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
from transformers import BertTokenizer


class VeritasMetaModel:

    def __init__(self, config):
        super(VeritasMetaModel, self).__init__(config)
        
        self.num_query_token = 32 #8, 32, 64, 128, 256, 512
        self.visual_encoder_num_features = config.mm_hidden_size #config.mm_hidden_size: 1024, config.hidden_size: 5120
        if self.visual_encoder_num_features not in [config.mm_hidden_size, config.hidden_size]:
            self.vision_projection = nn.Linear(config.mm_hidden_size, self.visual_encoder_num_features)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True) #modify
            self.mm_projector = build_vision_projector(config) #modify
            
            self.tokenizer = self.init_tokenizer(truncation_side="left")
            self.misinformation_projector, self.query_tokens, self.language_projection, self.layernorm_vision = build_misinformation_projector(config, self.num_query_token, self.visual_encoder_num_features) #modify
            self.misinformation_projector.resize_token_embeddings(len(self.tokenizer))

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
            self.anyres_image_newline = nn.Parameter(
                torch.empty(config.mm_hidden_size, dtype=self.dtype)
            )

    def init_tokenizer(self, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

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
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            self.misinformation_projector, self.query_tokens, self.language_projection, self.layernorm_vision = build_misinformation_projector(config, self.num_query_token, self.visual_encoder_num_features) #modify
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
            
            for p in self.misinformation_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print(f"####pretrain_mm_mlp_adapter: {pretrain_mm_mlp_adapter}")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        # if pretrain_misinformation_mlp_adapter is not None:
        #     print(f"####pretrain_misinformation_mlp_adapter is {pretrain_misinformation_mlp_adapter}")
        #     misinformation_projector_weights = torch.load(pretrain_misinformation_mlp_adapter, map_location='cpu')
        #     def get_w(weights, keyword):
        #         return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

        #     self.misinformation_projector.load_state_dict(get_w(misinformation_projector_weights, 'misinformation_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]
    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class VeritasMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features_raw = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features_raw)
        return image_features, image_features_raw
    
    def prepare_inputs_labels_for_multimodal_anyres(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, source_ids, images, anyres_images, image_sizes=None, anyres_image_sizes=None
    ):
        
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, image_features_raw = self.encode_images(concat_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # images: torch.Size([2, 3, 336, 336])
            image_features, image_features_raw = self.encode_images(images) #image_features: torch.Size([bs, 576, 4096]), image_features_raw: torch.Size([bs, 576, 1024])
            

        if type(anyres_images) is list or anyres_images.ndim == 5: #anyres_images (bs*[]): [torch.Size([3, 3, 336, 336]), torch.Size([2, 3, 336, 336])]
            if type(anyres_images) is list:
                anyres_images = [x.unsqueeze(0) if x.ndim == 3 else x for x in anyres_images]
            concat_anyres_images = torch.cat([anyres_image for anyres_image in anyres_images], dim=0) #torch.Size([5, 3, 336, 336])
            anyres_split_sizes = [anyres_image.shape[0] for anyres_image in anyres_images] #[3, 2]
            encoded_anyres_image_features, encoded_anyres_image_features_raw = self.encode_images(concat_anyres_images) #torch.Size([5, 576, 4096]), torch.Size([5, 576, 1024])
            encoded_anyres_image_features_raw = torch.split(encoded_anyres_image_features_raw, anyres_split_sizes) #[torch.Size([3, 576, 1024]), torch.Size([2, 576, 1024])]
            anyres_image_features_raw = []
            for idx, anyres_image_feat in enumerate(encoded_anyres_image_features_raw):
                anyres_image_features_raw.append(anyres_image_feat) #[torch.Size([3, 576, 1024]), torch.Size([2, 576, 1024])]
            
            anyres_mm_patch_merge_type = "spatial_unpad" #getattr(self.config, "mm_patch_merge_type", "flat")
            anyres_image_aspect_ratio = "anyres_max_9" #getattr(self.config, "image_aspect_ratio", "square")
            anyres_image_grid_pinpoints = "(1x1),...,(6x6)"

            if anyres_mm_patch_merge_type == "flat":
                anyres_image_features_raw = [x.flatten(0, 1) for x in anyres_image_features_raw]
            elif anyres_mm_patch_merge_type.startswith("spatial"):
                new_anyres_image_features = []
                for image_idx, anyres_image_feature in enumerate(anyres_image_features_raw):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    if anyres_image_feature.shape[0] > 1:
                        # print(f"####anyres_image_feature.shape[0]: {anyres_image_feature.shape[0]}") #3
                        base_anyres_image_feature = anyres_image_feature[0] #torch.Size([576, 1024])
                        anyres_image_feature = anyres_image_feature[1:] #torch.Size([2, 576, 1024])
                        height = width = self.get_vision_tower().num_patches_per_side #24
                        assert height * width == base_anyres_image_feature.shape[0]

                        if "anyres_max" in anyres_image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", anyres_image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1)) #9
                        
                        if anyres_image_aspect_ratio == "anyres" or "anyres_max" in anyres_image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size #336
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(anyres_image_sizes[image_idx], anyres_image_grid_pinpoints, vision_tower_image_size) #2, 1
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            anyres_image_feature = anyres_image_feature.view(num_patch_height, num_patch_width, height, width, -1) #torch.Size([1, 2, 24, 24, 1024])
                        else:
                            anyres_image_feature = anyres_image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in anyres_mm_patch_merge_type:
                            anyres_image_feature = anyres_image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            anyres_image_feature = anyres_image_feature.flatten(1, 2).flatten(2, 3) 
                            anyres_image_feature = nn.functional.max_pool2d(anyres_image_feature, 2)
                            anyres_image_feature = anyres_image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in anyres_mm_patch_merge_type and "anyres_max" in anyres_image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = anyres_image_feature.shape[2] #24
                            anyres_image_feature = anyres_image_feature.permute(4, 0, 2, 1, 3).contiguous() #torch.Size([1024, 1, 24, 2, 24])
                            anyres_image_feature = anyres_image_feature.flatten(1, 2).flatten(2, 3) #torch.Size([1024, 24, 48])
                            anyres_image_feature = unpad_image(anyres_image_feature, anyres_image_sizes[image_idx]) #torch.Size([1024, 24, 36])
                            c, h, w = anyres_image_feature.shape #1, 576, 1024
                            times = math.sqrt(h * w / (max_num_patches * unit**2)) #0.408248290463863
                            
                            if times > 1.1:
                                anyres_image_feature = anyres_image_feature[None]
                                anyres_image_feature = nn.functional.interpolate(anyres_image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            anyres_image_feature = torch.cat((anyres_image_feature, self.model.anyres_image_newline[:, None, None].expand(*anyres_image_feature.shape[:-1], 1).to(anyres_image_feature.device)), dim=-1) #torch.Size([1024, 24, 37])
                            anyres_image_feature = anyres_image_feature.flatten(1, 2).transpose(0, 1) #torch.Size([888, 1024])
                        elif "unpad" in anyres_mm_patch_merge_type:
                            anyres_image_feature = anyres_image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            anyres_image_feature = anyres_image_feature.flatten(1, 2).flatten(2, 3)
                            anyres_image_feature = unpad_image(anyres_image_feature, anyres_image_sizes[image_idx])
                            anyres_image_feature = torch.cat((anyres_image_feature, self.model.anyres_image_newline[:, None, None].expand(*anyres_image_feature.shape[:-1], 1).to(anyres_image_feature.device)), dim=-1)
                            anyres_image_feature = anyres_image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            anyres_image_feature = anyres_image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            anyres_image_feature = anyres_image_feature.flatten(0, 3)
                        if "nobase" in anyres_mm_patch_merge_type:
                            pass
                        else:
                            anyres_image_feature = torch.cat((base_anyres_image_feature, anyres_image_feature), dim=0) #torch.Size([1464, 1024])
                        new_anyres_image_features.append(anyres_image_feature)
                    else: # single image operations
                        anyres_image_feature = anyres_image_feature[0]
                        if "unpad" in anyres_mm_patch_merge_type:
                            anyres_image_feature = torch.cat((anyres_image_feature, self.model.anyres_image_newline[None]), dim=0)
                        new_anyres_image_features.append(anyres_image_feature)
                anyres_image_features_raw = new_anyres_image_features # [torch.Size([1464, 1024]), torch.Size([1076, 1024])]
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {anyres_mm_patch_merge_type}")
        else:
            anyres_image_features, anyres_image_features_raw = self.encode_images(anyres_images)


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        
        anyres_max_len = max(x.shape[0] for x in anyres_image_features_raw)
        anyres_image_features_raw_padded = []
        for i, cur_anyres_image_embed in enumerate(anyres_image_features_raw):
            cur_len = cur_anyres_image_embed.shape[0]
            anyres_image_features_raw_padded.append(torch.cat((cur_anyres_image_embed, torch.zeros((anyres_max_len - cur_len, cur_anyres_image_embed.shape[1]), dtype=cur_anyres_image_embed.dtype, device=cur_anyres_image_embed.device)), dim=0))
        
        #  anyres_image_features_raw_padded: [torch.Size([1464, 1024]), torch.Size([1464, 1024])]
        anyres_image_features_raw = torch.stack(anyres_image_features_raw_padded, dim=0) #torch.Size([2, 1464, 1024])
        
        if self.get_model().visual_encoder_num_features != self.get_model().config.mm_hidden_size:
            anyres_image_features_raw = self.get_model().vision_projection(anyres_image_features_raw)
        # modify: use image_feature only for cross attention
        image_atts = torch.ones(anyres_image_features_raw.size()[:-1], dtype=torch.long).to(anyres_image_features_raw.device) #torch.Size([bs, 1464])
        query_tokens = self.get_model().query_tokens.expand(anyres_image_features_raw.shape[0], -1, -1) #torch.Size([bs, 32, 768])
        
        
        instruct_questions = []
        for batch_idx, cur_source_id in enumerate(source_ids):
            if cur_source_id == torch.tensor(0, dtype=torch.long):
                instruct_questions.append('Is there any visual misinformation?')
            elif cur_source_id == torch.tensor(1, dtype=torch.long):
                instruct_questions.append('Is there any cross-modal misinformation?')
            elif cur_source_id == torch.tensor(2, dtype=torch.long):
                instruct_questions.append('Is there any textual misinformation?')
            else:
                instruct_questions.append('Is there any misinformation?')

        text_Qformer = self.get_model().tokenizer(
            instruct_questions,
            padding='longest',
            # padding='max_length', #update
            truncation=True,
            # max_length=512,
            max_length=32, #update
            return_tensors="pt",
        ).to(anyres_image_features_raw.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(anyres_image_features_raw.device) #torch.Size([bs, 32])
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1) #torch.Size([bs, 32+16])

        image_features_layernorm = self.get_model().layernorm_vision(anyres_image_features_raw) #torch.Size([bs, 1464, 4096]) #update
        query_output = self.get_model().misinformation_projector.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_features_layernorm,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ) #query_output.last_hidden_state torch.Size([bs, 48, 768])
        
        misinformation_features = self.get_model().language_projection(query_output.last_hidden_state[:,:query_tokens.size(1),:]) #torch.Size([bs, 32, 4096])
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels #torch.Size([bs, 1711]), tensor([[-100, -100, -100,  ...,  383, 1296,    2]], device='cuda:0')
        _position_ids = position_ids #None
        _attention_mask = attention_mask # torch.Size([bs, 1711]), tensor([[True, True, True,  ..., True, True, True]], device='cuda:0')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool) 
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) #torch.Size([1711])
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # split into before and after image token(-200)
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] #[-1, 255, 1711]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) #torch.Size([1710, 4096])
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # tuple(torch.Size([255, 4096]), torch.Size([1455, 4096]))
            # cur_labels_noim tuple(torch.Size([255]), torch.Size([1455, 4096]))
            cur_new_input_embeds = []
            cur_new_labels = []
            #transfer the image token id into the image embedding
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx] #torch.Size([576, 4096])

                    cur_misinformation_features = misinformation_features[cur_image_idx] #modify, torch.Size([32, 4096])

                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(cur_misinformation_features) #modify
                    
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_labels.append(torch.full((cur_misinformation_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)) #modify
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds) #torch.Size([2286, 4096]), 2286=1710+576, after 2318=1710+576+32
            cur_new_labels = torch.cat(cur_new_labels) #torch.Size([2286]), after torch.Size([2318])

            new_input_embeds.append(cur_new_input_embeds) #list[torch.Size([2286, 4096])]
            new_labels.append(cur_new_labels) #list[torch.Size([2286])]
            
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None) #4096
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds) 

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)  #torch.Size([bs, 2286])
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device) #torch.Size([bs, 2286])
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)  #torch.Size([bs, 2286])

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0) #torch.Size([bs, 2286, 4096])

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded #torch.Size([bs, 2286])

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, source_ids,
        images, image_sizes=None,
    ):
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, image_features_raw = self.encode_images(concat_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, image_features_raw = self.encode_images(images) #torch.Size([bs, 576, 4096])

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        if self.get_model().visual_encoder_num_features == self.get_model().config.hidden_size:
            image_features_input = image_features
        elif self.get_model().visual_encoder_num_features == self.get_model().config.mm_hidden_size:
            image_features_input = image_features_raw
        # modify: use image_feature only for cross attention
        image_atts = torch.ones(image_features_input.size()[:-1], dtype=torch.long).to(image_features_input.device) #torch.Size([bs, 576])
        query_tokens = self.get_model().query_tokens.expand(image_features_input.shape[0], -1, -1) #torch.Size([bs, 32, 768])
        
        
        instruct_questions = []
        for batch_idx, cur_source_id in enumerate(source_ids):
            if cur_source_id == torch.tensor(0, dtype=torch.long):
                instruct_questions.append('Is there any visual misinformation?')
            elif cur_source_id == torch.tensor(1, dtype=torch.long):
                instruct_questions.append('Is there any cross-modal misinformation?')
            elif cur_source_id == torch.tensor(2, dtype=torch.long):
                instruct_questions.append('Is there any textual misinformation?')
            else:
                instruct_questions.append('Is there any misinformation?')

        text_Qformer = self.get_model().tokenizer(
            instruct_questions,
            # padding='longest',
            padding='max_length', #update
            truncation=True,
            # max_length=512,
            max_length=64, #update
            return_tensors="pt",
        ).to(image_features_input.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_features_input.device) #torch.Size([bs, 32])
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1) #torch.Size([bs, 43])

        image_features_layernorm = self.get_model().layernorm_vision(image_features_input) #torch.Size([bs, 576, 4096]) #update
        query_output = self.get_model().misinformation_projector.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_features_layernorm,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ) #query_output.last_hidden_state torch.Size([bs, 43, 768])
        
        
        misinformation_features = self.get_model().language_projection(query_output.last_hidden_state[:,:query_tokens.size(1),:]) #torch.Size([bs, 32, 4096])
        # atts_misinformation = torch.ones(misinformation_features.size()[:-1], dtype=torch.long).to(image_features.device) #torch.Size([bs, 32])
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels #torch.Size([bs, 1711]), tensor([[-100, -100, -100,  ...,  383, 1296,    2]], device='cuda:0')
        _position_ids = position_ids #None
        _attention_mask = attention_mask # torch.Size([bs, 1711]), tensor([[True, True, True,  ..., True, True, True]], device='cuda:0')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool) 
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) #torch.Size([1711])
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # split into before and after image token(-200)
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] #[-1, 255, 1711]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) #torch.Size([1710, 4096])
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # tuple(torch.Size([255, 4096]), torch.Size([1455, 4096]))
            # cur_labels_noim tuple(torch.Size([255]), torch.Size([1455, 4096]))
            cur_new_input_embeds = []
            cur_new_labels = []
            #transfer the image token id into the image embedding
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx] #torch.Size([576, 4096])

                    cur_misinformation_features = misinformation_features[cur_image_idx] #modify, torch.Size([32, 4096])

                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(cur_misinformation_features) #modify
                    
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_labels.append(torch.full((cur_misinformation_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)) #modify
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds) #torch.Size([2286, 4096]), 2286=1710+576, after 2318=1710+576+32
            cur_new_labels = torch.cat(cur_new_labels) #torch.Size([2286]), after torch.Size([2318])

            new_input_embeds.append(cur_new_input_embeds) #list[torch.Size([2286, 4096])]
            new_labels.append(cur_new_labels) #list[torch.Size([2286])]
            
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None) #4096
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds) 

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)  #torch.Size([bs, 2286])
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device) #torch.Size([bs, 2286])
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)  #torch.Size([bs, 2286])

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0) #torch.Size([bs, 2286, 4096])

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded #torch.Size([bs, 2286])

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
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
