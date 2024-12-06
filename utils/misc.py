import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
import os
from segment_anything.modeling import ImageEncoderViT
from functools import partial
import gc

from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.sam import Sam
from segment_anything.modeling.transformer import TwoWayTransformer

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
    
    def visualize(self, image: torch.Tensor, attn: torch.Tensor, name='vis', ri=20, rj=10):
        """
        image: a SINGLE image (C, H, W)
        attn: a (head, H*W, H*W) shaped tensor
        """

        scale = 2
        im_size = 1024 // scale
        patch_size = 16 // scale

        image_cv = (image.permute(1, 2, 0) / 4).repeat(4, 4, 1).cpu().numpy().copy()
        image_cv = cv2.resize(image_cv, (im_size * 4, im_size * 4))

        gc.collect()

        heatmap = attn[:, ri * 64 + rj, :].view(4, 4, 64, 64).permute(0, 2, 1, 3).reshape(64*4, 64*4).cpu().numpy().copy()
        heatmap = cv2.resize(heatmap, (im_size*4, im_size*4))

        gc.collect()

        image_cv[:, :, 0] += heatmap * 20.0
        image_cv[:, :, 1] += heatmap * 20.0
        image_cv = (np.clip(image_cv, 0, 1) * 255).astype(np.uint8)

        for i in range(4):
            for j in range(4):
                cv2.rectangle(image_cv, (rj * patch_size + i * im_size, ri * patch_size + j * im_size), (rj * patch_size + i * im_size + patch_size, ri * patch_size + j * im_size + patch_size), (0, 255, 0), 1)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.save_dir + name + '.jpg', image_cv)

def build_encoder_only(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size=None,
    checkpoint=None,
    save=False,
    global_attn_div: int = 1,
    window_attn_div: int = 1,
):
    prompt_embed_dim = 256
    image_size = 1024 if image_size is None else image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            save_attention=save,
            global_attn_div=global_attn_div,
            window_attn_div=window_attn_div,
        )
    image_encoder.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        filtered_state_dict = {}
        for key in state_dict.keys():
            if key.startswith('image_encoder.'):
                filtered_state_dict[key.removeprefix('image_encoder.')] = state_dict[key]
        image_encoder.load_state_dict(filtered_state_dict)
    return image_encoder

def build_encoder_only_with_extrapolation(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    align_corners=True,
    save: bool=False
):
    prompt_embed_dim = 256
    image_size = 2048
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder=ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14*2,
        out_chans=prompt_embed_dim,
        global_attn_div=4,
        window_attn_div=1,
        save_attention=save,
    )
    image_encoder.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        filtered_state_dict = {}
        for key in state_dict.keys():
            if key.startswith('image_encoder.'):
                new_key = str.removeprefix(key, 'image_encoder.')
                if key.endswith('pos_embed'):
                    param = state_dict[key].permute(0, 3, 1, 2)
                    param = F.interpolate(param, size=(128, 128), align_corners=True, mode='bilinear')
                    param = param.permute(0, 2, 3, 1)
                    filtered_state_dict[new_key] = param
                elif key.endswith('.7.attn.rel_pos_h') or new_key.endswith('.7.attn.rel_pos_w') \
                     or new_key.endswith('.15.attn.rel_pos_h') or new_key.endswith('.15.attn.rel_pos_w') \
                     or new_key.endswith('.23.attn.rel_pos_h') or new_key.endswith('.23.attn.rel_pos_w') \
                     or new_key.endswith('.31.attn.rel_pos_h') or new_key.endswith('.31.attn.rel_pos_w'):
                    param = state_dict[key].unsqueeze(0).unsqueeze(0)
                    param = F.interpolate(param, size=(255, 80), align_corners=align_corners, mode='bilinear')
                    param = param.squeeze(0).squeeze(0)
                    filtered_state_dict[new_key] = param
                elif key.endswith('.attn.rel_pos_h') or new_key.endswith('.attn.rel_pos_w'):
                    param = state_dict[key].unsqueeze(0).unsqueeze(0)
                    param = F.interpolate(param, size=(55, 80), align_corners=align_corners, mode='bilinear')
                    param = param.squeeze(0).squeeze(0)
                    filtered_state_dict[new_key] = param
                else:
                    filtered_state_dict[new_key] = state_dict[key]
            # else:
            #     filtered_state_dict[key] = state_dict[key]
        image_encoder.load_state_dict(filtered_state_dict)
        #sam = ipex.optimize_transformers(sam)
    return image_encoder

def build_sam_with_extrapolation(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    align_corners=True,
):
    prompt_embed_dim = 256
    image_size = 2048
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14*2,
            out_chans=prompt_embed_dim,
            global_attn_div=4,
            window_attn_div=1,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        filtered_state_dict = {}
        for key in state_dict.keys():
            if key.startswith('image_encoder.'):
                new_key = key
                if new_key.endswith('pos_embed'):
                    param = state_dict[key].permute(0, 3, 1, 2)
                    param = F.interpolate(param, size=(128, 128), align_corners=True, mode='bilinear')
                    param = param.permute(0, 2, 3, 1)
                    filtered_state_dict[new_key] = param
                elif new_key.endswith('.7.attn.rel_pos_h') or new_key.endswith('.7.attn.rel_pos_w') \
                     or new_key.endswith('.15.attn.rel_pos_h') or new_key.endswith('.15.attn.rel_pos_w') \
                     or new_key.endswith('.23.attn.rel_pos_h') or new_key.endswith('.23.attn.rel_pos_w') \
                     or new_key.endswith('.31.attn.rel_pos_h') or new_key.endswith('.31.attn.rel_pos_w'):
                    param = state_dict[key].unsqueeze(0).unsqueeze(0)
                    param = F.interpolate(param, size=(255, 80), align_corners=align_corners, mode='bilinear')
                    param = param.squeeze(0).squeeze(0)
                    filtered_state_dict[new_key] = param
                elif new_key.endswith('.attn.rel_pos_h') or new_key.endswith('.attn.rel_pos_w'):
                    param = state_dict[key].unsqueeze(0).unsqueeze(0)
                    param = F.interpolate(param, size=(55, 80), align_corners=align_corners, mode='bilinear')
                    param = param.squeeze(0).squeeze(0)
                    filtered_state_dict[new_key] = param
                else:
                    filtered_state_dict[key] = state_dict[key]
            else:
                filtered_state_dict[key] = state_dict[key]
        sam.load_state_dict(filtered_state_dict)
        #sam = ipex.optimize_transformers(sam)
    return sam