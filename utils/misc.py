import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
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

def show_mask(mask, ax, random_color=False, gt=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif gt:
        color = np.array([255/255, 10/255, 10/255, 0.6])
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def get_edge(image: torch.Tensor, gaussian_kernel_size: int = 5, canny_low_thresh: int = 50, canny_high_thresh: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """
    returns gray_image and edges
    """
    color_image = image.permute(1, 2, 0).cpu().numpy()
    color_image = cv2.normalize(color_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    color_image = (color_image * 255).astype(np.uint8)
    color_image = cv2.GaussianBlur(color_image, (gaussian_kernel_size, gaussian_kernel_size), 0)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    low_threshold = canny_low_thresh
    high_threshold = canny_high_thresh
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return gray_image, edges

def get_patches_on_edge(edge: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(edge, 16, 16)

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
    use_canny_bias: bool = False,
    canny_bias: float = 8/3,
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
            use_canny_bias=use_canny_bias,
            canny_bias=canny_bias,
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
    save: bool=False,
    use_canny_bias: bool = False,
    canny_bias: float = 8/3,
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
        use_canny_bias=use_canny_bias,
        canny_bias=canny_bias,
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
    use_canny_bias: bool = False,
    canny_bias: float = 8/3,
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
            use_canny_bias=use_canny_bias,
            canny_bias=canny_bias,
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