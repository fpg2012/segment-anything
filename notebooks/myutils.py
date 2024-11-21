import torch
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from segment_anything.modeling import ImageEncoderViT
from functools import partial
import gc

class COCOMValDataset(Dataset):
    
    def __init__(self, dataset_dir, transform, image_size=1024, num_images=100):
        self.dataset_dir = dataset_dir
        self.num_images = num_images
        self.image_size = image_size
        self.image_name_list = self.load_images_list(self.num_images)
        self.transform = transform
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    def load_images_list(self, num_images: int):
        files = os.listdir(self.dataset_dir)
        images = []
        cnt = 0
        for fn in files:
            filename = os.fsdecode(fn)
            if cnt >= num_images:
                break
            if filename.endswith('.jpg'):
                images.append(self.dataset_dir + filename)
                cnt += 1
        return images

    def __len__(self):
        return self.num_images

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, index):
        image_path = self.image_name_list[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1)
        input_image_torch = self.preprocess(input_image_torch)
        return input_image_torch, index

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