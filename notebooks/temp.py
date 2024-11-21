from myutils import *

EV_CFG = {
    "sam_checkpoint": "../checkpoints/sam_vit_h_4b8939.pth",
    "model_type": "vit_h",
    "device": "xpu",
    "dataset_path": '../datasets/COCO_MVal/img/',
    "batch_size": 2,
    "num_images": 100,
    'visualize_dir': './vis_result/',
    "topk": 100,
}

sam_encoder = build_encoder_only(
    encoder_embed_dim=1280,
    encoder_depth=32,
    encoder_num_heads=16,
    encoder_global_attn_indexes=[7, 15, 23, 31],
    image_size=2048,
    checkpoint=EV_CFG['sam_checkpoint'],
)