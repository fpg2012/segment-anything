import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from utils import build_sam_with_extrapolation

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
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

if __name__ == '__main__':
    image = cv2.imread('images/parrots.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_checkpoint = "../checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex

    # sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint, global_attention_div=1, window_attention_div=1)
    sam = build_sam_with_extrapolation(encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=sam_checkpoint,
        align_corners=True
        )
    sam.to(device=device)
    if device == 'xpu':
        sam = ipex.optimize(sam)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([[600, 500]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        ax[i].imshow(image)
        show_mask(mask, ax[i])
        show_points(input_point, input_label, ax[i])
        # ax[i].title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        # ax[i].axis('off')
    plt.show()

    input_point = np.array([[600, 500], [450, 100]])
    input_label = np.array([1, 1])
    mask_input = logits[2, :, :]
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        mask_input=mask_input[None, :, :],
    )

    plt.figure(figsize=(5,5))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.show()

    input_point = np.array([[600, 500], [450, 100], [900, 100], [200, 400]])
    input_label = np.array([1, 1, 0, 0])
    mask_input = logits[0, :, :]
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        mask_input=mask_input[None, :, :],
    )

    plt.figure(figsize=(5,5))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.show()