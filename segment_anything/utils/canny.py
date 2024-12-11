import torch
import cv2
import numpy as np
import torch.nn.functional as F

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