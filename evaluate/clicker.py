import torch
import numpy as np
import cv2

class MyClicker:
    def __init__(self, gt_mask = None):
        self.gt = gt_mask
        self.point_coord_list = []
        self.point_label_list = []

    def clear(self):
        self.gt = None
        self.point_coord_list.clear()
        self.point_label_list.clear()
    
    def reset(self, gt_mask):
        self.clear()
        self.gt = gt_mask

    def get_next_click(self, mask: np.ndarray):
        if mask is None:
            mask = np.zeros_like(self.gt)
        diff = np.logical_xor(mask, self.gt)
        diff_dt = cv2.distanceTransform(diff.astype(np.uint8), cv2.DIST_L2, 3)
        max_dt = np.max(diff_dt)
        click_coord_y, click_coord_x = np.where(diff_dt == max_dt)
        click_coord_y, click_coord_x = click_coord_y[0], click_coord_x[0]
        click_label = 1
        if mask[click_coord_y, click_coord_x] > 0:
            click_label = 0
        self.point_coord_list.append([click_coord_x, click_coord_y])
        self.point_label_list.append(click_label)