import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch.nn.functional as F
import numpy as np

class MyDataset(Dataset):

    def __init__(self, image_size=1024):
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.image_size = image_size

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

class COCOMValDataset(MyDataset):
    
    def __init__(self, dataset_dir, transform, image_size=1024, num_images=100):
        super().__init__(image_size)
        self.dataset_dir = dataset_dir
        self.num_images = num_images
        self.image_name_list = self.load_images_list(self.num_images)
        self.transform = transform
    
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

    def __getitem__(self, index):
        image_path = self.image_name_list[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1)
        input_image_torch = self.preprocess(input_image_torch)
        return input_image_torch, index
    
class DAVISDataset(MyDataset):
    def __init__(self, dataset_dir, image_size=1024, transform=None):
        super().__init__(image_size)
        self.dataset_dir = dataset_dir
        self.filename_list = []
        self.transform = transform
        self._load_image_list()
        
    def _load_image_list(self):
        files = os.listdir(self.dataset_dir + 'img/')
        for fn in files:
            filename = os.fsdecode(fn)
            if filename.endswith('.jpg'):
                self.filename_list.append(str.removesuffix(filename, '.jpg'))
    
    def _merge_all_masks(self, gt: np.ndarray):
        gt = np.max(gt, axis=2)
        return gt
    
    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        filename = self.filename_list[index]

        img = cv2.imread(self.dataset_dir + 'img/' + filename + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = torch.as_tensor(img).permute(2, 0, 1)

        gt = cv2.imread(self.dataset_dir + 'gt/' + filename + '.png')
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = gt.astype(np.uint8)
        gt = self._merge_all_masks(gt) # an image in DAVIS may contain 1~4 masks; TODO 
        gt = gt.astype(bool)
        # gt = torch.as_tensor(gt)
        return img, gt

