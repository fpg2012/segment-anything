import torch
import gc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

from segment_anything.utils.transforms import ResizeLongestSide
import sys
sys.path.append("..")
from utils.misc import ImageEncoderViT, build_encoder_only_with_extrapolation, build_encoder_only
from utils.dataset import COCOMValDataset

sys.path.append("notebooks")
EV_CFG = {
    "sam_checkpoint": "../checkpoints/sam_vit_h_4b8939.pth",
    "model_type": "vit_h",
    "device": "cuda:3",
    "dataset_path": '../datasets/COCO_MVal/img/',
    "batch_size": 1,
    "num_images": 100,
    'attn_dir': './attention/',
    "topk": 100,
    "border_overlap_save_dir": './border_overlap/',
    "canny_low_thresh": 45,
    "canny_high_thresh": 145,
    "gaussian_kernel_size": 5,
    "image_size": 1024,
}

class Evaluator:
    
    def __init__(self, encoder: ImageEncoderViT, dataset: Dataset, batch_size: int = 2, save_dir='./', 
                 topk=100, border_overlap_save_dir='./', canny_low_thresh=50, canny_high_thresh=150, gaussian_kernel_size=5,
                 device='cpu', image_size=1024, **params):
        self.encoder: ImageEncoderViT = encoder
        self.dataloader: DataLoader = DataLoader(dataset, batch_size, shuffle=False)
        self.device = device
        self.batch_size: int = batch_size
        self.save_dir = save_dir
        self.image_size = image_size

        self.canny_low_thresh = canny_low_thresh
        self.canny_high_thresh = canny_high_thresh
        self.gaussian_kernel_size = gaussian_kernel_size
        self.topk = topk
        self.border_overlap_save_dir = border_overlap_save_dir
        
        self.attention_saves = {}

        self.overlap_size = {}
        self.border_attention_sum = {}
        self.overlap_attention_sum = {}

    def evaluate_with_images(self):
        cnt = 0
        to_tensor = ToTensor()
        with torch.no_grad():
            for (X, ind) in self.dataloader:
                print(f'batch {ind}')
                self.encoder.clear_attention_saves()
                self.encoder.init_attention_saves()
                torch.xpu.empty_cache()
                # for i in range(self.batch_size):
                #     self.show_image_and_edge(X[i])
                
                X = X.to(self.device)
                _, edge = self.get_edge(X[0])
                edge_tensor = to_tensor(edge).to(self.device)
                patches_on_edge = self.patch_on_edge(edge_tensor).view(1, self.image_size // 16, self.image_size // 16)
                self.encoder.set_sum_mask(patches_on_edge)

                self.encoder(X)
                
                sts = self.encoder.statistics_saves['subset_sum']
                for key, value in sts.items():
                    if key not in self.border_attention_sum.keys():
                        self.border_attention_sum[key] = sts[key]
                    else:
                        self.border_attention_sum[key] += sts[key]

            self._save_stats(self.border_attention_sum, 'border_attention_sum')
    
    def _save_stats(self, st: dict, name):
        for key, result in st.items():
            result /= len(self.dataloader.dataset)
            torch.save(result, self.border_overlap_save_dir + f'{name}-{key}.pt')

            plt.bar(range(16), result.cpu().numpy())
            plt.ylim((0.0, 1.0))
            plt.title(f'{name}-{key}')
            plt.xlabel('head')
            plt.ylabel(f'{name}')
            plt.grid(True)
            plt.savefig(self.border_overlap_save_dir + f'{name}-{key}.svg')
            plt.clf()

    def calc_edge_overlap(self, images: torch.Tensor):
        to_tensor = ToTensor()
        k = self.topk
        for key, attn in self.attention_saves.items():
            # attn: (B*Head, 4096, 4096)
            for i in range(self.batch_size):
                real_attn = attn[i] # (Head=16, Q=4096, K=4096)

                _, edge = self.get_edge(images[i])
                edge_tensor =  to_tensor(edge).to(self.device) # (1, H, W)
                patches_on_edge = self.patch_on_edge(edge_tensor) # (1, H/16, W/16) = (1, 64, 64)
                del edge_tensor
                del edge
                patches_on_edge = patches_on_edge.view(1, 1, 4096).expand(-1, 4096, -1)
                
                attn_top, _ = real_attn.topk(k)
                # (16, 4096) => (16, 4096, 1) ==repeat=> (16, 4096, 4096)
                rankk = attn_top[:, :, k-1].view(16, 4096, 1).expand(-1, -1, 4096)
                # topk elements becomes 1, other elements becomes 0
                binary = (real_attn >= rankk).to(torch.float32)
                overlap = binary * patches_on_edge
                overlap_size = torch.sum(overlap, dim=(1, 2)) / (100 * 4096) # (16)
                del binary
                border_attention_sum = torch.sum(real_attn * patches_on_edge, dim=(1, 2)) / (4096)
                overlap_attention_sum = torch.sum(real_attn * overlap, dim=(1, 2)) / (4096)
                
                if key not in self.overlap_size.keys():
                    self.overlap_size[key] = overlap_size
                else:
                    self.overlap_size[key] += overlap_size
                if key not in self.border_attention_sum.keys():
                    self.border_attention_sum[key] = border_attention_sum
                else:
                    self.border_attention_sum[key] += border_attention_sum
                if key not in self.overlap_attention_sum.keys():
                    self.overlap_attention_sum[key] = overlap_attention_sum
                else:
                    self.overlap_attention_sum[key] += overlap_attention_sum

    def save_pickle(self, ind):
        print(f'... saving {ind}')
        for key, attn in self.attention_saves.items():
            torch.save(attn, f'{self.save_dir}/{key}-{ind}.pt')
        self.attention_saves = {}
        gc.collect()
    
    def get_edge(self, image: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        color_image = image.permute(1, 2, 0).cpu().numpy()
        color_image = cv2.normalize(color_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        color_image = (color_image * 255).astype(np.uint8)
        color_image = cv2.GaussianBlur(color_image, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        low_threshold = self.canny_low_thresh
        high_threshold = self.canny_high_thresh
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)
        return gray_image, edges
    
    def show_image_and_edge(self, image: torch.Tensor):
        gray_image, edges = self.get_edge(image)
        combined = cv2.vconcat([gray_image, edges])
        combined = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        transform = ToTensor()
        edge_tensor = transform(edges)
        result = self.patch_on_edge(edge_tensor)[0]

        result_image = result.cpu().numpy()
        result_image = cv2.normalize(result_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        result_image = (result_image * 255).astype(np.uint8)
        result_image = cv2.resize(result_image, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST)

        cv2.imshow('Canny result', combined)
        cv2.imshow('Patch on Edge', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def patch_on_edge(self, edge: torch.Tensor) -> torch.Tensor:
        result = F.max_pool2d(edge, 16, 16)
        # result = F.avg_pool2d(edge, 16, 16)
        return result

if __name__ == '__main__':
    sam_encoder = build_encoder_only(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=EV_CFG['sam_checkpoint'],
        save=True,
    )
    sam_encoder.to(device=EV_CFG['device'])
    
    image_size = EV_CFG["image_size"]
    batch_size = EV_CFG['batch_size']

    dataset = COCOMValDataset(EV_CFG['dataset_path'], ResizeLongestSide(image_size), image_size=image_size, num_images=EV_CFG["num_images"])

    evaluator = Evaluator(sam_encoder, dataset, **EV_CFG)
    evaluator.evaluate_with_images()
