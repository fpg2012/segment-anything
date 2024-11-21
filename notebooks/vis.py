import torch
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything.modeling import ImageEncoderViT
import gc
from myutils import *
from segment_anything.utils.transforms import ResizeLongestSide

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

class Evaluator:
    
    def __init__(self, encoder: ImageEncoderViT, dataset: Dataset, visualizer: Visualizer = None, topk = 100, batch_size: int = 2, device='cpu'):
        self.encoder: ImageEncoderViT = encoder
        self.dataloader: DataLoader = DataLoader(dataset, batch_size, shuffle=False)
        self.visualizer: Visualizer = visualizer
        self.device = device
        self.batch_size: int = batch_size
        self.avg_cosine_similarity = {}
        self.avg_topk_overlap = {}
        self.topk = topk

    def evaluate_with_images(self):
        with torch.no_grad():
            for (X, ind) in self.dataloader:
                self.encoder.clear_attention_saves()
                torch.xpu.empty_cache()
                X = X.to(self.device)
                self.encoder(X)
                attention_saves = self.encoder.attention_saves
                self.visualize(X, attention_saves, ind)
                self.calc_similarity(attention_saves)
            for key, cos_sim in self.avg_cosine_similarity.items():
                cos_sim /= len(self.dataloader.dataset)
                print(f'{key} cosine_similarity', cos_sim)
                torch.save(cos_sim, self.visualizer.save_dir + f'avg_cosine_similarity-{key}.pt')
                plt.imshow(cos_sim.cpu().numpy())
                plt.savefig(self.visualizer.save_dir + f'avg_cosine_similarity-{key}.pdf')
                plt.clf()
            for key, topk_overlap in self.avg_topk_overlap.items():
                topk_overlap /= len(self.dataloader.dataset)
                print(f'{key} topk_overlap', topk_overlap)
                torch.save(topk_overlap, self.visualizer.save_dir + f'avg_topk_overlap-{key}.pt')
                plt.imshow(topk_overlap.cpu().numpy())
                plt.savefig(self.visualizer.save_dir + f'avg_topk_overlap-{key}.pdf')
                plt.clf()
        
    def calc_similarity(self, attention_saves):
        print('...calculating similarity...')
        k = self.topk
        for key, attn in attention_saves.items():
            for i in range(self.batch_size):
                real_attn = attn[i*16:i*16+16]
                reshaped_attn = real_attn.view(16, -1).to(self.device)
                reshaped_attn = F.normalize(reshaped_attn, dim=1)
                cos_sim = reshaped_attn @ reshaped_attn.transpose(0, 1)
                if key not in self.avg_cosine_similarity.keys():
                    self.avg_cosine_similarity[key] = torch.zeros((16, 16)).to(self.device)
                self.avg_cosine_similarity[key] += cos_sim.to(self.device)
                del reshaped_attn

                attn_top, _ = real_attn.topk(k)
                rankk = attn_top[:, :, k-1].view(16, 4096, 1).expand(-1, -1, 4096)
                binary = (real_attn >= rankk).to(torch.float32).view(16, -1)
                topk_overlap = binary @ binary.transpose(0, 1) / 4096.0
                
                if key not in self.avg_topk_overlap.keys():
                    self.avg_topk_overlap[key] = torch.zeros((16, 16)).to(self.device)
                self.avg_topk_overlap[key] += topk_overlap.to(self.device)
                del binary
                
    def visualize(self, images, attention_saves, ind):
        print('...visualizing...')
        ri, rj = self.get_random_patch()
        for key, attn in attention_saves.items():
            for i in range(self.batch_size):
                self.visualizer.visualize(images[i], attn[i*16:i*16 + 16], name=f'img{ind[i]}-{key}', ri=ri, rj=rj)
    
    def get_random_patch(self):
        i, j = np.random.randint(5, 32), np.random.randint(16, 48)
        return i, j 

if __name__ == '__main__':
    sam_encoder = build_encoder_only(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=EV_CFG['sam_checkpoint'],
    )
    sam_encoder.to(device=EV_CFG['device'])

    gc.collect()
    
    image_size = 1024
    batch_size = EV_CFG['batch_size']

    dataset = COCOMValDataset(EV_CFG['dataset_path'], ResizeLongestSide(image_size), image_size=image_size, num_images=EV_CFG["num_images"])

    evaluator = Evaluator(sam_encoder, dataset, visualizer=Visualizer(EV_CFG['visualize_dir']), topk=EV_CFG['topk'], device=EV_CFG['device'])
    evaluator.evaluate_with_images()