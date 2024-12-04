import torch
import numpy as np
from clicker import MyClicker
from segment_anything import sam_model_registry, SamPredictor
from datasets.dataset import MyDataset, DAVISDataset
import matplotlib.pyplot as plt

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

class MyEvaluator:

    def __init__(self, predictor: SamPredictor, clicker: MyClicker, dataset: MyDataset):
        self.clicker = clicker
        self.predictor = predictor
        self.dataset = dataset
        self.max_clicks = 12

    def calc_iou(self, masks: np.ndarray, gt: np.ndarray):
        union = np.logical_or(masks, gt)
        intersection = np.logical_and(masks, gt)
        ious = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))
        return ious
    
    def evaluate(self, visualize=False):
        mean_noc_85, mean_noc_90, mean_noc_95 = 0, 0, 0
        iou_series = np.zeros(self.max_clicks+1)
        for index, (img, gt) in enumerate(self.dataset):
            noc_85, noc_90, noc_95, ious = self.test_interactive(img, gt, visualize=visualize)
            print(f'index: {index}')
            print(f'Noc@85,90,95: {noc_85}, {noc_90}, {noc_95}')
            print(f'IoU: {ious}')
            mean_noc_85 += noc_85 / len(self.dataset) if noc_85 < 100 else 0
            mean_noc_90 += noc_90 / len(self.dataset) if noc_90 < 100 else 0
            mean_noc_95 += noc_95 / len(self.dataset) if noc_95 < 100 else 0
            iou_series += ious / len(self.dataset)
        return {
            'NoC@85': mean_noc_85,
            'NoC@90': mean_noc_90,
            'NoC@95': mean_noc_95,
            'iou_series': iou_series,
        }
    
    def test_interactive(self, img, gt, visualize=False):
        iou_series = np.zeros(self.max_clicks+1)
        self.predictor.set_image(img)

        self.clicker.reset(gt)
        self.clicker.get_next_click(None)

        masks, _, logits = self.predictor.predict(
            point_coords=np.array(self.clicker.point_coord_list),
            point_labels=np.array(self.clicker.point_label_list),
            multimask_output=True,
        )
        ious = self.calc_iou(masks, gt)
        max_i = np.argmax(ious)
        max_iou = ious[max_i]
        iou_series[:] = max_iou
        prev_mask, prev_logits = masks[max_i], logits[max_i]
        prev_logits = prev_logits[None, :, :]

        # init with an arbitrarily large number
        noc_85 = 233
        noc_90 = 233
        noc_95 = 233
        for noc in range(0, 20):
            if visualize:
                plt.imshow(img)
                show_mask(prev_mask, plt.gca())
                # show_mask(gt[None, :, :], plt.gca(), gt=True)
                show_mask(np.logical_xor(prev_mask, gt[None, :, :]), plt.gca(), gt=True)
                show_points(np.array(self.clicker.point_coord_list), np.array(self.clicker.point_label_list), plt.gca())
                plt.show()
            if noc + 1 < noc_85 and max_iou > 0.85:
                noc_85 = noc + 1
            if noc + 1 < noc_90 and max_iou > 0.90:
                noc_90 = noc + 1
            if noc + 1 < noc_95 and max_iou > 0.95:
                noc_95 = noc + 1
            if max_iou >= 0.99:
                break
            self.clicker.get_next_click(prev_mask)
            prev_mask, _, prev_logits = self.predictor.predict(
                point_coords=np.array(self.clicker.point_coord_list),
                point_labels=np.array(self.clicker.point_label_list),
                multimask_output=False,
                mask_input=prev_logits,
            )
            iou = self.calc_iou(prev_mask, gt)[0]
            prev_mask = prev_mask[0]
            if iou > max_iou:
                max_iou = iou
                iou_series[(noc + 1):] = max_iou
        return noc_85, noc_90, noc_95, iou_series

if __name__ == '__main__':
    dataset = DAVISDataset('../datasets/DAVIS/')
    sam = sam_model_registry["vit_h"](checkpoint="../checkpoints/sam_vit_h_4b8939.pth", global_attention_div=1, window_attention_div=1)
    device = 'xpu'
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
        sam = sam.to(device=device)
        sam = ipex.optimize(sam)
    else:
        sam = sam.to(device=device)
    predictor = SamPredictor(sam)
    clicker = MyClicker()
    evaluator = MyEvaluator(predictor, clicker, dataset)
    result = evaluator.evaluate()
    print(result)
    plt.plot(np.arange(0, evaluator.max_clicks+1), result['iou_series'])