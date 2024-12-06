import numpy as np
from clicker import MyClicker
from segment_anything import sam_model_registry, SamPredictor
import sys
sys.path.append("..")
from utils.dataset import MyDataset, DAVISDataset
from utils.misc import show_mask, show_points, build_sam_with_extrapolation
import matplotlib.pyplot as plt
import argparse

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
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, default='vit_h')
    parser.add_argument('--result_file', type=str, default='result.svg')
    parser.add_argument('--enable_extrapolation', type=bool, default=False)
    args = parser.parse_args()

    model = args.model
    device = args.device
    checkpoint = args.checkpoint

    enable_extrapolation: bool = args.enable_extrapolation

    if not enable_extrapolation:
        sam = sam_model_registry[model](checkpoint=checkpoint, global_attention_div=1, window_attention_div=1)
        dataset = DAVISDataset('../datasets/DAVIS/')
    else:
        sam = build_sam_with_extrapolation(encoder_embed_dim=1280,
            encoder_depth=32,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[7, 15, 23, 31],
            checkpoint=checkpoint,
            align_corners=True
        )
        dataset = DAVISDataset('../datasets/DAVIS/')

    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
        sam.to(device=device)
        sam = ipex.optimize(sam)
    else:
        sam.to(device=device)
    predictor = SamPredictor(sam)
    clicker = MyClicker()
    evaluator = MyEvaluator(predictor, clicker, dataset)
    result = evaluator.evaluate()

    print(result)
    plt.plot(np.arange(0, evaluator.max_clicks+1), result['iou_series'])
    plt.savefig(args.result_file)