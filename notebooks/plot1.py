import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_border_attn_drop(dir1, dir2, dir3, fig_name):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    bas7_1024 = torch.load(f'{dir1}/border_attention_sum-attn-block-7.pt').to('cpu').numpy()
    bas15_1024 = torch.load(f'{dir1}/border_attention_sum-attn-block-15.pt').to('cpu').numpy()
    bas23_1024 = torch.load(f'{dir1}/border_attention_sum-attn-block-23.pt').to('cpu').numpy()
    bas31_1024 = torch.load(f'{dir1}/border_attention_sum-attn-block-31.pt').to('cpu').numpy()

    bas7_2048 = torch.load(f'{dir2}/border_attention_sum-attn-block-7.pt').to('cpu').numpy()
    bas15_2048 = torch.load(f'{dir2}/border_attention_sum-attn-block-15.pt').to('cpu').numpy()
    bas23_2048 = torch.load(f'{dir2}/border_attention_sum-attn-block-23.pt').to('cpu').numpy()
    bas31_2048 = torch.load(f'{dir2}/border_attention_sum-attn-block-31.pt').to('cpu').numpy()

    bas7_2048_biased = torch.load(f'{dir3}/border_attention_sum-attn-block-7.pt').to('cpu').numpy()
    bas15_2048_biased = torch.load(f'{dir3}/border_attention_sum-attn-block-15.pt').to('cpu').numpy()
    bas23_2048_biased = torch.load(f'{dir3}/border_attention_sum-attn-block-23.pt').to('cpu').numpy()
    bas31_2048_biased = torch.load(f'{dir3}/border_attention_sum-attn-block-31.pt').to('cpu').numpy()

    # bas7_1024_i = torch.load('border_overlap_1_intel/border_attention_sum-attn-block-7.pt').to('cpu').numpy()
    # bas15_1024_i = torch.load('border_overlap_1_intel/border_attention_sum-attn-block-15.pt').to('cpu').numpy()
    # bas23_1024_i = torch.load('border_overlap_1_intel/border_attention_sum-attn-block-23.pt').to('cpu').numpy()
    # bas31_1024_i = torch.load('border_overlap_1_intel/border_attention_sum-attn-block-31.pt').to('cpu').numpy()

    # bas7_2048_i = torch.load('border_overlap_2_intel/border_attention_sum-attn-block-7.pt').to('cpu').numpy()
    # bas15_2048_i = torch.load('border_overlap_2_intel/border_attention_sum-attn-block-15.pt').to('cpu').numpy()
    # bas23_2048_i = torch.load('border_overlap_2_intel/border_attention_sum-attn-block-23.pt').to('cpu').numpy()
    # bas31_2048_i = torch.load('border_overlap_2_intel/border_attention_sum-attn-block-31.pt').to('cpu').numpy()

    x = np.arange(1, 16+1)
    width = 0.2

    ax[0, 0].bar(x - width, bas7_1024, width, label='image_size=1024')
    ax[0, 0].bar(x, bas7_2048, width, label='image_size=2048')
    ax[0, 0].bar(x + width, bas7_2048_biased, width, label='image_size=2048 (biased)')
    # ax[0, 0].bar(x + width/2*3, bas7_2048_i, width, label='image_size=2048i')
    ax[0, 0].set_title('Layer 7')
    # ax[0, 0].grid(axis='y')
    # ax[0, 0].set_xlabel('head')
    ax[0, 0].set_ylabel('attention sum')
    ax[0, 0].set_ylim((0.0, 1.0))
    ax[0, 0].set_xticks(x)
    ax[0, 0].legend()

    ax[0, 1].bar(x - width, bas15_1024, width, label='image_size=1024')
    ax[0, 1].bar(x, bas15_2048, width, label='image_size=2048')
    ax[0, 1].bar(x + width, bas15_2048_biased, width, label='image_size=2048 (biased)')
    # ax[0, 1].bar(x + width/2, bas15_1024_i, width, label='image_size=1024i')
    # ax[0, 1].bar(x + width/2*3, bas15_2048_i, width, label='image_size=2048i')
    ax[0, 1].set_title('Layer 15')
    # ax[0, 1].grid(axis='y')
    # ax[0, 1].set_xlabel('head')
    ax[0, 1].set_ylabel('attention sum')
    ax[0, 1].set_ylim((0.0, 1.0))
    ax[0, 1].set_xticks(x)
    ax[0, 1].legend()

    ax[1, 0].bar(x - width, bas23_1024, width, label='image_size=1024')
    ax[1, 0].bar(x, bas23_2048, width, label='image_size=2048')
    ax[1, 0].bar(x + width, bas23_2048_biased, width, label='image_size=2048 (biased)')
    # ax[1, 0].bar(x + width/2, bas23_1024_i, width, label='image_size=1024i')
    # ax[1, 0].bar(x + width/2*3, bas23_2048_i, width, label='image_size=2048i')
    ax[1, 0].set_title('Layer 23')
    # ax[1, 0].grid(axis='y')
    ax[1, 0].set_xlabel('head')
    ax[1, 0].set_ylabel('attention sum')
    ax[1, 0].set_ylim((0.0, 1.0))
    ax[1, 0].set_xticks(x)
    ax[1, 0].legend()

    ax[1, 1].bar(x - width, bas31_1024, width, label='image_size=1024')
    ax[1, 1].bar(x, bas31_2048, width, label='image_size=2048')
    ax[1, 1].bar(x + width, bas31_2048_biased, width, label='image_size=2048 (biased)')
    # ax[1, 1].bar(x + width/2, bas31_1024_i, width, label='image_size=1024i')
    # ax[1, 1].bar(x + width/2*3, bas31_2048_i, width, label='image_size=2048i')
    ax[1, 1].set_title('Layer 31')
    # ax[1, 1].grid(axis='y')
    ax[1, 1].set_xlabel('head')
    ax[1, 1].set_ylabel('attention sum')
    ax[1, 1].set_ylim((0.0, 1.0))
    ax[1, 1].set_xticks(x)
    ax[1, 1].legend()

    plt.savefig(fig_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1', type=str)
    parser.add_argument('--dir2', type=str)
    parser.add_argument('--dir3', type=str)
    parser.add_argument('--fig_name', type=str)
    args = parser.parse_args()
    plot_border_attn_drop(dir1=args.dir1, dir2=args.dir2, dir3=args.dir3, fig_name=args.fig_name)
