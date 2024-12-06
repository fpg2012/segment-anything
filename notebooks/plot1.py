import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_border_attn_drop():
    # plt.style.use('ggplot')
    fig, ax = plt.subplots(2, 2)
    bas7_1024 = torch.load('border_overlap_1/border_attention_sum-attn-block-7.pt').to('cpu').numpy()
    bas15_1024 = torch.load('border_overlap_1/border_attention_sum-attn-block-15.pt').to('cpu').numpy()
    bas23_1024 = torch.load('border_overlap_1/border_attention_sum-attn-block-23.pt').to('cpu').numpy()
    bas31_1024 = torch.load('border_overlap_1/border_attention_sum-attn-block-31.pt').to('cpu').numpy()

    bas7_2048 = torch.load('border_overlap_2/border_attention_sum-attn-block-7.pt').to('cpu').numpy()
    bas15_2048 = torch.load('border_overlap_2/border_attention_sum-attn-block-15.pt').to('cpu').numpy()
    bas23_2048 = torch.load('border_overlap_2/border_attention_sum-attn-block-23.pt').to('cpu').numpy()
    bas31_2048 = torch.load('border_overlap_2/border_attention_sum-attn-block-31.pt').to('cpu').numpy()

    x = np.arange(1, 16+1)
    width = 0.35

    ax[0, 0].bar(x - width/2, bas7_1024, width, label='image_size=1024')
    ax[0, 0].bar(x + width/2, bas7_2048, width, label='image_size=2048')
    ax[0, 0].set_title('Layer 7')
    ax[0, 0].grid(axis='y')
    # ax[0, 0].set_xlabel('head')
    ax[0, 0].set_ylabel('attention sum')
    ax[0, 0].set_ylim((0.0, 1.0))
    ax[0, 0].set_xticks(x)
    ax[0, 0].legend()

    ax[0, 1].bar(x - width/2, bas15_1024, width, label='image_size=1024')
    ax[0, 1].bar(x + width/2, bas15_2048, width, label='image_size=2048')
    ax[0, 1].set_title('Layer 15')
    ax[0, 1].grid(axis='y')
    # ax[0, 1].set_xlabel('head')
    ax[0, 1].set_ylabel('attention sum')
    ax[0, 1].set_ylim((0.0, 1.0))
    ax[0, 1].set_xticks(x)
    ax[0, 1].legend()

    ax[1, 0].bar(x - width/2, bas23_1024, width, label='image_size=1024')
    ax[1, 0].bar(x + width/2, bas23_2048, width, label='image_size=2048')
    ax[1, 0].set_title('Layer 23')
    ax[1, 0].grid(axis='y')
    ax[1, 0].set_xlabel('head')
    ax[1, 0].set_ylabel('attention sum')
    ax[1, 0].set_ylim((0.0, 1.0))
    ax[1, 0].set_xticks(x)
    ax[1, 0].legend()

    ax[1, 1].bar(x - width/2, bas31_1024, width, label='image_size=1024')
    ax[1, 1].bar(x + width/2, bas31_2048, width, label='image_size=2048')
    ax[1, 1].set_title('Layer 31')
    ax[1, 1].grid(axis='y')
    ax[1, 1].set_xlabel('head')
    ax[1, 1].set_ylabel('attention sum')
    ax[1, 1].set_ylim((0.0, 1.0))
    ax[1, 1].set_xticks(x)
    ax[1, 1].legend()

    plt.show()

if __name__ == '__main__':
    plot_border_attn_drop()
