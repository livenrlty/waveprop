import numpy as np
import matplotlib.pyplot as plt

import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')  # sets the device to be CPU
    return device

def to_tensor(array):
    tensor = torch.from_numpy(array).float()
    tensor = torch.movedim(tensor, -1, 1)
    return tensor

def to_numpy(tensor):
    array = torch.movedim(tensor, 1, -1).cpu().detach().numpy()
    return array
  
def plot_seq(seq):
    if seq.shape[-1] > 1:
        fig, ax = plt.subplots(nrows=1, ncols=seq.shape[-1],
                              figsize=(3*seq.shape[-1], 3))

        for i in range(seq.shape[-1]):
            ax[i].imshow(seq[..., i])
            ax[i].set_yticks([])
            ax[i].set_xticks([])

        fig.tight_layout()
    else:
        plt.imshow(seq[..., 0])
    plt.show()