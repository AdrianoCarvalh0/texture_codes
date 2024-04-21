import torch
import numpy as np
from skimage.morphology import skeletonize

def calculate_cldice(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    res_labels = torch.argmax(inputs, dim=1) # (Bs, C, H, W)
    # convert to bool
    result = res_labels==1

    values = []
    for idx, _ in enumerate(result):
        Vp = result[idx].squeeze().numpy(force=True)
        Vl = targets[idx].numpy(force=True)
        Sp = skeletonize(Vp)
        Sl = skeletonize(Vl)
        Tprec = np.sum(np.logical_and(Sp, Vl)) / np.sum(Sp)
        Tsens = np.sum(np.logical_and(Sl, Vp)) / np.sum(Sl)
        values.append(2*(Tprec*Tsens) / (Tprec+Tsens))

    return np.mean(values)
