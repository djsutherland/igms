import numpy as np
import torch


def pil(x):
    from PIL import Image

    # [-1, 1] to [0, 255]|
    a = ((torch.as_tensor(x, device="cpu") + 1) * (255 / 2)).byte().numpy()
    if len(a.shape) == 4:
        a = np.concatenate(a, axis=2)
    return Image.fromarray(a.transpose(1, 2, 0))
