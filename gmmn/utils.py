import numpy as np
import torch


def pil(x, input_scale=(-1, 1)):
    from PIL import Image

    x = torch.as_tensor(x, device="cpu")

    lo, hi = input_scale
    x = (x - lo) * (255 / (hi - lo))

    x = x.byte().numpy()
    if len(x.shape) == 4:
        x = np.concatenate(x, axis=2)
    return Image.fromarray(x.transpose(1, 2, 0))


def as_tensors(X, *rest):
    X = torch.as_tensor(X)
    return [X] + [
        None if r is None else torch.as_tensor(r, device=X.device, dtype=X.dtype)
        for r in rest
    ]
