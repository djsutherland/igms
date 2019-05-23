from functools import partial

import numpy as np
import random
import torch


def pil(x, **kwargs):
    from PIL import Image
    from torchvision.utils import make_grid

    kwargs.setdefault("normalize", True)
    kwargs.setdefault("range", (-1, 1))
    grid = make_grid(x, **kwargs).mul_(255).add_(0.5).clamp_(0, 255)
    array = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(array)


def as_tensors(X, *rest):
    X = torch.as_tensor(X)
    return [X] + [
        None if r is None else torch.as_tensor(r, device=X.device, dtype=X.dtype)
        for r in rest
    ]


def get_optimizer(spec, **kwargs):
    parts = spec.split(":")
    kind = parts.pop(0)
    if kind == "adam":
        if parts:
            kwargs["lr"] = float(parts.pop(0))
        if parts:
            kwargs["betas"] = (float(b) for b in parts.pop(0).split(","))
        assert not parts
        return partial(torch.optim.Adam, **kwargs)
    else:
        raise ValueError(f"unknown kind {kind}")


def set_other_seeds(worker_id):
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % (2 ** 32))
