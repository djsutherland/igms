import argparse
from functools import partial
import random
import shlex

import numpy as np
import torch


def pil(x, **kwargs):
    "Converts a tensor of images to a PIL.Image."
    from PIL import Image
    from torchvision.utils import make_grid

    kwargs.setdefault("normalize", True)
    kwargs.setdefault("range", (-1, 1))
    grid = make_grid(x, **kwargs).mul_(255).add_(0.5).clamp_(0, 255)
    array = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(array)


def fill_diagonal(X, val, inplace=None):
    # there's gotta be a better way to do this....
    if inplace is None:
        inplace = not X.requires_grad

    if inplace:
        if hasattr(X, "fill_diagonal_"):  # pytorch 1.2
            X.fill_diagonal_(val)
            return X
        else:
            X[torch.eye(X.shape[0], device=X.device, dtype=torch.uint8)] = val
    else:
        mask = X.new_ones()
        fill_diagonal(X, 0, inplace=True)
        return X * mask


def as_parameter(X, requires_grad=None, **kwargs):
    if X is None:
        return None
    elif isinstance(X, torch.nn.Parameter):
        return X
    else:
        X = torch.as_tensor(X, **kwargs)
        if requires_grad is None:
            requires_grad = X.requires_grad
        return torch.nn.Parameter(X, requires_grad=requires_grad)


def as_tensors(X, *rest):
    "Calls as_tensor on a bunch of args, all of the first's device and dtype."
    X = torch.as_tensor(X)
    return [X] + [
        None if r is None else torch.as_tensor(r, device=X.device, dtype=X.dtype)
        for r in rest
    ]


def get_optimizer(spec, **kwargs):
    "Get a torch.optim optimizer from a simple spec."
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
    "Set seeds for numpy and stdlib; for dataloader workers."
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % (2 ** 32))


################################################################################
# Some functions useful in argument parsing, etc.


def floats(s):
    return tuple(float(x) for x in s.split(","))


def float_or_none(s):
    return float(s) if s else None


def floats_or_none(s):
    return floats(s) if s else None


################################################################################
# argparse nicety


class ArgumentParser(argparse.ArgumentParser):
    """
    Some slightly nicer defaults for argparse to allow passing in arguments
    from files: make a file (e.g. `args.txt`) with lines like you were typing
    them at the command line, then use @args.txt in the command line to
    substitute them in as if you had typed them there.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("fromfile_prefix_chars", "@")
        super().__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, arg_line):
        return shlex.split(arg_line, comments=True)
