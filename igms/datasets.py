import os

import numpy as np
import torch
from torchvision.datasets import CelebA as tv_CelebA
from torchvision import transforms


class CelebA(tv_CelebA):
    # https://github.com/pytorch/vision/pull/1008 does this
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(
            os.path.join(self.root, self.base_folder, "list_attr_celeba.txt")
        ) as f:
            _ = f.readline()
            self.attr_names = f.readline().split()

    @staticmethod
    def default_transform(out_size=64, max_crop=160, min_crop=140):
        return transforms.Compose(
            [
                transforms.CenterCrop(178),
                #     transforms.RandomCrop(160),
                transforms.RandomResizedCrop(
                    out_size, scale=(min_crop / 178, max_crop / 178), ratio=(1, 1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # from [0, 1] to [-1, 1]
            ]
        )


def get_dataset(spec, out_size, **kwargs):
    parts = spec.split(":")
    kind = parts.pop(0).lower()
    kwargs["root"] = os.path.expanduser(parts.pop(0) if parts else "") or "data"
    if kind == "celeba":
        if parts:
            kwargs["split"] = parts.pop(0)
        assert not parts
        if "transform" not in kwargs:
            kwargs["transform"] = CelebA.default_transform(out_size=out_size)
        return CelebA(**kwargs)
    else:
        raise ValueError(f"Unknown dataset {kind}")
