import os

import numpy as np
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader
from torchvision import transforms


class CelebA(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split="train",
        transform=None,
        target_transform=None,
        attr_query=None,
    ):
        self.path = path
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        super().__init__()

        s_pth = os.path.join(path, "list_eval_partition.txt")
        splits = pd.read_csv(s_pth, delim_whitespace=True, header=None, index_col=1)[0]

        a_pth = os.path.join(path, "Anno", "list_attr_celeba.txt")
        attr = pd.read_csv(a_pth, delim_whitespace=True, header=1, index_col=0) == 1

        if split == "all":
            self.filenames = list(splits)
            mask = np.full(splits.shape[0], True, dtype=bool)
        else:
            s = {"train": 0, "valid": 1, "test": 2}[split]
            self.filenames = list(splits[s])
            mask = splits.index == s

        if attr_query is not None:
            mask = mask & np.asarray(attr.eval(attr_query))

        self.filenames = splits[mask].values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr_names = list(attr.columns)

    def __getitem__(self, i):
        X = default_loader(
            os.path.join(self.path, "img_align_celeba", self.filenames[i])
        )
        if self.transform is not None:
            X = self.transform(X)

        y = self.attr[i, :]
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y

    def __len__(self):
        return self.filenames.shape[0]

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
    if kind == "celeba":
        kwargs["path"] = (
            os.path.expanduser(parts.pop(0) if parts else "") or "data/celebA"
        )
        if parts:
            kwargs["attr_query"] = parts.pop(0) or None
        if parts:
            kwargs["split"] = parts.pop(0)
        assert not parts
        return CelebA(**kwargs, transform=CelebA.default_transform(out_size=out_size))
    else:
        raise ValueError(f"Unknown dataset {kind}")
