import os

from torchvision import datasets as tv_datasets
from torchvision import transforms

from .utils import str2bool

_registry = {}


class CelebA(tv_datasets.CelebA):
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


_registry["celeba"] = (CelebA, CelebA.default_transform, [("split", str)])


def default_mnist_transform(out_size=28):
    return transforms.Compose(
        [
            transforms.Resize((out_size, out_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # from [0, 1] to [-1, 1]
        ]
    )


_registry["mnist"] = (tv_datasets.MNIST, default_mnist_transform, [("train", str2bool)])


def get_dataset(spec, out_size, transform_args={}, **kwargs):
    parts = spec.split(":")
    kind = parts.pop(0).lower()
    if kind not in _registry:
        raise ValueError(f"Unknown dataset {kind}")

    if "root" not in kwargs:
        kwargs["root"] = os.path.expanduser(parts[0] if parts else "") or "data"

    cls, transform, arg_info = _registry[kind]
    assert len(parts) <= len(arg_info)
    for (arg_name, parser), val in zip(arg_info, parts):
        kwargs[arg_name] = parser(val)

    if "transform" not in kwargs:
        kwargs["transform"] = transform(out_size=out_size, **transform_args)

    return cls(**kwargs)
