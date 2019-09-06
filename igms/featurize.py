from copy import copy
import os
import types

import torch
from torch import nn
from torchvision import models

try:
    from torch.nn import Identity
except ImportError:  # added in torch 1.1

    class Identity(nn.Module):
        def forward(self, input):
            return input


################################################################################
# Normalization layer, since we need to do this to generator samples instead of
# on loading the data.

# These are for data in [0, 1]
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STDDEV = (0.229, 0.224, 0.225)


class NormalizeLayer(torch.nn.Module):
    def __init__(self, means=_IMAGENET_MEAN, sds=_IMAGENET_STDDEV, **kwargs):
        super().__init__()
        self.register_buffer("means", torch.as_tensor(means, **kwargs))
        self.register_buffer("sds", torch.as_tensor(sds, **kwargs))

    def forward(self, input):
        assert len(input.shape) == 4
        cent = input - self.means[None, :, None, None]
        return cent / self.sds[None, :, None, None]


################################################################################
# Registry of forward() monkey patches to extract features instead of getting
# classification outputs.

_registry = {}


def extractor(cls, kind):
    def decorator(f):
        _registry[cls, kind] = f
        return f

    return decorator


def patch(model, kind):
    try:
        f = _registry[type(model), kind]
    except KeyError:
        raise TypeError(f"Don't know how to patch {kind} for type {type(model)}")

    model.forward = types.MethodType(f, model)


@extractor(models.ResNet, "end")
def resnet_end(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    # x = self.fc(x)
    return x


@extractor(models.ResNet, "through")
def resnet_through(self, x):
    bits = []

    def add():
        bits.append(x.view(x.size(0), -1))

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    add()

    x = self.layer1(x)
    add()
    x = self.layer2(x)
    add()
    x = self.layer3(x)
    add()
    x = self.layer4(x)
    add()

    x = self.avgpool(x)
    add()
    # x = x.view(x.size(0), -1)
    # x = self.fc(x)

    return torch.cat(bits, 1)


@extractor(models.VGG, "through")
@extractor(models.AlexNet, "through")
def vgg_through(self, x):
    bits = []

    def add():
        bits.append(x.view(x.size(0), -1))

    # x = self.features(x)
    for layer in self.features:
        x = layer(x)
        if isinstance(layer, nn.MaxPool2d):
            add()

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)

    # x = self.classifier(x)
    for layer in self.classifier:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            add()

    return torch.cat(bits, 1)


@extractor(models.VGG, "end")
@extractor(models.AlexNet, "end")
def vgg_end(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    # x = self.classifier(x)
    x = self.classifier[:-1](x)
    return x


@extractor(Identity, "through")
@extractor(Identity, "end")
def identity_extractor(self, x):
    return x.view(x.size(0), -1)


################################################################################
# Main method to load stuff.


def make_featurizer(model, kind="through", input_scale=(-1, 1)):
    """
    Makes a featurizer that *shares data* with model, but whose forward()
    computes features instead of the final output, according to what's
    registered with @extractor above.

    Assumes that model wants input like torchvision models, i.e. z-scored
    according to IMAGENET_MEAN, _IMAGENET_STDEV above, but takes input inside
    input_scale (default [-1, 1]). If input_scale is None, doesn't do any
    scaling.

    Calls model.eval() and sets requires_grad to False for parameters inside
    `model` as well, since there doesn't seem to be a way to do this only on
    the copy.
    """
    featurizer = copy(model)
    patch(featurizer, kind)
    featurizer.eval()  # v important for batch norm!
    for p in featurizer.parameters():
        p.requires_grad = False

    if input_scale is None:
        return featurizer

    lo, hi = input_scale
    scale = hi - lo
    means = [m * scale + lo for m in _IMAGENET_MEAN]
    stds = [s * scale for s in _IMAGENET_STDDEV]
    norm_layer = NormalizeLayer(means, stds)
    return nn.Sequential(norm_layer, featurizer)


def load_smoothing_imagenet_model(noise_level, **load_args):
    import tarfile

    # load their checkpoint
    folder = os.path.join(torch.hub._get_torch_home(), "checkpoints")
    os.makedirs(folder, exist_ok=True)

    tar_fn = "locuslab-smoothing.tar"
    if not os.path.exists(os.path.join(folder, tar_fn)):
        from torchvision.datasets import utils

        utils.download_file_from_google_drive(
            "1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4", folder, filename=tar_fn
        )

    with tarfile.open(os.path.join(folder, tar_fn), "r") as tar:
        fn = f"models/imagenet/resnet50/noise_{noise_level:.2f}/checkpoint.pth.tar"
        checkpoint = torch.load(tar.extractfile(fn), **load_args)

    # they checkpointed the model inside Sequential(DataParallel(model))
    def rewrite(k):
        assert k.startswith("1.module.")
        return k[9:]

    assert checkpoint["arch"] == "resnet50"
    sd = {rewrite(k): v for k, v in checkpoint["state_dict"].items()}

    model = models.resnet50(pretrained=False)
    model.load_state_dict(sd)
    return model


def load_featurizer(spec, input_scale=(-1, 1), **load_args):
    feats = []
    for subspec in spec.split("+"):
        parts = subspec.split(":")

        f_kind = parts.pop(0) if parts else "through"

        model_name = parts.pop(0) if parts else "smoothing"
        if model_name == "smoothing":
            noise_level = float(parts.pop(0)) if parts else 0.25
            model = load_smoothing_imagenet_model(noise_level=noise_level, **load_args)
        elif model_name == "identity":
            model = Identity()
        else:
            if not hasattr(models, model_name):
                raise ValueError(f"unknown model type {model_name}")
            model = getattr(models, model_name)(pretrained=True)

        assert not parts
        feats.append(make_featurizer(model, kind=f_kind, input_scale=input_scale))

    return feats[0] if len(feats) == 1 else Parallel(*feats)


class Parallel(nn.Sequential):
    """
    A parallel container: like torch.nn.Sequential, but concats the outputs
    of the subparts together.
    """

    def forward(self, input):
        return torch.cat([module(input) for module in self._modules.values()], 1)
