import inspect

from torch import nn

################################################################################
# Infrastructure for the make_generator() function.

_registry = {}
_expected_params = {"z_dim", "output_size", "output_channels"}


def register(original_cls=None, *, name=None):
    def decorator(cls):
        sig = inspect.signature(cls)
        arg_info = []
        for arg_name, param in sig.parameters.items():
            if arg_name in _expected_params:
                continue
            fn = param.annotation
            arg_info.append((arg_name, str if fn is inspect.Parameter.empty else fn))

        _registry[name or cls.__name__] = (cls, arg_info)
        return cls

    return decorator(original_cls) if original_cls else decorator


def make_generator(spec, **kwargs):
    parts = spec.split(":")
    cls, arg_info = _registry[parts[0]]
    assert len(parts) - 1 <= len(arg_info)
    for s, (name, parser) in zip(parts[1:], arg_info):
        kwargs[name] = parser(s)
    return cls(**kwargs)


################################################################################
# Actual generator classes.


class Generator(nn.Module):
    def __init__(
        self, z_dim: int = 128, output_channels: int = 3, output_size: int = 64
    ):
        self.z_dim = z_dim
        self.output_channels = output_channels
        self.output_size = output_size
        self.output_shape = (self.output_channels, self.output_size, self.output_size)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        else:
            assert len(input.shape) == 4
        return self.main(input)


# based on https://github.com/pytorch/examples/blob/master/dcgan/main.py#L117
@register(name="dcgan")
class DCGANGenerator(Generator):
    def __init__(
        self,
        z_dim: int = 128,
        output_channels: int = 3,
        output_size: int = 64,
        n_filters: int = 64,
    ):
        super().__init__(
            z_dim=z_dim, output_channels=output_channels, output_size=output_size
        )
        self.n_filters = n_filters
        assert self.output_size % 16 == 0

        super().__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution, of shape z_dim * 1 * 1
            nn.ConvTranspose2d(
                z_dim, n_filters * 8, output_size // 16, 1, 0, bias=False
            ),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (n_filters*8) x 4 x 4
            nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # state size. (n_filters*4) x 8 x 8
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(True),
            # state size. (n_filters*2) x 16 x 16
            nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # state size. (n_filters) x 32 x 32
            nn.ConvTranspose2d(n_filters, output_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (output_channels) x 64 x 64
        )

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        self.main.apply(weights_init)
