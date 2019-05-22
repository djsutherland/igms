from torch import nn


# based on https://github.com/pytorch/examples/blob/master/dcgan/main.py#L117
class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=128, n_filters=64, output_channels=3, output_size=64):
        self.z_dim = z_dim
        self.n_filters = n_filters
        self.output_channels = output_channels
        self.output_size = output_size
        assert output_size % 16 == 0

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

    def forward(self, input):
        return self.main(input)


def make_generator(spec, **kwargs):
    parts = spec.split(":")
    kind = parts.pop(0)
    if kind == "dcgan":
        if parts:
            kwargs["n_filters"] = int(parts.pop(0))
        assert not parts
        return DCGANGenerator(**kwargs)
    else:
        raise ValueError(f"unknown generator type {kind}")
