from torch import nn


# based on https://github.com/pytorch/examples/blob/master/dcgan/main.py#L117
class DCGANGenerator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3, output_size=64):
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.output_size = output_size
        assert output_size % 16 == 0

        super().__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution, of shape nz * 1 * 1
            nn.ConvTranspose2d(nz, ngf * 8, output_size // 16, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
