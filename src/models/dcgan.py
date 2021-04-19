import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngf=64, nc=1, nz=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, nc, 5, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.model(z)


def create_d_head(use_sigmoid: bool, ndf: int):
    if use_sigmoid:
        return nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    return nn.Sequential(
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2),
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
    )


def create_d_big_head(use_sigmoid: bool, ndf: int):
    if use_sigmoid:
        return nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )


class Discriminator(nn.Module):
    def __init__(self, use_sigmoid: bool, n_heads: int, use_big_head_d, nc=1, ndf=64):

        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.ConvTranspose2d(nc, ndf, 5, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
        )

        self.n = n_heads
        self.use_sigmoid = use_sigmoid
        if use_big_head_d:
            assert n_heads == 1
            setattr(self, f"head_{0}", create_d_big_head(use_sigmoid, ndf=ndf))
        else:
            for i in range(n_heads):
                setattr(self, f'head_{i}', create_d_head(use_sigmoid, ndf))

    def forward(self, x, head_id):
        s1 = self.shared_layers(x)
        if head_id == -1:
            s = 0
            for i in range(self.n):
                s += getattr(self, "head_%i" % i)(s1)
            return (s / self.n).squeeze()
        elif head_id >= 0 and head_id < self.n:
            return getattr(self, "head_%i" % head_id)(s1).squeeze()
        else:
            RuntimeError(f"Invalid head id: {head_id}")

# d1 = Discriminator(False, 1, True)
# d2 = Discriminator(False, 4, False)
# print(f'd1 parameters: {sum(param.numel() for param in d1.parameters())}')
# print(f'd2 parameters: {sum(param.numel() for param in d2.parameters())}')
# import torch
# z = torch.randn(10, 100)

# g = Generator()
# with torch.no_grad():
#     gz = g(z)
# with torch.no_grad():
#     p1 = d1(gz, -1)
