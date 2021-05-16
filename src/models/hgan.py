import torch
import torch.nn as nn


class Generator(torch.nn.Module):
    def __init__(self, input_dim=100, num_filters=[512, 256, 128, 64], output_dim=3):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                deconv = nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
            else:
                deconv = nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            # Initializer
            nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
            nn.init.constant_(deconv.bias, 0.0)

            # Batch normalization
            bn_name = 'bn' + str(i + 1)
            self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        nn.init.normal_(out.weight, mean=0.0, std=0.02)
        nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


class Discriminator_vanilla(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator_vanilla, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x).squeeze()


class Discriminator_f6(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator_f6, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 6, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 31 x 31

            nn.Conv2d(ndf, ndf * 2, 6, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 14 x 14

            nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 6, 2, 0, bias=False))

    def forward(self, x):
        return self.main(x).squeeze()


class Discriminator_f8(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator_f8, self).__init__()
        self.main = nn.Sequential(

            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 8, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 30 x 30

            nn.Conv2d(ndf, ndf * 2, 8, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 13 x 13

            nn.Conv2d(ndf * 2, ndf * 4, 8, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),

            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 4 x 4

            nn.Conv2d(ndf * 4, 1, 4, 2, 0, bias=False))

    def forward(self, x):
        return self.main(x).squeeze()


class Discriminator_f6_dense(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator_f6_dense, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 6, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 31 x 31

            nn.Conv2d(ndf, ndf * 2, 6, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 14 x 14

            nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True))

        self.linear = nn.Sequential(
            nn.Linear(ndf * 4 * 6 * 6, 1))

    def forward(self, x):
        output = self.main(x).view(x.size(0), -1)
        output = self.linear(output)

        return output.view(-1, 1).squeeze()


class Discriminator_f4_dense(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator_f4_dense, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True))

        self.linear = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, 1))

    def forward(self, x):
        output = self.main(x).view(x.size(0), -1)
        output = self.linear(output)

        return output.view(-1, 1).squeeze()


# discriminator with kernel size = 4 and stride = 3
class Discriminator_f4s3(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator_f4s3, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 21 x 21
            nn.Conv2d(ndf, ndf * 2, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 2 x 2
            nn.Conv2d(ndf * 4, 1, 4, 3, 1, bias=False))

    def forward(self, x):
        return self.main(x).squeeze()


class Discriminator_dense(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator_dense, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf) x 32 x 32 )
        self.linear = nn.Sequential(nn.Linear(ndf * 32 * 32, 1))

    def forward(self, x):
        x = self.main(x)
        return self.linear(x.view(x.size(0), -1)).squeeze()


class Discriminator_f16(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator_f16, self).__init__()
        self.main = nn.Sequential(

            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 16, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 26 x 26

            nn.Conv2d(ndf, ndf * 2, 16, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7

            nn.Conv2d(ndf * 2, 1, 7, 2, 0, bias=False))

    def forward(self, x):
        return self.main(x).squeeze()


class Discriminator(nn.Module):
    def __init__(self, use_sigmoid: bool, n=8):
        assert n == 8
        super(Discriminator, self).__init__()

        self.n = n
        self.head_0 = Discriminator_vanilla(ndf=64, nc=3)
        self.head_1 = Discriminator_f6(ndf=64, nc=3)
        self.head_2 = Discriminator_f8(ndf=32, nc=3)
        self.head_3 = Discriminator_f6_dense(ndf=16, nc=3)
        self.head_4 = Discriminator_f4_dense(ndf=64, nc=3)
        self.head_5 = Discriminator_f4s3(ndf=64, nc=3)
        self.head_6 = Discriminator_dense(ndf=64, nc=3)
        self.head_7 = Discriminator_f16(ndf=16, nc=3)
        if use_sigmoid:
            self.last = nn.Sigmoid()
        else:
            self.last = nn.Identity()
    
    def forward(self, img, head_id):
        if head_id == -1:
            s = 0
            for i in range(self.n):
                s += self.last(getattr(self, "head_%i" % i)(img))
            return (s / self.n).squeeze()
        elif head_id >= 0 and head_id < self.n:
            return self.last(getattr(self, "head_%i" % head_id)(img).squeeze())
        else:
            RuntimeError(f"Invalid head id: {head_id}")
