# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   freqvae.py
    Time:        2022/10/25 11:10:33
    Editor:      Figo
-----------------------------------
'''
import abc
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class freqvae_cifar(AbstractAutoEncoder):
    def __init__(self, d=16, z=2048, **kwargs):
        super(freqvae_cifar, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),  # B x 16 x 16 x 16
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),  # B x 32 x 8 x 8
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),  # B x 32 x 8 x 8
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),  # B x 32 x 8 x 8
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),  # B x 32 x 8 x 8
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),  # B x 32 x 8 x 8
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),  # B x 16 x 16 x 16
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),  # B x 3 x 32 x 32
        )

        self.xi_bn = nn.BatchNorm2d(3)

        self.f = 8  # Encoder????????
        self.d = d  # 32 feature-dim
        self.z = z  # ??????? (Gauss?????????)
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)  # B x 2048  ????
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)  # B x 2048  ????
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)  # B x 2048  ????????????

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)  # ??????
        return h, self.fc11(h1), self.fc12(h1)  # ori??????

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
             return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, with_latent=False):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.xi_bn(xi)
        if with_latent:
           return xi, hi_projected, mu, logvar, hi
        return xi, mu, logvar, hi
