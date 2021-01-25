from torch import nn
from torchvision import models

# generatorの1ブロックを示す
class ResidualBlock(nn.Module):
    def __init__(self, nf=24):
        super(ResidualBlock, self).__init__()
        self.Block = nn.Sequential(
            # 畳み込み
            # バッチノーマライゼーション
            # ReLU
            # 畳み込み
            # バッチノーマライゼーション
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
        )
    def forward(self, x):
        out = self.Block(x)
        return x + out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #初めの部分
        #conv
        #relu
        self.conv1 = nn.Conv2d(3, 24, kernel_size=9, padding=4)
        self.relu = nn.PReLU()

        # 中間部分
        self.residualLayer = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        # 最終部分
        self.pixelShuffle = nn.Sequential(
            nn.Conv2d(24, 24*4, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.conv1(x)
        skip = self.relu(x)

        # skip connection
        x = self.residualLayer(skip)
        x = self.pixelShuffle(x + skip)
        return x


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        # vgg model load
        vgg = models.vgg19(pretrained=True)
        # 層の取り出し
        self.contentLayers = nn.Sequential(*list(vgg.features)[:31]).cuda().eval()
        for param in self.contentLayers.parameters():
            param.requires_grad = False

    def forward(self, fakeFrame, frameY):
        MSELoss = nn.MSELoss()
        # 層の出力のMSE Loss
        content_loss = MSELoss(self.contentLayers(fakeFrame), self.contentLayers(frameY))
        return content_loss


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(512, 1024, kernel_size=1),
			nn.LeakyReLU(0.2),

            Flatten(),
            nn.Linear(1024, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.net(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)