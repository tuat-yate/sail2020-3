from torch import nn
from torchvision import models

# generatorの1ブロックを示す
class ResidualBlock(nn.Module):
    # 畳み込み
    # バッチノーマライゼーション
    # ReLU
    # 畳み込み
    # バッチノーマライゼーション
    def __init__(self, nf=24):
        super(ResidualBlock, self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
        )
    # elementwise sum
    def forward(self, x):
        out = self.Block(x)
        return x + out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #conv
        #relu
        self.conv1 = nn.Conv2d(3, 24, kernel_size=9, padding=4)
        self.relu = nn.PReLU()

        self.residualLayer = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        self.pixelShuffle = nn.Sequential(
            nn.Conv2d(24, 24*4, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            #nn.Conv2d(24, 12, kernel_size=3, padding=5),
            nn.PixelShuffle(2),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.conv1(x)
        skip = self.relu(x)

        x = self.residualLayer(skip)
        x = self.pixelShuffle(x + skip)
        #nn.BatchNorm2d(96)
        #print(x.shape)
        return x


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.contentLayers = nn.Sequential(*list(vgg.features)[:31]).cuda().eval()
        #self.contentLayers = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.contentLayers.parameters():
            param.requires_grad = False

    def forward(self, fakeFrame, frameY):
        MSELoss = nn.MSELoss()
        content_loss = MSELoss(self.contentLayers(fakeFrame), self.contentLayers(frameY))
        return content_loss


class Discriminator(nn.Module):
    def __init__(self, size=64):
        super(Discriminator, self).__init__()
        size = int(size / 8) ** 2

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

