import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, canales):
        super().__init__()
        self.bloque = nn.Sequential(
            nn.Conv2d(canales, canales, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(canales),
            nn.ReLU(inplace=True),
            nn.Conv2d(canales, canales, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(canales)
        )

    def forward(self, x):
        return x + self.bloque(x)

class GeneradorResNet(nn.Module):
    def __init__(self, canales_entrada=3, canales_salida=3, num_bloques=6):
        super().__init__()
        modelo = [
            nn.Conv2d(canales_entrada, 64, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_bloques):
            modelo += [ResidualBlock(256)]

        modelo += [
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, canales_salida, 7, 1, 3),
            nn.Tanh()
        ]

        self.modelo = nn.Sequential(*modelo)

    def forward(self, x):
        return self.modelo(x)
