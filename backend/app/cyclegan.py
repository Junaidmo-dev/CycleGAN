import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """ResNet residual block with reflection padding"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet-based Generator for CycleGAN
    
    Architecture:
    - Encoder: 3 conv layers with downsampling
    - Transformer: 9 ResNet blocks
    - Decoder: 3 deconv layers with upsampling
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        """
        Args:
            input_nc: Number of input channels (3 for RGB)
            output_nc: Number of output channels (3 for RGB)
            ngf: Number of filters in the first conv layer
            n_blocks: Number of ResNet blocks (6 or 9)
        """
        super(ResnetGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                 kernel_size=3, stride=2, padding=1,
                                 output_padding=1, bias=False),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
