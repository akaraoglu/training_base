import torch
import torch.nn as nn


class ShallowUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n=32):
        super(ShallowUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.downsample(enc1))
        enc3 = self.enc3(self.downsample(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.downsample(enc3))

        # Decoder path
        dec3 = self.upsample_and_concat(bottleneck, enc3)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upsample_and_concat(dec3, enc2)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upsample_and_concat(dec2, enc1)
        dec1 = self.dec1(dec1)

        # Final output
        return self.final_conv(dec1)

    def downsample(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)

    def upsample_and_concat(self, x, skip_connection):
        # Upsample x to match the size of skip_connection
        x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate along the channels dimension
        return torch.cat([x, skip_connection], dim=1)
