from torch import nn
from torch.nn import functional as F
import torch
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

class Decoder_Block(nn.Module):
    # UNet Decoder block
    def __init__(self, x_channel, y_channel, dims, stride=2,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # use transpose conv to upsample
        if stride == 1:
            self.upconv = nn.Identity()
        else:
            self.upconv = nn.ConvTranspose2d(x_channel, x_channel, stride, stride)

        # Convolutional block: conv + bn + relu + conv + bn + relu
        self.conv = nn.Sequential(
            nn.Conv2d(x_channel + y_channel, dims, 3, padding=1),
            nn.BatchNorm2d(dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims, dims, 3, padding=1),
            nn.BatchNorm2d(dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x = self.upconv(x)
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        return x
    
class UNetHead(BaseDecodeHead):


    def __init__(self, feature_strides, ratio=1, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        # build decoder blocks: use Decoder_Block
        decoder_blocks = []
        last_channels = self.in_channels[-1]
        for i in range(len(feature_strides) - 2, -1, -1):
            next_channels = int(self.in_channels[i] * ratio)
            decoder_blocks.append(
                Decoder_Block(
                    last_channels,
                    self.in_channels[i],
                    next_channels,
                    stride=feature_strides[i + 1] // feature_strides[i],
                )
            )
            last_channels = next_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)


    def forward(self, inputs):

        x = self._transform_inputs(inputs)[::-1]

        output = None

        for i in range(len(self.feature_strides) - 1):
            if i == 0:
                output = self.decoder_blocks[i](x[i], x[i + 1])
            else:
                output = self.decoder_blocks[i](output, x[i + 1])


        output = self.cls_seg(output)
        return output
    
if __name__ == "__main__":
    decoder = UNetHead(
        feature_strides=[4, 8, 16, 32], ratio=1, 
        in_channels=[256, 512, 1024, 2048], channels=128, num_classes=6,
        in_index=[0,1,2,3])
    
    x = torch.randn(1, 256, 32, 32)
    y = torch.randn(1, 512, 16, 16)
    z = torch.randn(1, 1024, 8, 8)
    w = torch.randn(1, 2048, 4, 4)
    inputs = [x, y, z, w]
    output = decoder(inputs)
    print(output.shape)