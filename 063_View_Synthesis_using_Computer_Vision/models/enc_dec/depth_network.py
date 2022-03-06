import torch
import torch.nn as nn


class Unet(nn.Module):
    """ Reimplementation of Unet that allows for simpler modifications.
        Input size has to be 256x256, output is also 256x256
    """

    def __init__(
            self,
            num_filters=32,
            channels_in=3,
            channels_out=1,
    ):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.conv8 = nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4, stride=2, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # input channels * 2 because of U-Net architecture
        self.dconv1 = nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=3, stride=1, padding=1)
        self.dconv2 = nn.Conv2d(num_filters * 8 * 2, num_filters * 8, kernel_size=3, stride=1, padding=1)
        self.dconv3 = nn.Conv2d(num_filters * 8 * 2, num_filters * 8, kernel_size=3, stride=1, padding=1)
        self.dconv3 = nn.Conv2d(num_filters * 8 * 2, num_filters * 8, kernel_size=3, stride=1, padding=1)
        self.dconv4 = nn.Conv2d(num_filters * 8 * 2, num_filters * 8, kernel_size=3, stride=1, padding=1)
        self.dconv5 = nn.Conv2d(num_filters * 8 * 2, num_filters * 4, kernel_size=3, stride=1, padding=1)
        self.dconv6 = nn.Conv2d(num_filters * 4 * 2, num_filters * 2, kernel_size=3, stride=1, padding=1)
        self.dconv7 = nn.Conv2d(num_filters * 2 * 2, num_filters, kernel_size=3, stride=1, padding=1)
        self.dconv8 = nn.Conv2d(num_filters * 2, channels_out, kernel_size=3, stride=1, padding=1)

        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
        self.batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
        self.batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
        self.batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
        self.batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (num_filters) x 128 x 128
        e2 = self.batch_norm2_0(self.conv2(self.leaky_relu(e1)))
        # state size is (num_filters x 2) x 64 x 64
        e3 = self.batch_norm4_0(self.conv3(self.leaky_relu(e2)))
        # state size is (num_filters x 4) x 32 x 32
        e4 = self.batch_norm8_0(self.conv4(self.leaky_relu(e3)))
        # state size is (num_filters x 8) x 16 x 16
        e5 = self.batch_norm8_1(self.conv5(self.leaky_relu(e4)))
        if input.shape[-1] > 128:
            e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
            e7 = self.batch_norm8_3(self.conv7(self.leaky_relu(e6)))
            e8 = self.conv8(self.leaky_relu(e7))
        elif input.shape[-1] > 64:
            e6 = self.batch_norm8_2(self.conv6(self.leaky_relu(e5)))
            e7 = self.conv7(self.leaky_relu(e6))
        else:
            e6 = self.conv6(self.leaky_relu(e5))

        # Decoder
        # Deconvolution layers:
        # state size is (num_filters x 8) x 1 x 1
        if input.shape[-1] > 128:
            d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e8))))
            d1 = torch.cat((d1_, e7), 1)
            d2_ = self.batch_norm8_5(self.dconv2(self.up(self.relu(d1))))
            d2 = torch.cat((d2_, e6), 1)
            d3_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d2))))
            d3 = torch.cat((d3_, e5), 1)
            d4_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d3))))
            # state size is (num_filters x 8) x 16 x 16
            d4 = torch.cat((d4_, e4), 1)
            d5_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d4))))
            # state size is (num_filters x 4) x 32 x 32
            d5 = torch.cat((d5_, e3), 1)
            d6_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d5))))
            # state size is (num_filters x 2) x 64 x 64
            d6 = torch.cat((d6_, e2), 1)
            d7_ = self.batch_norm(self.dconv7(self.up(self.relu(d6))))
            # state size is (num_filters) x 128 x 128
            # d7_ = torch.Tensor(e1.data.new(e1.size()).normal_(0, 0.5))
            d7 = torch.cat((d7_, e1), 1)
            d8 = self.dconv8(self.up(self.relu(d7)))
            # state size is (nc) x 256 x 256
            # output = self.tanh(d8)
            # print(d8)
        elif input.shape[-1] > 64:
            d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e7))))
            d1 = torch.cat((d1_, e6), 1)
            d2_ = self.batch_norm8_6(self.dconv3(self.up(self.relu(d1))))
            d2 = torch.cat((d2_, e5), 1)
            d3_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d2))))
            d3 = torch.cat((d3_, e4), 1)
            d4_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d3))))
            d4 = torch.cat((d4_, e3), 1)
            d5_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d4))))
            d5 = torch.cat((d5_, e2), 1)
            d6_ = self.batch_norm(self.dconv7(self.up(self.relu(d5))))
            d6 = torch.cat((d6_, e1), 1)
            d7 = self.dconv8(self.up(self.relu(d6)))

            return d7

        else:
            d1_ = self.batch_norm8_4(self.dconv1(self.up(self.relu(e6))))
            d1 = torch.cat((d1_, e5), 1)
            d2_ = self.batch_norm8_7(self.dconv4(self.up(self.relu(d1))))
            d2 = torch.cat((d2_, e4), 1)
            d3_ = self.batch_norm4_1(self.dconv5(self.up(self.relu(d2))))
            d3 = torch.cat((d3_, e3), 1)
            d4_ = self.batch_norm2_1(self.dconv6(self.up(self.relu(d3))))
            d4 = torch.cat((d4_, e2), 1)
            d5_ = self.batch_norm(self.dconv7(self.up(self.relu(d4))))
            d5 = torch.cat((d5_, e1), 1)
            d6 = self.dconv8(self.up(self.relu(d5)))

            return d6

        return d8
