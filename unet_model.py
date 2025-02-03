import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()

        # Encoder (downsampling path)
        self.enc_conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.enc_conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout after first encoder block

        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout after second encoder block

        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p=0.5)  # Dropout after third encoder block

        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p=0.5)  # Dropout after fourth encoder block

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bottleneck_dropout = nn.Dropout(p=0.5)  # Dropout in bottleneck

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout_dec4 = nn.Dropout(p=0.5)  # Dropout after fourth decoder block

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout_dec3 = nn.Dropout(p=0.5)  # Dropout after third decoder block

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout_dec2 = nn.Dropout(p=0.5)  # Dropout after second decoder block

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dropout_dec1 = nn.Dropout(p=0.5)  # Dropout after first decoder block

        # Output layer
        self.output_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = F.relu(self.enc_conv1(x))
        c1 = F.relu(self.enc_conv1_2(c1))
        c1 = self.dropout1(c1)
        p1 = self.pool1(c1)

        c2 = F.relu(self.enc_conv2(p1))
        c2 = F.relu(self.enc_conv2_2(c2))
        c2 = self.dropout2(c2)
        p2 = self.pool2(c2)

        c3 = F.relu(self.enc_conv3(p2))
        c3 = F.relu(self.enc_conv3_2(c3))
        c3 = self.dropout3(c3)
        p3 = self.pool3(c3)

        c4 = F.relu(self.enc_conv4(p3))
        c4 = F.relu(self.enc_conv4_2(c4))
        c4 = self.dropout4(c4)
        p4 = self.pool4(c4)

        # Bottleneck
        b = F.relu(self.bottleneck_conv1(p4))
        b = F.relu(self.bottleneck_conv2(b))
        b = self.bottleneck_dropout(b)

        # Decoder
        u4 = self.upconv4(b)
        u4 = torch.cat((u4, c4), dim=1)
        u4 = F.relu(self.dec_conv4(u4))
        u4 = F.relu(self.dec_conv4_2(u4))
        u4 = self.dropout_dec4(u4)

        u3 = self.upconv3(u4)
        u3 = torch.cat((u3, c3), dim=1)
        u3 = F.relu(self.dec_conv3(u3))
        u3 = F.relu(self.dec_conv3_2(u3))
        u3 = self.dropout_dec3(u3)

        u2 = self.upconv2(u3)
        u2 = torch.cat((u2, c2), dim=1)
        u2 = F.relu(self.dec_conv2(u2))
        u2 = F.relu(self.dec_conv2_2(u2))
        u2 = self.dropout_dec2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat((u1, c1), dim=1)
        u1 = F.relu(self.dec_conv1(u1))
        u1 = F.relu(self.dec_conv1_2(u1))
        u1 = self.dropout_dec1(u1)

        # Output layer
        output = self.output_conv(u1)
        return output
