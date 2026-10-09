import torch
import torch.nn as nn
from config import NUM_CLASSES, BATCH_SIZE

class UrbanSoundCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(UrbanSoundCNN, self).__init__()

        self.conv1 = self._make_conv_block(3, 32, pool_kernel=(2, 2), pool_stride=(2, 2))
        self.conv2 = self._make_conv_block(32, 64, pool_kernel=(1, 2), pool_stride=(1, 2))
        self.conv3 = self._make_conv_block(64, 128, pool_kernel=(1, 2), pool_stride=(1, 2))
        self.conv4 = self._make_conv_block(128, 256, pool_kernel=(1, 2), pool_stride=(1, 2))

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels, pool_kernel, pool_stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        out = self.classifier(x)
        return out
  
# if __name__ == "__main__":
#     dummy_input = torch.randn(BATCH_SIZE, 3, 128, 173)
#     model = UrbanSoundCNN(num_classes=10)
#     output = model(dummy_input)
#     print(f"Kích thước đầu vào: {dummy_input.shape}")
#     print(f"Kích thước đầu ra:  {output.shape}")
