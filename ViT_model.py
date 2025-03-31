import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights

class linear_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(linear_layer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return x

class ViTModel(nn.Module):
    def __init__(self, model_name='vit_b_16', num_classes=15, pretrained=True):
        super(ViTModel, self).__init__()
        # Load the specified ViT model
        if model_name == 'vit_b_16':
            self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        elif model_name == 'vit_b_32':
            self.model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT if pretrained else None)
        elif model_name == 'vit_l_16':
            self.model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError("model_name must be one of ['vit_b_16', 'vit_b_32', 'vit_l_16']")
        
        # Modify the input layer to accept grayscale (1-channel) images
        self.model.conv_proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.model.conv_proj.out_channels,
            kernel_size=self.model.conv_proj.kernel_size,
            stride=self.model.conv_proj.stride,
            padding=self.model.conv_proj.padding,
            bias=self.model.conv_proj.bias is not None
        )

        # Modify the classifier to match the number of classes
        in_features = self.model.heads.head.in_features
        # self.model.heads.head = linear_layer(in_features, num_classes)

        # Sigmoid activation for multi-label classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x