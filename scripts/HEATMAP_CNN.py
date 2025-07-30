import torch
import torch.nn as nn
from torchvision import models

# --- Create a Custom Model Class ---
class HeatmapResNet(nn.Module):
    def __init__(self, in_channels=60, grid_size=16):
        super().__init__()
        
        base_resnet = models.resnet18(pretrained=True)
        self.modify_first_layer(base_resnet, in_channels)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-2])
        self.pooling_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor_head = nn.Linear(
            in_features=512,
            out_features=grid_size * grid_size
        )
        self.grid_size = grid_size

    def modify_first_layer(self, model, in_channels):
        # This is the exact logic from your script
        new_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False
        )
        with torch.no_grad():
            original_weights = model.conv1.weight.clone()
            avg_weights = torch.mean(original_weights, dim=1, keepdim=True)
            new_conv1.weight = nn.Parameter(avg_weights.repeat(1, in_channels, 1, 1))
        
        model.conv1 = new_conv1

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling_layer(x)
        x = torch.flatten(x, 1)
        x = self.regressor_head(x)
        output_grid = x.view(-1, self.grid_size, self.grid_size)

        return output_grid

# --- How to use the new model ---
IN_CHANNELS = 60
GRID_SIZE = 16

# Instantiate our custom model
model = HeatmapResNet(in_channels=IN_CHANNELS, grid_size=GRID_SIZE)

# You can print it to see its structure
# print(model)

# Test it with a dummy input
dummy_input = torch.randn(4, IN_CHANNELS, 640, 640) # batch_size=4
output = model(dummy_input)

print("Successfully processed the input!")
print("Final output shape:", output.shape) # Should be torch.Size([4, 16, 16])


# # Load the pre-trained ResNet-18 model
# resnet = models.resnet18(pretrained=True)

# # Define your desired number of input channels
# IN_CHANNELS = 60 


# new_conv1 = nn.Conv2d(
#     in_channels=IN_CHANNELS,
#     out_channels=resnet.conv1.out_channels,
#     kernel_size=resnet.conv1.kernel_size,
#     stride=resnet.conv1.stride,
#     padding=resnet.conv1.padding,
#     bias=False
# )

# with torch.no_grad():
#     original_weights = resnet.conv1.weight.clone()
#     avg_weights = torch.mean(original_weights, dim=1, keepdim=True) # Shape: [64, 1, 7, 7]

#     new_conv1.weight = nn.Parameter(avg_weights.repeat(1, IN_CHANNELS, 1, 1)) # Shape: [64, 60, 7, 7]

# resnet.conv1 = new_conv1

# new_regressor_head = nn.Linear(
#     in_features=512,
#     out_features=16*16
# )

# resnet.regressor_head = new_regressor_head