import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
# Using Batch Normalization in the DoubleConv module improves the performance of the model. (https://arxiv.org/abs/1502.03167)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        DoubleConv module consists of two consecutive convolutional layers with batch normalization and ReLU activation.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False), # bias=False because of batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the DoubleConv module.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        """
        UNET model implementation for semantic segmentation.
        
        Args:
            in_channels (int): Number of input channels. Default is 1.
            out_channels (int): Number of output channels. Default is 1.
            features (list): List of feature channels for each level of the UNET. Default is [64, 128, 256, 512].
        """
        super(UNET, self).__init__()
        self.ups = nn.ModuleList() # ModuleList is needed for registering the modules correctly
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET (Decoder)
        for feature in reversed(features): # reversed because we are going up
            self.ups.append(
                nn.ConvTranspose2d( # ConvTranspose2d is the opposite of Conv2d
                    feature*2, feature, kernel_size=2, stride=2, # feature*2 because of concatenation
                )
            )
            self.ups.append(DoubleConv(feature*2, feature)) 

        self.bottleneck = DoubleConv(features[-1], features[-1]*2) # Last of features because we are going up
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # Last layer doing 1x1 conv to get desired # of output channels for the segm. map

    def forward(self, x):
        """
        Forward pass of the UNET model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        skip_connections = [] # List to store skip connections for concatenation

        for down in self.downs:
            x = down(x) # Pass through the DoubleConv
            skip_connections.append(x) # Append the output of the DoubleConv to the skip connections for later usage
            x = self.pool(x) 

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse the skip connections for correct concatenation

        for idx in range(0, len(self.ups), 2): # 2 because we have 2 modules in each step (upsampling and DoubleConv)
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] # idx//2 because we are going up and we need to match the skip connections

            if x.shape != skip_connection.shape: # If the shapes do not match, we need to resize the skip connection
                x = TF.resize(x, size=skip_connection.shape[2:]) 

            concat_skip = torch.cat((skip_connection, x), dim=1) 
            x = self.ups[idx+1](concat_skip) # idx+1 because we need to pass through the DoubleConv

        return self.final_conv(x) # Final 1x1 conv to get the desired number of output channels for the segm. map