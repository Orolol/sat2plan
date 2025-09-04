import torch
import torch.nn as nn
import math
# Removed old block imports: from sat2plan.logic.blocks.blocks import CNN_Block, UVCCNNlock, PixelwiseViT, DownsamplingBlock, UpsamplingBlock

# Fixup Initialization helper
def _init_weights(m, num_blocks_in_stage):
    if isinstance(m, R3GANResidualBlock):
        # Initialize the block itself (handled within the block's init)
        pass
    elif isinstance(m, nn.Conv2d):
        # Standard Kaiming init for non-residual block conv layers
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Standard init for linear layers
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class R3GANResidualBlock(nn.Module):
    """
    R3GAN Residual Block based on Config E:
    Inverted bottleneck (1x1 -> GroupedConv 3x3 -> 1x1)
    LeakyReLU activations
    Fixup Initialization (with biases added before layers)
    """
    def __init__(self, channels, expansion_ratio=4, groups=16, num_blocks_in_stage=2):
        super().__init__()
        self.num_blocks = num_blocks_in_stage # Needed for fixup scaling
        hidden_dim = int(channels * expansion_ratio)

        # Define layers individually for Fixup bias application
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_grouped = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=groups, bias=False)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False) # Projection

        # Add bias terms for fixup scaling (one per conv/act layer)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.bias3 = nn.Parameter(torch.zeros(1))
        self.bias4 = nn.Parameter(torch.zeros(1))
        # Bias for the final projection conv is added implicitly via the residual connection

        # Apply Fixup Initialization
        self._init_fixup()

    def _init_fixup(self):
        # Zero-initialize the last conv layer in the block
        nn.init.constant_(self.conv2.weight, 0)

        # Scale down other conv layers
        scale = self.num_blocks ** (-0.25) # As per paper Config D description
        # Init first conv
        nn.init.kaiming_normal_(self.conv1.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        self.conv1.weight.data.mul_(scale)
        # Init grouped conv
        nn.init.kaiming_normal_(self.conv_grouped.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        self.conv_grouped.weight.data.mul_(scale)

    def forward(self, x):
        identity = x
        # Apply fixup biases before layers
        out = self.conv1(x + self.bias1)
        out = self.act1(out + self.bias2)
        out = self.conv_grouped(out + self.bias3)
        out = self.act2(out + self.bias4)
        out = self.conv2(out) # No bias added before final conv, added with residual
        # Residual connection
        return identity + out


####################################################################################################################
################################################## DISCRIMINATEUR (R3GAN) ##########################################
####################################################################################################################

class Discriminator(nn.Module):
    """
    R3GAN Discriminator based on Config E:
    - Fully symmetric ResNet-style architecture.
    - Uses R3GANResidualBlock with inverted bottleneck and grouped conv.
    - Bilinear downsampling (via AvgPool2d).
    - No normalization layers.
    - Fixup initialization.
    - LeakyReLU activations.
    - Takes concatenated (condition, image) as input.
    """
    def __init__(self, in_channels=3, img_size=256, base_features=96, num_blocks_per_stage=2, groups=16, expansion_ratio=4):
        super().__init__()
        self.img_size = img_size
        # Feature map sizes based on FFHQ-256 config in paper (Table 9)
        # Example: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        # Channels:   96 -> 192 -> 384 -> 768 -> 768 -> 768 -> 768 (Length 7)
        features_list = [base_features * mult for mult in [1, 2, 4, 8, 8, 8, 8]]
        # Corrected num_stages calculation to end at 4x4
        num_stages = int(math.log2(img_size)) - 2 # e.g., log2(256)-2 = 6 stages for 256->4

        # Initial 1x1 convolution
        # Input is concatenated condition (e.g., satellite) and real/fake image (e.g., map)
        self.initial = nn.Sequential(
             nn.Conv2d(in_channels * 2, features_list[0], kernel_size=1, stride=1, padding=0, bias=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

        self.layers = nn.ModuleList()
        current_channels = features_list[0]

        # Downsampling stages (should run num_stages times)
        for i in range(num_stages):
            # Indexing features_list needs care: use i+1 for output channels
            out_channels = features_list[min(i + 1, len(features_list) - 1)]
            # Transition Layer (Resampling + optional 1x1 conv)
            transition = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2), # Bilinear downsampling approximation
                nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=False) if current_channels != out_channels else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
            # Residual Blocks for this stage
            res_blocks = nn.Sequential(
                *[R3GANResidualBlock(out_channels, expansion_ratio=expansion_ratio, groups=groups, num_blocks_in_stage=num_blocks_per_stage) for _ in range(num_blocks_per_stage)]
            )
            self.layers.append(nn.Sequential(transition, res_blocks))
            current_channels = out_channels # Update current_channels for the next stage

        # Final Classifier Head (operates on 4x4 feature map)
        # Paper: global 4x4 depthwise conv -> flatten -> linear
        self.final = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=4, groups=current_channels, bias=False), # Depthwise 4x4
            nn.Flatten(),
            nn.Linear(current_channels, 1) # Output single logit
        )

        # Initialize weights (Fixup is handled in blocks, init others)
        self.apply(lambda m: _init_weights(m, num_blocks_per_stage))


    def forward(self, x, y):
        # x: condition (e.g., satellite image), y: real/fake image (e.g., map)
        out = torch.cat([x, y], dim=1)
        # print(f"Shape after cat in Discriminator: {out.shape}") # DEBUG print shape removed
        out = self.initial(out)
        for stage in self.layers:
            out = stage(out)
        out = self.final(out)
        return out


####################################################################################################################
################################################### GENERATEUR (R3GAN) #############################################
####################################################################################################################

class Generator(nn.Module):
    """
    R3GAN Generator based on Config E:
    - Fully symmetric ResNet-style architecture (mirrors Discriminator).
    - Uses R3GANResidualBlock with inverted bottleneck and grouped conv.
    - Bilinear upsampling.
    - No normalization layers.
    - Fixup initialization.
    - LeakyReLU activations.
    - Takes condition image as input.
    - Tanh output activation.
    """
    # Renamed __init__ parameter 'out_channels' to 'final_out_channels' to avoid loop conflict
    def __init__(self, in_channels=3, final_out_channels=3, img_size=256, base_features=96, num_blocks_per_stage=2, groups=16, expansion_ratio=4):
        super().__init__()
        self.img_size = img_size
        # Feature map sizes based on FFHQ-256 config in paper (Table 9) - Reversed for Generator
        # Example: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        # Channels: 768 -> 768 -> 768 -> 768 -> 384 -> 192 -> 96 (Length 7)
        features_list = [base_features * mult for mult in [8, 8, 8, 8, 4, 2, 1]] # Reversed order
        # Corrected num_stages calculation to start from 4x4
        num_stages = int(math.log2(img_size)) - 2 # e.g., log2(256)-2 = 6 stages for 4->256

        # Initial Layer (Processes input condition image down to the smallest feature map size, 4x4)
        self.initial_down = nn.ModuleList()
        current_channels = in_channels
        # Downsample from img_size to 4x4 using num_stages steps
        for i in range(num_stages):
             # Determine output channels for this downsampling step
             # We want to reach features_list[0] (e.g., 768) at the end
             # A simple approach: gradually increase channels, then hit max
             temp_out_channels = base_features * (2**i) if i < 3 else features_list[0] # Example logic, might need tuning
             temp_out_channels = min(temp_out_channels, features_list[0]) # Cap at max features

             self.initial_down.append(nn.Sequential(
                 nn.Conv2d(current_channels, temp_out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.LeakyReLU(0.2, inplace=True)
             ))
             current_channels = temp_out_channels

        # Ensure final output channels match the first stage of the upsampler (features_list[0])
        if current_channels != features_list[0]:
             self.initial_down.append(nn.Sequential(
                  nn.Conv2d(current_channels, features_list[0], kernel_size=1, bias=False), # 1x1 conv to match channels
                  nn.LeakyReLU(0.2, inplace=True)
             ))
        current_channels = features_list[0] # Should be 768 for 256x256 example

        # Core Upsampling Stages (should run num_stages times)
        self.layers = nn.ModuleList()
        # Renamed loop variable 'out_channels' to 'stage_out_channels'
        for i in range(num_stages):
            # Indexing features_list needs care: use i+1 for output channels
            stage_out_channels = features_list[min(i + 1, len(features_list) - 1)]
            # Transition Layer (Upsampling + optional 1x1 conv)
            transition = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, stage_out_channels, kernel_size=1, bias=False) if current_channels != stage_out_channels else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
            # Residual Blocks for this stage
            res_blocks = nn.Sequential(
                *[R3GANResidualBlock(stage_out_channels, expansion_ratio=expansion_ratio, groups=groups, num_blocks_in_stage=num_blocks_per_stage) for _ in range(num_blocks_per_stage)]
            )
            self.layers.append(nn.Sequential(transition, res_blocks))
            current_channels = stage_out_channels # Update current_channels for the next stage

        # Final Layer (1x1 conv + Tanh)
        # Use the 'out_channels' parameter from __init__, not the loop variable
        self.final = nn.Sequential(
            # Use the 'out_channels' parameter passed to __init__ (e.g., 3)
            # Ensure this uses the init parameter 'out_channels' (should be 3)
            # Use the 'final_out_channels' (renamed from 'out_channels' in __init__)
            nn.Conv2d(current_channels, final_out_channels, kernel_size=1, bias=False),
            nn.Tanh()
        )

        # Initialize weights (Fixup is handled in blocks, init others)
        self.apply(lambda m: _init_weights(m, num_blocks_per_stage))

    def forward(self, x):
        # x: condition image (e.g., satellite)
        # print(f"Generator input shape: {x.shape}") # DEBUG removed
        out = x
        # Initial downsampling of condition
        for down_block in self.initial_down:
             out = down_block(out)
             # print(f"Generator shape after down_block: {out.shape}") # DEBUG removed

        # Upsampling stages
        for stage in self.layers:
            out = stage(out)
            # print(f"Generator shape after upsampling stage: {out.shape}") # DEBUG removed

        # Final output layer
        # print(f"Generator shape BEFORE final layer: {out.shape}") # DEBUG removed
        out = self.final(out)
        # print(f"Generator shape AFTER final layer: {out.shape}") # DEBUG removed
        return out
