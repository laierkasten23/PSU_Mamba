import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mamba_ssm.modules.mamba_simple import Mamba

from sklearn.decomposition import PCA


def flatten_for_scan(x, scan_type='x'):
    """
    Input: x: (B, C, D, H, W) tensor
    Output: Flattened tensor and unpermute function
    scan_type: 'x', 'y', 'z', 'yz-diag', 'xy-diag'
    - 'x': Flatten along the x-axis
    - 'y': Flatten along the y-axis
    - 'z': Flatten along the z-axis
    - 'yz-diag': Flatten along the diagonal of the yz-plane
    - 'xy-diag': Flatten along the diagonal of the xy-plane
    """
    B, C, D, H, W = x.shape

    if scan_type == 'x':
        print("Using x-scan")
        x_perm = x.permute(0, 1, 4, 3, 2)  # (B, C, X, Y, Z)
        flatten = x_perm.reshape(B, C, -1).transpose(-1, -2)
        unpermute = lambda t: t.reshape(B, C, W, H, D).permute(0, 1, 4, 3, 2)
        return flatten, unpermute

    elif scan_type == 'y':
        print("Using y-scan")
        x_perm = x.permute(0, 1, 3, 4, 2)  # (B, C, Y, X, Z)
        flatten = x_perm.reshape(B, C, -1).transpose(-1, -2)
        unpermute = lambda t: t.reshape(B, C, H, W, D).permute(0, 1, 4, 2, 3)
        return flatten, unpermute

    elif scan_type == 'z':
        print("Using z-scan")
        # No permutation
        flatten = x.reshape(B, C, -1).transpose(-1, -2)
        unpermute = lambda t: t.transpose(-1, -2).reshape(B, C, D, H, W)
        return flatten, unpermute

    elif scan_type == 'yz-diag':
        print("Using yz-diag-scan")
        x_perm = x.permute(0, 1, 4, 3, 2)  # (B, C, X, Y, Z)
        X, Y, Z = x_perm.shape[2:]
        z_coords, y_coords = torch.meshgrid(
            torch.arange(Z, device=x.device),
            torch.arange(Y, device=x.device),
            indexing='ij'
        )
        #diag_order = torch.argsort(z_coords + y_coords, dim=None) #TODO
        diag_order = torch.argsort((z_coords + y_coords).flatten()) #! changed
        x_flat = x_perm.reshape(B, C, X, Y * Z)[:, :, :, diag_order]
        flatten = x_flat.reshape(B, C, -1).transpose(-1, -2)
        unpermute = lambda t: t.reshape(B, C, X, Y, Z).permute(0, 1, 4, 3, 2)
        return flatten, unpermute

    elif scan_type == 'xy-diag':
        x_perm = x.permute(0, 1, 2, 4, 3)  # (B, C, Z, X, Y)
        D, X, Y = x_perm.shape[2:]
        x_coords, y_coords = torch.meshgrid(
            torch.arange(X, device=x.device),
            torch.arange(Y, device=x.device),
            indexing='ij'
        )
        #diag_order = torch.argsort(x_coords + y_coords, dim=None) #TODO check
        diag_order = torch.argsort((x_coords + y_coords).flatten()) #! changed
        x_flat = x_perm.reshape(B, C, D, X * Y)[:, :, :, diag_order]
        flatten = x_flat.reshape(B, C, -1).transpose(-1, -2)
        unpermute = lambda t: t.reshape(B, C, D, X, Y).permute(0, 1, 2, 4, 3)
        return flatten, unpermute

    else:
        raise ValueError(f"Unsupported scan type: {scan_type}")

    


def flatten_pca_scan(x, mask):
    """
    Function to flatten the input tensor x using PCA scan.
    Input: x: (B, C, D, H, W) tensor
           mask: (B, 1, D, H, W) binary mask tensor
    Output: Flattened tensor and indices
    
    """
    B, C, D, H, W = x.shape
    x_flat_list = []
    index_list = []

    for b in range(B):
        coords = torch.nonzero(mask[b, 0], as_tuple=False).float().cpu().numpy()

        if coords.shape[0] < 3:
            raise ValueError("Too few non-zero voxels for PCA.")

        pca = PCA(n_components=1)
        scores = pca.fit_transform(coords).squeeze()
        sorted_indices = torch.tensor(scores).argsort()
        sorted_coords = coords[sorted_indices]
        flat_idx = [int(D * H * x[0] + H * x[1] + x[2]) for x in sorted_coords]

        x_b = x[b].reshape(C, -1)[:, flat_idx].T
        x_flat_list.append(x_b)
        index_list.append(flat_idx)

    x_flat = torch.stack(x_flat_list, dim=0)  # (B, N, C)
    return x_flat, index_list


class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class MambaLayer(nn.Module):
    # TODO 
    # - check patch size to avoid segmenting ears
    # - scan paths
    # - put mamba also in encoding layers
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, scan_type='y'):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.scan_type = scan_type # self.scan_type = 'x'  # 'x', 'y', 'z', 'xy-diag', 'yz-diag', 'xz-diag'
    
    ## NEW VERSION FROM HERE: 

    @autocast(enabled=False)
    def forward(self, x, mask=None):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel() # total number of voxels in the 3D image (i.e., Z × Y × X)
        img_dims = x.shape[2:]
        

        if self.scan_type == 'pca':
            if mask is None:
                raise ValueError("PCA scan requires a segmentation mask.")
            x_flat, indices = flatten_pca_scan(x, mask)
            x_norm = self.norm(x_flat)
            x_mamba = self.mamba(x_norm)

            out = torch.zeros((B, C, img_dims[0] * img_dims[1] * img_dims[2]), device=x.device)
            for b in range(B):
                out[b, :, indices[b]] = x_mamba[b].T
            return out.reshape(B, C, *img_dims)

        else:
            x_flat, unpermute = flatten_for_scan(x, self.scan_type)
            x_norm = self.norm(x_flat)
            x_mamba = self.mamba(x_norm)
            out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
            return unpermute(out)

        
        x_flat = flatten_for_scan(x, self.scan_type)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out  # Already in original shape; no need to permute back if you unify shape
    
    ## NEW VERSION UNTIL #TODO delete what comes after
    
    
    ''' # ! Original Version
    @autocast(enabled=False)
    def forward(self, x):
        xy_scan = True 
        x_scan = False
        y_scan = False
        z_scan = False
        
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel() # total number of voxels in the 3D image (i.e., Z × Y × X)
        img_dims = x.shape[2:]
        
        if x_scan:
            x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)           
            x_norm = self.norm(x_flat)
            x_mamba = self.mamba(x_norm)
            out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
            return out
    
        if y_scan:
            x = x.permute(0, 1, 2, -1, -2) # ! permute to y direction from (z, y, x) to (z, x, y)
            x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
            x_norm = self.norm(x_flat)
            x_mamba = self.mamba(x_norm)
            out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
            out = out.permute(0, 1, 2, -1, -2)
            return out
        
        if z_scan:
            x = x.permute(0, 1, 4, 3, 2) # ! permute to z direction from (B, C, z, y, x) to (B, C, x, y, z)
            x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
            x_norm = self.norm(x_flat)
            x_mamba = self.mamba(x_norm)
            out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
            out = out.permute(0, 1, 4, 3, 2)
            return out
        
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        #print('out.shape:', out.shape)
        return out
    
    
    # ! Version for diagonal scan path in yz direction
    @autocast(enabled=False)
    def forward(self, x):
        print("HEEEEEEEEEEEEEELLLLLLLLLLLOOOOOOOOOOO")
        print("yz diagonal scan path")
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        # n_tokens = x.shape[2:].numel() # as no 1D flattening anymore. 
        img_dims = x.shape[2:]
        # ** Step 1: Permute to prioritize X over (Y, Z) **
        x = x.permute(0, 1, 4, 3, 2)  # Now (B, C, X, Y, Z)

        # ** Step 2: Apply diagonal scan order in (Y, Z) **
        X, Y, Z = img_dims[2], img_dims[1], img_dims[0]  # Extract dimensions

        # Create sorting indices for diagonal traversal in (Y, Z)
        z_coords, y_coords = torch.meshgrid(
            torch.arange(Z, device=x.device), torch.arange(Y, device=x.device), indexing='ij'
        )
        diag_order = torch.argsort(z_coords + y_coords, dim=None)  # Sort by diagonal sum

        # Reshape and apply diagonal ordering within each X slice
        x_flat = x.reshape(B, C, X, Y * Z)  # Flatten (Y, Z)
        x_flat = x_flat[:, :, :, diag_order]  # Apply diagonal scan order

        # ** Step 3: Flatten and process with Mamba **
        x_flat = x_flat.reshape(B, C, X * Y * Z).transpose(-1, -2)  # Final flattening
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        # ** Step 4: Restore original image dimensions and ordering **
        out = x_mamba.transpose(-1, -2).reshape(B, C, X, Y, Z)  # Reshape back
        out = out.permute(0, 1, 4, 3, 2)  # Convert back to (B, C, Z, Y, X)

        return out
'''

class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()

        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    
class UNetResEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]

        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op = conv_op,
                input_channels = input_channels,
                output_channels = stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ), 
            *[
                BasicBlockD(
                    conv_op = conv_op,
                    input_channels = stem_channels,
                    output_channels = stem_channels,
                    kernel_size = kernel_sizes[0],
                    stride = 1,
                    conv_bias = conv_bias,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )


        input_channels = stem_channels

        stages = []
        for s in range(n_stages):

            stage = nn.Sequential(
                BasicResBlock(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )


            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs

        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        #print("UNetResEncoder x.shape:", x.shape)
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            #print("UNetResEncoder s(x).shape for stage s", s , s(x).shape)
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):

        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder


        stages = []
        upsample_layers = []

        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]


            upsample_layers.append(UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = 2 * input_features_skip,
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op = encoder.conv_op,
                        input_channels = input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        stride = 1,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)

                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    
class UMambaBot(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1    

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = UNetResEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.mamba_layer = MambaLayer(dim = features_per_stage[-1])

        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        #print("IN U-Mamba Bot now")
        # ([2, 1, 160, 128, 112]))
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_bot_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
    ):
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaBot'
    network_class = UMambaBot
    kwargs = {
        'UMambaBot': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model
