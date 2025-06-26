import os
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
from nnunetv2.utilities.pca_utils import extract_patches_and_origins, reassemble_patches



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
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, scan_type='x'):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand)
        self.local_pca_vectors = None
        self.local_pca_coords = None
        self.scan_type = scan_type
        self.register_buffer('principal_vector', torch.zeros(3), persistent=True)

    def set_local_pca_vectors(self, vectors, coords):
        # Ensure both are torch tensors
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        self.local_pca_vectors = vectors
        self.local_pca_coords = coords.long()

    def get_patch_index(self, patch_origin):
        print("Patch origin:", patch_origin)
        arr = torch.tensor(patch_origin, dtype=torch.long)
        if self.local_pca_coords is None:
            print("[MambaLayer] No local PCA coordinates set. Returning None.")
            return None
        # Ensure coords are also long/int
        coords = self.local_pca_coords
        if coords.dtype != torch.long:
            coords = coords.long()
        matches = torch.all(coords == arr, dim=1)
        idx = torch.where(matches)[0]
        return idx[0].item() if len(idx) > 0 else None

    def get_local_pca_vector(self, patch_origin):
        idx = self.get_patch_index(patch_origin)
        print("Index of patch origin:", idx)
        if idx is not None:
            return self.local_pca_vectors[idx]
        else:
            return self.principal_vector

    def set_scan_vector(self, vector):
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector) if isinstance(vector, np.ndarray) else vector
        self.principal_vector.copy_(vector.to(self.principal_vector.device, dtype=self.principal_vector.dtype))

    def flatten_for_scan(self, x, scan_type='pca', patch_origin=None):
        B, C, D, H, W = x.shape
        if scan_type == 'pca':
            #print("[MambaLayer] Using PCA scan type.")
            #print("TRYING TO GET LOCAL PCA, failing afterwards ")
            vector = self.get_local_pca_vector(patch_origin) if patch_origin else self.principal_vector
            #print("made it")
            return self.flatten_pca_scan(x, vector)

        permute, unpermute = {
            'x':  ((0, 1, 4, 3, 2), lambda t: t.reshape(B, C, W, H, D).permute(0, 1, 4, 3, 2)),
            'y':  ((0, 1, 3, 4, 2), lambda t: t.reshape(B, C, H, W, D).permute(0, 1, 4, 2, 3)),
            'z':  (None, lambda t: t.transpose(-1, -2).reshape(B, C, D, H, W)),
        }[scan_type]
        
        x_perm = x.permute(*permute) if permute else x
        x_flat = x_perm.reshape(B, C, -1).transpose(-1, -2)
        return x_flat, unpermute

    def flatten_pca_scan(self, x, principal_vector):
        # ! TODO: INCLUDE LOGIC HERE TO FALL BACK TO x!! 
        """
        Flattens the input tensor x using the given principal_vector.
        Input:
            x: (B, C, D, H, W) tensor
            principal_vector: (3,) numpy or torch array (should be unit vector)
        Output:
            x_flat: (B, N, C) tensor, where N = D*H*W
            indices: list of sorted indices for each batch
        """
        if principal_vector is None:
            print("[MambaLayer] Principal vector is None. Skipping PCA flatten.")
            return None, None
        
        B, C, D, H, W = x.shape
        device = x.device
        zz, yy, xx = torch.meshgrid(
            torch.arange(D, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        coords = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3).float()
        # Project coordinates onto principal vector
        if isinstance(principal_vector, np.ndarray):
            principal_vector = torch.from_numpy(principal_vector).to(device, dtype=torch.float32)
        else: 
            principal_vector = principal_vector.to(device, dtype=torch.float32)
        # TODO: Forse qua niente matmul, ma dot product??? Gets messed up here? 
        projections = torch.matmul(coords, principal_vector)
        sorted_idx = torch.argsort(projections)

        x_flat_list = [x[b].reshape(C, -1)[:, sorted_idx].T for b in range(B)]
        x_flat = torch.stack(x_flat_list)

        def unpermute(t):
            out = torch.zeros((B, D*H*W, C), device=t.device,  dtype=t.dtype)
            for b in range(B):
                out[b, sorted_idx] = t[b]
            return out.permute(0, 2, 1).reshape(B, C, D, H, W)

        return x_flat, unpermute

    def forward(self, x):
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        B, C = x.shape[:2]
        assert C == self.dim
        
        if self.scan_type == 'pca':
            if not torch.any(self.principal_vector):
                print("[MambaLayer] No principal vector. Falling back to 'x'.")
                x_flat, unpermute = self.flatten_for_scan(x, 'x')
                #print("x_flat shape:", x_flat.shape)
                
            else:
                x_flat, unpermute = self.flatten_pca_scan(x, self.principal_vector)
        else:
            x_flat, unpermute = self.flatten_for_scan(x, self.scan_type)

        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *x.shape[2:])
        return unpermute(out)

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
    
class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
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
                 mamba_scan_type: str = 'pca'  # 'x', 'y', 'z', 'yz-diag', 'xy-diag', 'pca' # TODO 
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

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        #feature_map_size = input_size[::-1] # ! CHANGE 
        feature_map_size = input_size 
        #print("feature_map_size Input size", feature_map_size)
        
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True
        #print("do_channel_token", do_channel_token)
        #print("feature_map_sizeS size", feature_map_sizes)
        

        #print(f"feature_map_sizes: {feature_map_sizes}")
        #print(f"do_channel_token: {do_channel_token}")
        
        self.mamba_scan_type = mamba_scan_type
        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ),
            MambaLayer(
                dim=stem_channels,
                scan_type=self.mamba_scan_type
                # Not channel_token: scan path will be used
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels

        stages = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )
            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        #self.dropout_op = dropout_op
        #self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    
    

    def forward(self, x):
        if self.stem is not None:
            conv = self.stem[0]
            mamba = self.stem[1]
            rest = self.stem[2:]

            x = conv(x)

            def pad_to_divisible_torch(x, patch_size):
                # x: (B, C, D, H, W)
                _, _, D, H, W = x.shape
                pd, ph, pw = patch_size
                pad_d = (pd - D % pd) if D % pd != 0 else 0
                pad_h = (ph - H % ph) if H % ph != 0 else 0
                pad_w = (pw - W % pw) if W % pw != 0 else 0
                # Pad in (left, right) order for each dim, so pad only at the end
                padding = (0, pad_w, 0, pad_h, 0, pad_d)  # (W_left, W_right, H_left, H_right, D_left, D_right)
                if any([pad_d, pad_h, pad_w]):
                    x = torch.nn.functional.pad(x, padding)
                return x

            patch_size = (160, 128, 112) # TODO dangerous hardcoded value, should be passed as argument
            x = pad_to_divisible_torch(x, patch_size)
            # divide the volume in mini sub volumes and where each sub volume is a patch
            patches, coords = extract_patches_and_origins(x, patch_size)
            #print("[ResidualMambaEncoder] Extracted {} patches from input volume.".format(len(patches)))
            #print("[ResidualMambaEncoder] Patches[0], coords[0]:", patches[0], coords[0])
            
            mamba_outputs = []

            for i, (patch, origin) in enumerate(zip(patches, coords)):
                if mamba.local_pca_vectors is not None:
                    principal_vector = mamba.local_pca_vectors[i]
                    principal_vector = mamba.local_pca_vectors[i]       
                    #print("[ResidualMambaEncoder] Using principal vector for patch origin {}: {}".format(origin, principal_vector))
                    #! TODO: here incorporate that output can be none and directly go to scan x. 
                    
                    x_flat, unpermute = mamba.flatten_pca_scan(patch, principal_vector)
                
                else:
                    print("[ResidualMambaEncoder] Falling back to 'x' scan path.")
                    x_flat, unpermute = mamba.flatten_for_scan(patch, scan_type='x')
                x_norm = mamba.norm(x_flat)
                x_mamba = mamba.mamba(x_norm)
                patch_out = unpermute(x_mamba)
                mamba_outputs.append(patch_out)

            x = reassemble_patches(mamba_outputs, coords, x.shape)

            for block in rest:
                x = block(x)

        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            ret.append(x)

        return ret if self.return_skips else ret[-1]


    def compute_conv_feature_map_size(self, input_size):
        #print("Input size encoder initial", input_size)
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
            #print("Input size encoder", input_size)

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
            #print("encoder.output_channels[-s]", encoder.output_channels[-s])
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
    
class UMambaEnc(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
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
                 stem_channels: int = None,
                 mamba_scan_type: str = 'pca', #TODO
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
        self.encoder = ResidualMambaEncoder(
            input_size,
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
        self.mamba_scan_type = mamba_scan_type
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_first_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True, 
        output_folder=None
    ):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaEnc'
    network_class = UMambaEnc
    kwargs = {
        'UMambaEnc': {
            'input_size': configuration_manager.patch_size,
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
    # Initialize the model
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
    
    if output_folder is not None:
        vec_path = os.path.join(output_folder, "local_pca_vectors.npy")
        coord_path = os.path.join(output_folder, "local_pca_coords.npy")

        if os.path.isfile(vec_path) and os.path.isfile(coord_path):
            vectors = np.load(vec_path)
            coords = np.load(coord_path)
            print(f"[Network] Loaded {len(vectors)} local PCA vectors and {len(coords)} coords.")

            # Ensure correct types
            vectors = torch.from_numpy(vectors).float()
            coords = torch.from_numpy(coords).long()

            for module in model.modules():
                if hasattr(module, "set_local_pca_vectors"):
                    module.set_local_pca_vectors(vectors, coords)
                    print(f"[Network] Set local PCA vectors for module: {module.__class__.__name__}")
        else:
            print("[Network] No PCA vector/coord files found. Skipping injection.")


    return model
