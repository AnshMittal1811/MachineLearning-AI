from typing import Optional, Union

import MinkowskiEngine as Me
import torch
import torch.nn as nn

from lib.utils import logger
from lib.data import transforms3d

from lib.modeling import utils
from lib.config import config

from .resnet3d import ResNetBlock3d
from .resnet_sparse import BasicBlock as SparseBasicBlock


class BlockContent:
    def __init__(self, data: Union[torch.Tensor, Me.SparseTensor], encoding: Optional[torch.Tensor]):
        self.data = data
        self.encoding = encoding


class UNetSparse(nn.Module):
    def __init__(self, num_output_features: int, num_features: int = 64) -> None:
        super().__init__()
        block = UNetBlockInner(num_features * 8, num_features * 8)
        block = UNetBlock(num_features * 4, num_features * 8, num_features * 16, num_features * 4, block)
        block = UNetBlock(num_features * 2, num_features * 4, num_features * 8, num_features * 2, block)
        block = UNetBlockHybridSparse(num_features, num_features * 2, num_features * 6, num_features * 2, block)
        block = UNetBlockOuterSparse(None, num_features, num_features * 2, num_output_features, block)
        self.model = block

    def forward(self, x: torch.Tensor, batch_size, frustum_mask) -> BlockContent:
        output = self.model(BlockContent(x, frustum_mask), batch_size)

        return output


class UNetBlock(nn.Module):
    def __init__(self,
                 num_input_features: Optional[int], num_inner_features: int,
                 num_outer_features: Optional[int], num_output_features: int,
                 submodule: Optional[nn.Module]) -> None:
        super().__init__()

        num_input_features = num_output_features if num_input_features is None else num_input_features
        num_outer_features = num_inner_features * 2 if num_outer_features is None else num_outer_features

        downsample = nn.Conv3d(num_input_features, num_inner_features, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder = nn.Sequential(
            ResNetBlock3d(num_input_features, num_inner_features, stride=2, downsample=downsample),
            ResNetBlock3d(num_inner_features, num_inner_features, stride=1)
        )

        self.submodule = submodule
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(num_outer_features, num_output_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(num_output_features),
            nn.ReLU(inplace=True)
        )

        self.verbose = True
        self.logger = logger

    def forward(self, x: BlockContent) -> BlockContent:
        content = x.data

        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        processed = self.submodule(BlockContent(encoded, None))
        decoded = self.decoder(processed.data) if not self.verbose else self.forward_verbose(processed.data,
                                                                                             self.decoder)

        output = torch.cat([content, decoded], dim=1)

        if self.verbose:
            # self.print_summary(content, decoded)
            self.verbose = False

        return BlockContent(output, processed.encoding)

    def print_summary(self, x: torch.Tensor, output: torch.Tensor):
        shape_before = list(x.shape)
        shape_after = list(output.shape)
        self.logger.info(
            f"{shape_before} --> {shape_after}\t[{type(self).__name__}]\tInput: {type(x).__name__}\tDecoded: {type(output).__name__}")

    def forward_verbose(self, x: torch.Tensor, modules: torch.nn.Module) -> torch.Tensor:
        for module in modules.children():
            shape_before = list(x.shape)
            x = module(x)
            shape_after = list(x.shape)
            self.logger.info(f"[{type(self).__name__} - {type(module).__name__}]\t {shape_before} --> {shape_after}")

        return x


class UNetBlockOuterSparse(UNetBlock):
    def __init__(self,
                 num_input_features: Optional[int], num_inner_features: int,
                 num_outer_features: Optional[int], num_output_features: int,
                 submodule: Optional[nn.Module]):
        super().__init__(num_input_features, num_inner_features, num_outer_features, num_output_features, submodule)

        # define encoders
        num_encoders = 1

        # define depth feature encoder
        self.num_depth_features = 2 if config.MODEL.PROJECTION.SIGN_CHANNEL else 2
        depth_downsample = nn.Sequential(
            Me.MinkowskiConvolution(self.num_depth_features, num_inner_features, kernel_size=1, stride=1, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(num_inner_features)
        )
        self.encoder_depth = nn.Sequential(
            SparseBasicBlock(self.num_depth_features, num_inner_features, dimension=3, downsample=depth_downsample)
        )

        # define image feature encoder
        num_encoders += 1
        self.num_image_features = 80
        if config.MODEL.BACKBONE.CONV_BODY == "R-50":
            self.num_image_features = 128

        feature_downsample = nn.Sequential(
            Me.MinkowskiConvolution(self.num_image_features, num_inner_features, kernel_size=1, stride=1, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(num_inner_features)
        )
        self.encoder_features = nn.Sequential(
            SparseBasicBlock(self.num_image_features, num_inner_features, dimension=3,
                             downsample=feature_downsample)
        )

        # define instance feature encoder
        num_encoders += 1
        self.num_instance_features = config.MODEL.INSTANCE2D.MAX + 1
        instance_downsample = nn.Sequential(
            Me.MinkowskiConvolution(self.num_instance_features, num_inner_features, kernel_size=1, stride=1, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(num_inner_features)
        )
        self.encoder_instances = nn.Sequential(
            SparseBasicBlock(self.num_instance_features, num_inner_features, dimension=3, downsample=instance_downsample)
        )

        num_combined_encoder_features = num_encoders * num_inner_features
        encoder_downsample = nn.Sequential(
            Me.MinkowskiConvolution(num_combined_encoder_features, num_inner_features, kernel_size=4, stride=2, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(num_inner_features)
        )
        self.encoder = nn.Sequential(
            SparseBasicBlock(num_combined_encoder_features, num_inner_features, stride=2, dimension=3, downsample=encoder_downsample),
            SparseBasicBlock(num_inner_features, num_inner_features, dimension=3)
        )

        # define proxy outputs
        self.proxy_occupancy_128_head = nn.Sequential(Me.MinkowskiLinear(num_outer_features, 1))

        num_proxy_input_features = num_outer_features + num_inner_features
        self.proxy_instance_128_head = nn.Sequential(
            SparseBasicBlock(num_proxy_input_features, num_proxy_input_features, dimension=3),
            Me.MinkowskiConvolution(num_proxy_input_features, self.num_instance_features, kernel_size=3, stride=1, bias=True, dimension=3)
        )

        self.num_semantic_features = config.MODEL.FRUSTUM3D.NUM_CLASSES
        self.proxy_semantic_128_head = nn.Sequential(
            SparseBasicBlock(num_proxy_input_features, num_proxy_input_features, dimension=3),
            Me.MinkowskiConvolution(num_proxy_input_features, self.num_semantic_features, kernel_size=3, stride=1, bias=True, dimension=3)
        )

        # define decoder
        num_conv_input_features = num_outer_features + num_inner_features
        num_conv_input_features += self.num_instance_features
        num_conv_input_features += self.num_semantic_features

        self.decoder = nn.Sequential(
            Me.MinkowskiConvolutionTranspose(num_conv_input_features, num_output_features, kernel_size=4, stride=2, bias=False, dimension=3, expand_coordinates=True),
            Me.MinkowskiInstanceNorm(num_output_features),
            Me.MinkowskiReLU()
        )

    def forward(self, x: BlockContent, batch_size: int):
        content = x.data
        cm = content.coordinate_manager
        key = content.coordinate_map_key

        # process each input feature type individually
        # concat all processed features and process with another encoder

        # process depth features
        start_features = 0
        end_features = self.num_depth_features

        features_depth = Me.SparseTensor(content.F[:, start_features:end_features], coordinate_manager=cm, coordinate_map_key=key)
        encoded_input = self.encoder_depth(features_depth) if not self.verbose else self.forward_verbose(features_depth, self.encoder_depth)

        # process image features
        start_features = end_features
        end_features += self.num_image_features

        features_features = Me.SparseTensor(content.F[:, start_features:end_features], coordinate_manager=cm, coordinate_map_key=key)
        encoded_features = self.encoder_features(features_features) if not self.verbose else self.forward_verbose(features_features, self.encoder_features)

        encoded_input = Me.cat(encoded_input, encoded_features)

        # process instance features
        start_features = end_features
        end_features += self.num_instance_features

        features_instances = Me.SparseTensor(content.F[:, start_features:end_features], coordinate_manager=cm, coordinate_map_key=key)
        encoded_instances = self.encoder_instances(features_instances) if not self.verbose else self.forward_verbose(features_instances, self.encoder_instances)

        encoded_input = Me.cat(encoded_input, encoded_instances)

        # process input features
        encoded = self.encoder(encoded_input) if not self.verbose else self.forward_verbose(encoded_input, self.encoder)

        # forward to next hierarchy
        processed: BlockContent = self.submodule(BlockContent(encoded, x.encoding), batch_size)

        if processed is None:
            return None

        sparse, dense = processed.data

        if sparse is not None:
            sparse = Me.SparseTensor(sparse.F, sparse.C, coordinate_manager=cm, tensor_stride=sparse.tensor_stride)

        # proxy occupancy output
        if sparse is not None:
            proxy_output = self.proxy_occupancy_128_head(sparse)
        else:
            proxy_output = None

        should_concat = proxy_output is not None

        if should_concat:
            proxy_mask = (Me.MinkowskiSigmoid()(proxy_output).F > config.MODEL.FRUSTUM3D.SPARSE_THRESHOLD_128).squeeze(1)

            # no valid voxels
            if proxy_mask.sum() == 0:
                cat = None
                proxy_instances = None
                proxy_semantic = None
            else:
                sparse_pruned = Me.MinkowskiPruning()(sparse, proxy_mask)  # mask out invalid voxels

                if len(sparse_pruned.C) == 0:
                    return BlockContent([None, [proxy_output, None, None], dense], processed.encoding)

                # Skip connection
                cat = utils.sparse_cat_union(encoded, sparse_pruned)

                # instance proxy prediction
                proxy_instances = self.proxy_instance_128_head(cat)
                proxy_semantic = self.proxy_semantic_128_head(cat)

                # Concat proxy outputs
                cat = utils.sparse_cat_union(cat, proxy_instances)
                cat = utils.sparse_cat_union(cat, proxy_semantic)

            if not config.MODEL.FRUSTUM3D.IS_LEVEL_128 and proxy_output is not None:
                output = self.decoder(cat) if not self.verbose else self.forward_verbose(cat, self.decoder)
                output = Me.SparseTensor(output.F, output.C, coordinate_manager=cm)  # Fix

            else:
                output = None
        else:
            output = None
            proxy_instances = None
            proxy_semantic = None

        if self.verbose:
            self.verbose = False

        return BlockContent([output, [proxy_output, proxy_instances, proxy_semantic], dense], processed.encoding)


class UNetBlockInner(UNetBlock):
    def __init__(self, num_inner_features: int, num_output_features: int):
        super().__init__(num_inner_features, num_inner_features, num_inner_features, num_output_features, None)

    def forward(self, x: BlockContent) -> BlockContent:
        content = x.data
        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        decoded = self.decoder(encoded) if not self.verbose else self.forward_verbose(encoded, self.decoder)
        output = torch.cat([content, decoded], dim=1)

        if self.verbose:
            self.verbose = False
        return BlockContent(output, encoded)


class UNetBlockOuter(UNetBlock):
    def __init__(self,
                 num_input_features: int, num_inner_features: int,
                 num_outer_features: int, num_output_features,
                 submodule: nn.Module):
        super().__init__(num_input_features, num_inner_features, num_outer_features, num_outer_features, submodule)

        self.encoder = nn.Sequential(
            nn.Conv3d(num_input_features, num_inner_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(num_inner_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_inner_features, num_inner_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(num_inner_features),
            nn.ReLU(inplace=True)
        )

        modules = list(self.decoder.children())[:-2]
        modules += [
            nn.ConvTranspose3d(num_outer_features, num_outer_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(num_outer_features), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_outer_features, num_output_features, 1, 1, 0),
        ]

        self.decoder = nn.Sequential(*modules)

    def forward(self, x: BlockContent) -> BlockContent:
        content = x.data
        encoded = self.encoder(x.data) if not self.verbose else self.forward_verbose(content, self.encoder)
        processed = self.submodule(BlockContent(encoded, None))
        output = self.decoder(processed.data) if not self.verbose else self.forward_verbose(processed.data, self.decoder)

        if self.verbose:
            self.verbose = False
        return BlockContent(output, processed.encoding)


class UNetBlockHybridSparse(UNetBlockOuter):
    """
    UNetBlock with a sparse encoder and sparse decoder.
    The encoder output is densified.
    The decoder input is sparsified.
    """

    def __init__(self,
                 num_input_features: int, num_inner_features: int,
                 num_outer_features: int, num_output_features: int,
                 submodule: nn.Module):
        super().__init__(num_input_features, num_inner_features, num_outer_features, num_output_features, submodule)

        downsample = Me.MinkowskiConvolution(num_input_features, num_inner_features, kernel_size=4, stride=2, bias=False, dimension=3)
        self.encoder = nn.Sequential(
            SparseBasicBlock(num_input_features, num_inner_features, stride=2, dimension=3, downsample=downsample),
            SparseBasicBlock(num_inner_features, num_inner_features, dimension=3)
        )

        self.num_inner_features = num_inner_features

        self.proxy_occupancy_64_head = nn.Sequential(nn.Linear(num_inner_features * 2, 1))

        self.proxy_instance_64_head = nn.Sequential(
            ResNetBlock3d(num_inner_features * 2, num_inner_features * 2),
            ResNetBlock3d(num_inner_features * 2, num_inner_features * 2),
            nn.Conv3d(num_inner_features * 2, config.MODEL.INSTANCE2D.MAX + 1, kernel_size=3, stride=1, padding=1,
                      bias=True)
        )

        self.proxy_semantic_64_head = nn.Sequential(
            ResNetBlock3d(num_inner_features * 2, num_inner_features * 2),
            ResNetBlock3d(num_inner_features * 2, num_inner_features * 2),
            nn.Conv3d(num_inner_features * 2, config.MODEL.FRUSTUM3D.NUM_CLASSES, kernel_size=3, stride=1, padding=1, bias=True)
        )

        num_conv_input_features = num_outer_features
        num_conv_input_features += config.MODEL.INSTANCE2D.MAX + 1
        num_conv_input_features += config.MODEL.FRUSTUM3D.NUM_CLASSES

        self.decoder = nn.Sequential(
            Me.MinkowskiConvolutionTranspose(num_conv_input_features, num_output_features, kernel_size=4, stride=2, bias=False, dimension=3, expand_coordinates=True), Me.MinkowskiInstanceNorm(num_output_features),
            Me.MinkowskiReLU(),
            SparseBasicBlock(num_output_features, num_output_features, dimension=3)
        )

    def forward(self, x: BlockContent, batch_size: int) -> BlockContent:
        # encode
        content = x.data
        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)

        # to dense at 64x64x64 with min_coordinate at 0,0,0
        shape = torch.Size([batch_size, self.num_inner_features, 64, 64, 64])
        min_coordinate = torch.IntTensor([0, 0, 0]).to(encoded.device)

        # mask out all voxels that are outside (<0 & >256)
        mask = (encoded.C[:, 1] < 256) & (encoded.C[:, 2] < 256) & (encoded.C[:, 3] < 256)
        mask = mask & (encoded.C[:, 1] >= 0) & (encoded.C[:, 2] >= 0) & (encoded.C[:, 3] >= 0)

        encoded = Me.MinkowskiPruning()(encoded, mask)

        if len(encoded.C) == 0:
            return BlockContent([None, None], content)

        dense, _, _ = encoded.dense(shape, min_coordinate=min_coordinate)

        # next hierarchy
        processed: BlockContent = self.submodule(BlockContent(dense, None))

        # decode
        # occupancy proxy -> mask
        proxy_flat = processed.data.view(batch_size, processed.data.shape[1], -1).permute(0, 2, 1)
        proxy_output = self.proxy_occupancy_64_head(proxy_flat)
        proxy_output = proxy_output.view(batch_size, 64, 64, 64)
        dense_to_sparse_mask: torch.Tensor = torch.sigmoid(proxy_output) > config.MODEL.FRUSTUM3D.DENSE_SPARSE_THRESHOLD

        # mask dense occupancy
        frustum_mask = x.encoding
        dense_to_sparse_mask = torch.masked_fill(dense_to_sparse_mask, frustum_mask.squeeze() == False, False)
        proxy_output = torch.masked_fill(proxy_output, frustum_mask.squeeze() == False, 0.0).unsqueeze(1)

        # instance proxy
        instance_prediction = self.proxy_instance_64_head(processed.data)
        instance_prediction = torch.masked_fill(instance_prediction, frustum_mask.squeeze() == False, 0.0)
        instance_prediction[:, 0] = torch.masked_fill(instance_prediction[:, 0], frustum_mask.squeeze() == False, 1.0)

        # semantic proxy
        semantic_prediction = self.proxy_semantic_64_head(processed.data)
        semantic_prediction = torch.masked_fill(semantic_prediction, frustum_mask.squeeze() == False, 0.0)
        semantic_prediction[:, 0] = torch.masked_fill(semantic_prediction[:, 0], frustum_mask.squeeze() == False, 1.0)

        proxy_output = [proxy_output, instance_prediction, semantic_prediction]

        if not config.MODEL.FRUSTUM3D.IS_LEVEL_64:
            coordinates, _, _ = transforms3d.Sparsify()(dense_to_sparse_mask, features=processed.data)
            locations = coordinates.long()

            dense_features = processed.data

            if instance_prediction is not None:
                dense_features = torch.cat([dense_features, instance_prediction], dim=1)

            if semantic_prediction is not None:
                dense_features = torch.cat([dense_features, semantic_prediction], dim=1)

            sparse_features = dense_features[locations[:, 0], :, locations[:, 1], locations[:, 2], locations[:, 3]]

            if coordinates.shape[0] == 0:
                return None

            coords_next = coordinates
            stride = encoded.tensor_stride[0]
            coords_next[:, 1:] *= stride  # "upsample coordinates"
            cm = encoded.coordinate_manager
            key, _ = cm.insert_and_map(coords_next, encoded.tensor_stride, string_id="decoded")
            sparse_features = Me.SparseTensor(sparse_features, coordinates=coords_next.int(), tensor_stride=4, coordinate_manager=cm)
            concat = utils.sparse_cat_union(encoded, sparse_features)

            output = self.decoder(concat) if not self.verbose else self.forward_verbose(concat, self.decoder)
        else:
            output = None
        if self.verbose:
            self.verbose = False
        return BlockContent([output, proxy_output], processed.encoding)

