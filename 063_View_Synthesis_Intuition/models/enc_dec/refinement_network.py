from models.enc_dec.residual_block import *


class RefineNet(nn.Module):
    '''
    Refinement network (RN) that consists of ResidualBlocks of type "id", "avg" and "ups".
    Based on ResNetDecoder in architectures.py
    See Appendix B and fig. 15(b) in SynSin paper.
    '''

    def __init__(self,
                 res_block_dims=[],
                 res_block_types=[],
                 activate_out=nn.Sigmoid(),
                 noisy_bn=True,
                 spectral_norm=True):
        '''
        Let n-many ResNet blocks, res_block_dims include n+1 elements 
        to specify input and output channels for each block.
        '''
        super().__init__()

        self.res_blocks = []
        for i in range(len(res_block_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(
                    in_ch=res_block_dims[i],
                    out_ch=res_block_dims[i + 1],
                    block_type=res_block_types[i],
                    noisy_bn=noisy_bn,
                    spectral_norm=spectral_norm
                )
            )
        self.res_blocks = nn.Sequential(*self.res_blocks)
        # Final activation can be:
        # - nn.Sigmoid to force output to be in range [0,1]
        # - nn.Tanh to force output to be in range [-1,1]
        self.activate_out = activate_out

    def forward(self, x):
        x = self.res_blocks(x)

        if self.activate_out:
            x = self.activate_out(x)

        return x


class ParallelRefinementNetwork(nn.Module):
    '''
    Refinement and segmentation network based on ResidualBlock.
    Produces rgb image prediction and the corresponding segmentation mask
    RGB and Seg network have shared weights up to shared_layers number of blocks
    '''

    def __init__(self,
                 ref_block_dims=[],
                 ref_block_types=[],
                 seg_block_dims=[],
                 seg_block_types=[],
                 shared_layers=0,
                 activate_out=nn.Sigmoid(),
                 noisy_bn=True,
                 spectral_norm=True):
        '''
        :param ref_block_dims: channels of refinement network
        :param ref_block_types: refinement block types
        :param seg_block_dims: channels of seg network
        :param seg_block_types: seg block types
        :param shared_layers: number of shared blocks between rgb and seg network
        :param activate_out: activation function for output
        :param noisy_bn: whether to inject noise while performing batch norm
        :param spectral_norm: whether to use spectral norm
        '''

        super().__init__()

        self.shared_layers = shared_layers

        if shared_layers == 0:

            self.rgb_blocks = []
            self.seg_blocks = []

            for i in range(len(ref_block_dims) - 1):
                self.rgb_blocks.append(
                    ResidualBlock(
                        in_ch=ref_block_dims[i],
                        out_ch=ref_block_dims[i + 1],
                        block_type=ref_block_types[i],
                        noisy_bn=noisy_bn,
                        spectral_norm=spectral_norm
                    )
                )
            for i in range(len(seg_block_dims) - 1):
                self.seg_blocks.append(
                    ResidualBlock(
                        in_ch=seg_block_dims[i],
                        out_ch=seg_block_dims[i + 1],
                        block_type=seg_block_types[i],
                        noisy_bn=noisy_bn,
                        spectral_norm=spectral_norm
                    )
                )

            self.rgb_blocks = nn.Sequential(*self.rgb_blocks)
            self.seg_blocks = nn.Sequential(*self.seg_blocks)
            
        elif shared_layers < 0:
            self.rgb_blocks = []
            for i in range(len(ref_block_dims) - 1):
                self.rgb_blocks.append(
                    ResidualBlock(
                        in_ch=ref_block_dims[i],
                        out_ch=ref_block_dims[i + 1],
                        block_type=ref_block_types[i],
                        noisy_bn=noisy_bn,
                        spectral_norm=spectral_norm
                    )
                )
            self.rgb_blocks = nn.Sequential(*self.rgb_blocks)

        elif shared_layers >= len(ref_block_dims) - 1:
            raise ValueError("Invalid number of shared layers. Shared {0} >= {1} Refinement".format(shared_layers, len(
                ref_block_dims) - 1))

        elif not ref_block_dims[shared_layers] == seg_block_dims[0]:
            raise ValueError("Output dims ({0}) of shared blocks and input dims ({1}) of seg blocks dont match!".format(
                ref_block_dims[shared_layers], seg_block_dims[0]))

        else:

            self.shared_blocks = []
            self.seg_blocks = []
            self.rgb_blocks = []

            for i in range(shared_layers):
                self.shared_blocks.append(
                    ResidualBlock(
                        in_ch=ref_block_dims[i],
                        out_ch=ref_block_dims[i + 1],
                        block_type=ref_block_types[i],
                        noisy_bn=noisy_bn,
                        spectral_norm=spectral_norm
                    )
                )
            for i in range(len(ref_block_dims) - shared_layers - 1):
                self.rgb_blocks.append(
                    ResidualBlock(
                        in_ch=ref_block_dims[shared_layers + i],
                        out_ch=ref_block_dims[shared_layers + i + 1],
                        block_type=ref_block_types[shared_layers + i],
                        noisy_bn=noisy_bn,
                        spectral_norm=spectral_norm
                    )
                )
            for i in range(len(seg_block_dims) - 1):
                self.seg_blocks.append(
                    ResidualBlock(
                        in_ch=seg_block_dims[i],
                        out_ch=seg_block_dims[i + 1],
                        block_type=seg_block_types[i],
                        noisy_bn=noisy_bn,
                        spectral_norm=spectral_norm
                    )
                )
            self.shared_blocks = nn.Sequential(*self.shared_blocks)
            self.rgb_blocks = nn.Sequential(*self.rgb_blocks)
            self.seg_blocks = nn.Sequential(*self.seg_blocks)

        # Final activation can be:
        # - nn.Sigmoid to force output to be in range [0,1]
        # - nn.Tanh to force output to be in range [-1,1]
        self.activate_out = activate_out

    def forward(self, x, input_seg):

        # ignores input_seg for now because it is not supported, but still gets it as argument for API consistency.

        if self.shared_layers == 0:
            img = self.rgb_blocks(x)
            seg = self.seg_blocks(x)
        elif self.shared_layers < 0:
            img = self.rgb_blocks(x)
            seg = None
        else:
            x = self.shared_blocks(x)
            img = self.rgb_blocks(x)
            seg = self.seg_blocks(x)

        if self.activate_out:
            img = self.activate_out(img)
            seg = self.activate_out(seg)

        return img, seg


class SequentialRefinementNetwork(nn.Module):
    '''
    Refinement and segmentation network based on ResidualBlock.
    Produces rgb image prediction and the corresponding segmentation mask
    RGB gets predicted first and from that the segmentation network predicts the seg mask.
    '''

    def __init__(self,
                 rgb_block_dims=[],
                 rgb_block_types=[],
                 seg_block_dims=[],
                 seg_block_types=[],
                 activate_out=nn.Sigmoid(),
                 noisy_bn=[True,False],
                 spectral_norm=[True,False],
                 concat_input_seg=False):
        '''
        :param rgb_block_dims: channels of refinement network for rgb
        :param rgb_block_types: refinement block types for rgb
        :param seg_block_dims: channels of seg network
        :param seg_block_types: seg block types
        :param activate_out: activation function for output
        :param noisy_bn: whether to inject noise while performing batch norm
        :param spectral_norm: whether to use spectral norm
        :param concat_input_seg: whether the segmentation image of the input (unchanged in any way) should be concatenated along with the rgb prediction. This is a guide for the network which colors it should use (can be different for different scenes).
        '''

        super().__init__()

        if rgb_block_dims[-1] != seg_block_dims[0]:
            raise ValueError(f"Last block dim of rgb and first block dim of seg must be equal, but they are: {rgb_block_dims[-1]} (rgb) != {seg_block_dims[0]} (seg). If you want to use concat_input_seg you still need to specify matching dims, the concatenation will be done anyways internally.")

        self.rgb_blocks = self.create_from_dims_and_type(rgb_block_dims, rgb_block_types, noisy_bn[0], spectral_norm[0])
        self.seg_blocks = self.create_from_dims_and_type(seg_block_dims, seg_block_types, noisy_bn[1], spectral_norm[1], concat_input_seg)

        self.concat_input_seg = concat_input_seg

        # Final activation can be:
        # - nn.Sigmoid to force output to be in range [0,1]
        # - nn.Tanh to force output to be in range [-1,1]
        self.activate_out = activate_out

    def create_from_dims_and_type(self, dims, type, noisy_bn, spectral_norm, concat_input_seg=False):
        blocks = []

        for i in range(len(dims) - 1):
            in_ch = dims[i]
            if i == 0 and concat_input_seg:
                in_ch += 3
            blocks.append(
                ResidualBlock(
                    in_ch=in_ch,
                    out_ch=dims[i + 1],
                    block_type=type[i],
                    noisy_bn=noisy_bn,
                    spectral_norm=spectral_norm
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, rgb_features, input_seg):

        rgb_img = self.rgb_blocks(rgb_features)
        if self.concat_input_seg:
            seg_input = torch.cat((rgb_img, input_seg), 1) # tensors are of shape BS x C x H x W --> concat at channel dimension
            seg = self.seg_blocks(seg_input)
        else:
            seg = self.seg_blocks(rgb_img)

        if self.activate_out:
            rgb_img = self.activate_out(rgb_img)
            seg = self.activate_out(seg)

        return rgb_img, seg
