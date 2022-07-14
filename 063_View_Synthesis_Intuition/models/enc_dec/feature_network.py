from models.enc_dec.residual_block import *

class FeatureNet(nn.Module):
    '''
    Spatial feature network (SFN)/Feature encoder that consists of ResidualBlocks of type "id" only.
    Based on ResNetEncoder in architectures.py
    See Appendix B and fig. 15(a) in SynSin paper.
    '''
    def __init__(self, 
                 res_block_dims=[], 
                 res_block_types=[],
                 noisy_bn=True, 
                 spectral_norm=True):
        '''
        Let n-many ResNet blocks, res_block_dims include n+1 elements 
        to specify input and output channels for each block.
        '''
        super().__init__()

        self.res_blocks = [] 
        for i in range(len(res_block_dims)-1):
            self.res_blocks.append( 
                ResidualBlock(
                    in_ch=res_block_dims[i],
                    out_ch=res_block_dims[i+1],
                    block_type=res_block_types[i], # In SynSin block_type is always "id"
                    noisy_bn=noisy_bn, 
                    spectral_norm=spectral_norm
                )
            )
        self.res_blocks = nn.Sequential(*self.res_blocks)

    def forward(self, x):
        return self.res_blocks(x)