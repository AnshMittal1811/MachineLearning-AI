from .swin_stem_pooling5_transformer import swin_stem_pooling5_encoder
from .swin_stem_pooling5_transformer import SwinStemPooling5TransformerMatting

from .decoder import SwinStemPooling5TransformerDecoderV1


__all__ = ['p3mnet_swin_t']


def p3mnet_swin_t(pretrained=True, img_size=512, **kwargs):
    encoder = swin_stem_pooling5_encoder(pretrained=pretrained, img_size=img_size, **kwargs)
    decoder = SwinStemPooling5TransformerDecoderV1()
    model = SwinStemPooling5TransformerMatting(encoder, decoder)
    return model
