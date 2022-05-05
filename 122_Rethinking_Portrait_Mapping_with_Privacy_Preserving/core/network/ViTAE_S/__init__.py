from .models import ViTAE_noRC_MaxPooling_bias_basic_stages4_14
from .models import ViTAE_noRC_MaxPooling_DecoderV1
from .models import ViTAE_noRC_MaxPooling_Matting


__all__ = ['p3mnet_vitae_s']


def p3mnet_vitae_s(pretrained=True, **kwargs):
    encoder = ViTAE_noRC_MaxPooling_bias_basic_stages4_14(pretrained=pretrained, **kwargs)
    decoder = ViTAE_noRC_MaxPooling_DecoderV1()
    model = ViTAE_noRC_MaxPooling_Matting(encoder, decoder)
    return model
