from .conv_template import ConvTemplate, SmallConvTemplate, SmallConvTemplate2D
from .standard_volume_decoder import StandardVolumeDecoder
from .geometry_volume_decoder import GeometryVolumeDecoder, GeometryVolumeMlpDecoder
from .geometry_mlp_decoder import GeometryMlpDecoder


def find_template_class(name: str):
    if name == "conv":
        return ConvTemplate
    if name == "small_conv":
        return SmallConvTemplate
    if name == "small_conv_2d":
        return SmallConvTemplate2D
    raise Exception("Unknown template class name {}".format(name))
