from .ifnet_geometry_pose import IFNetGeometryPose
from .ifnet_geometry_bbox import IFNetGeometryBbox 
from .ifnet_texture_pos_encoding import IFNetTexturePoseEncoding
from .ifnet_geometry_smplgt_earlyfusion_attention import IFNetGeometrySMPLGT_EarlyFusion_Attention


def get_models():
    return {
        'IFNetGeometryPose': IFNetGeometryPose,
        'IFNetGeometryBbox': IFNetGeometryBbox,
        'IFNetGeometrySMPLGT_EarlyFusion_Attention': IFNetGeometrySMPLGT_EarlyFusion_Attention,
        'IFNetTexturePoseEncoding': IFNetTexturePoseEncoding
    }
