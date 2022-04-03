from .models.projection_layer import get_projection_layer
from .models.simple_nvp import SimpleNVP


def get_invertible_network(
    # these are from neural-parts D-FAUST config
    n_layers=4,
    feature_dims=256 * 2,
    hidden_size=256,
    proj_dims=128,
    proj_type="simple",
    checkpoint=False,
    normalize=True,
    explicit_affine=True,
):
    net = SimpleNVP(
        n_layers=n_layers,
        feature_dims=feature_dims,
        hidden_size=hidden_size,
        projection=get_projection_layer(proj_dims=proj_dims, type=proj_type),
        checkpoint=checkpoint,
        normalize=normalize,
        explicit_affine=explicit_affine,
    )
    # self.config["invertible_network"]["n_blocks"], # 4
    # 2*self.feature_extractor.feature_size, # 2*256
    # self.config["invertible_network"]["hidden_size"], # 256
    # get_projection_layer(**self.config["projection_layer"]), #
    # self.config["invertible_network"]["checkpoint"], false
    # self.config["invertible_network"]["normalize"], true
    # self.config["invertible_network"]["explicit_affine"], true
    return net
