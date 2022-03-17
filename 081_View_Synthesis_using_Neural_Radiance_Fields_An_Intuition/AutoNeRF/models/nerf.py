import torch
import torch.nn as nn

from typing import Optional
import matplotlib.pyplot as plt




# Chunksize (Note: this isn't batchsize in the conventional sense. This only
# specifies the number of rays to be queried in one go. Backprop still happens
# only after all rays from the current "bundle" are queried and rendered).
chunksize = 16384  # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#--------------------------------------------------------------------------------------------------------------------

def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
	r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
	Each element of the list (except possibly the last) has dimension `0` of length
	`chunksize`.
	"""
	return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


#--------------------------------------------------------------------------------------------------------------------
def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor):
	r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

	Args:
	tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
	  is to be computed.

	Returns:
	cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
	  tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
	"""
	# TESTED
	# Only works for the last dimension (dim=-1)
	dim = -1
	# Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
	cumprod = torch.cumprod(tensor, dim)
	# "Roll" the elements along dimension 'dim' by 1 element.
	cumprod = torch.roll(cumprod, 1, dim)
	# Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
	cumprod[..., 0] = 1.

	return cumprod

#--------------------------------------------------------------------------------------------------------------------
def get_ray_bundle(height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor):
	r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

	Args:
	height (int): Height of an image (number of pixels).
	width (int): Width of an image (number of pixels).
	focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
	tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
	  transforms a 3D point from the camera frame to the "world" frame for the current example.

	Returns:
	ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
	  each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
	  row index `j` and column index `i`.
	  (TODO: double check if explanation of row and col indices convention is right).
	ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
	  direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
	  passing through the pixel at row index `j` and column index `i`.
	  (TODO: double check if explanation of row and col indices convention is right).
	"""
	# TESTED
	ii, jj = meshgrid_xy(
	  torch.arange(width).to(tform_cam2world),
	  torch.arange(height).to(tform_cam2world)
	)
	directions = torch.stack([(ii - width * .5) / focal_length,
	                        -(jj - height * .5) / focal_length,
	                        -torch.ones_like(ii)
	                       ], dim=-1)
	ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
	ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
	return ray_origins, ray_directions


#--------------------------------------------------------------------------------------------------------------------
def compute_query_points_from_rays(
	ray_origins: torch.Tensor,
	ray_directions: torch.Tensor,
	near_thresh: float,
	far_thresh: float,
	num_samples: int,
	randomize: Optional[bool] = True
	) -> (torch.Tensor, torch.Tensor):
	r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
	variables indicate the bounds within which 3D points are to be sampled.

	Args:
	ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
	  `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
	ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
	  `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
	near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
	  coordinate that is of interest/relevance).
	far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
	  coordinate that is of interest/relevance).
	num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
	  randomly, whilst trying to ensure "some form of" uniform spacing among them.
	randomize (optional, bool): Whether or not to randomize the sampling of query points.
	  By default, this is set to `True`. If disabled (by setting to `False`), we sample
	  uniformly spaced points along each ray in the "bundle".

	Returns:
	query_points (torch.Tensor): Query points along each ray
	  (shape: :math:`(width, height, num_samples, 3)`).
	depth_values (torch.Tensor): Sampled depth values along each ray
	  (shape: :math:`(num_samples)`).
	"""
	# TESTED
	# shape: (num_samples)
	depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
	if randomize is True:
		# ray_origins: (width, height, 3)
		# noise_shape = (width, height, num_samples)
		noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
		# depth_values: (num_samples)
		depth_values = depth_values \
		    + torch.rand(noise_shape).to(ray_origins) * (far_thresh
		        - near_thresh) / num_samples
	# (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
	# query_points:  (width, height, num_samples, 3)
	query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
	# TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
	return query_points, depth_values


#--------------------------------------------------------------------------------------------------------------------
def render_volume_density(
	radiance_field: torch.Tensor,
	ray_origins: torch.Tensor,
	depth_values: torch.Tensor
	) -> (torch.Tensor, torch.Tensor, torch.Tensor):
	r"""Differentiably renders a radiance field, given the origin of each ray in the
	"bundle", and the samlped depth values along them.

	Args:
	radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
	  we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
	  the paper) (shape: :math:`(width, height, num_samples, 4)`).
	ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
	  `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
	depth_values (torch.Tensor): Sampled depth values along each ray
	  (shape: :math:`(num_samples)`).

	Returns:
	rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
	depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
	acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
	  transmittance map).
	"""
	# TESTED
	sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
	rgb = torch.sigmoid(radiance_field[..., :3])
	one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
	dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
	              one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
	alpha = 1. - torch.exp(-sigma_a * dists)
	weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

	rgb_map = (weights[..., None] * rgb).sum(dim=-2)
	depth_map = (weights * depth_values).sum(dim=-1)
	acc_map = weights.sum(-1)

	return rgb_map, depth_map, acc_map


#--------------------------------------------------------------------------------------------------------------------
def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
	):
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        num_encoding_functions (optional, int): Number of encoding functions used to
            compute a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            computed positional encoding (default: True).
        log_sampling (optional, bool): Sample logarithmically in frequency space, as
            opposed to linearly (default: True).
    
    Returns:
        (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    # Now, encode the input using a set of high-frequency functions and append the
    # resulting values to the encoding.
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


#--------------------------------------------------------------------------------------------------------------------
class TinyNerfModel(torch.nn.Module):
	r"""Define a "very tiny" NeRF model comprising three fully connected layers.
	"""
	def __init__(self, filter_size=128, num_encoding_functions=6):
		super(TinyNerfModel, self).__init__()
		# Input layer (default: 39 -> 128)
		self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
		# Layer 3 (default: 128 -> 128)
		self.layer2 = torch.nn.Linear(filter_size, filter_size)
		# Layer 3 (default: 128 -> 64)
		self.layer3 = torch.nn.Linear(filter_size, 64)
		# Layer 3 (default: 64 -> 32)
		self.layer4 = torch.nn.Linear(64, 32)
		# Layer 3 (default: 32 -> 4)
		self.layer5 = torch.nn.Linear(32, 4)
		# Short hand for torch.nn.functional.relu
		self.relu = torch.nn.functional.leaky_relu

	def forward(self, x):
		x = self.relu(self.layer1(x))
		x = self.relu(self.layer2(x))
		x = self.relu(self.layer3(x))
		x = self.relu(self.layer4(x))
		x = self.layer5(x)
		return x


#--------------------------------------------------------------------------------------------------------------------
# One iteration of TinyNeRF (forward pass).
def run_one_iter_of_tinynerf(model, height, width, focal_length, tform_cam2world,
                             near_thresh, far_thresh, depth_samples_per_ray,
                             encoding_function, get_minibatches_function):
    
    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length,
                                                tform_cam2world)
    
    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)

    return rgb_predicted


#--------------------------------------------------------------------------------------------------------------------
def nerf_prediction(model, height, width, focal_length, tform_cam2world,
                             near_thresh, far_thresh, depth_samples_per_ray,
                             encoding_function, get_minibatches_function):
    with torch.no_grad():
        prediction = run_one_iter_of_tinynerf(model, height, width, focal_length, tform_cam2world,
                             near_thresh, far_thresh, depth_samples_per_ray,
                             encoding_function, get_minibatches_function)
    torch.cuda.empty_cache()
    return prediction

