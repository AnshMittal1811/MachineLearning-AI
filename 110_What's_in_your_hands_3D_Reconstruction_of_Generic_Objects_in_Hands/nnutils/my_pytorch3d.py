from typing import List, Tuple, Union, Callable
import torch
from pytorch3d.structures import Meshes as MeshesBase
from pytorch3d.loss.chamfer import _handle_pointcloud_input, knn_gather, knn_points


# overwrite pytorch3d: Meshes


class Meshes(MeshesBase):

    def __init__(
        self,
        verts=None,
        faces=None,
        textures=None,
        *,
        verts_normals=None,
    ) -> None:
        """
        Args:
            verts:
                Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the (x, y, z) coordinates of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  Meshes should be padded with fill value of 0 so they all have
                  the same number of vertices.
            faces:
                Can be either

                - List where each element is a tensor of shape (num_faces, 3)
                  containing the indices of the 3 vertices in the corresponding
                  mesh in verts which form the triangular face.
                - Padded long tensor of shape (num_meshes, max_num_faces, 3).
                  Meshes should be padded with fill value of -1 so they have
                  the same number of faces.
            textures: Optional instance of the Textures class with mesh
                texture properties.
            verts_normals:
                Optional. Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the normals of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  They should be padded with fill value of 0 so they all have
                  the same number of vertices.
                Note that modifying the mesh later, e.g. with offset_verts_,
                can cause these normals to be forgotten and normals to be recalculated
                based on the new vertex positions.

        Refer to comments above for descriptions of List and Padded representations.
        """
        super().__init__(verts, faces, textures, verts_normals=verts_normals)
        # reset shape for empty meshes
        if self.isempty():

            # Identify type of verts and faces.
            if isinstance(verts, list) and isinstance(faces, list):
                if self._N > 0:
                    if not (
                        all(v.device == self.device for v in verts)
                        and all(f.device == self.device for f in faces)
                    ):
                        self._num_verts_per_mesh = torch.tensor(
                            [len(v) for v in self._verts_list], device=self.device
                        )
                        self._num_faces_per_mesh = torch.tensor(
                            [len(f) for f in self._faces_list], device=self.device
                        )
            elif torch.is_tensor(verts) and torch.is_tensor(faces):
                if self._N > 0:
                    # Check that padded faces - which have value -1 - are at the
                    # end of the tensors
                    faces_not_padded = self._faces_padded.gt(-1).all(2)
                    self._num_faces_per_mesh = faces_not_padded.sum(1)

                    self._num_verts_per_mesh = torch.full(
                        size=(self._N,),
                        fill_value=self._V,
                        dtype=torch.int64,
                        device=self.device,
                    )

            else:
                raise ValueError(
                    "Verts and Faces must be either a list or a tensor with \
                        shape (batch_size, N, 3) where N is either the maximum \
                        number of verts or faces respectively."
                )

            # if self.isempty():
            #     self._num_verts_per_mesh = torch.zeros(
            #         (0,), dtype=torch.int64, device=self.device
            #     )
            #     self._num_faces_per_mesh = torch.zeros(
            #         (0,), dtype=torch.int64, device=self.device
            #     )

            # Set the num verts/faces on the textures if present.
            if textures is not None:
                shape_ok = self.textures.check_shapes(self._N, self._V, self._F)
                if not shape_ok:
                    msg = "Textures do not match the dimensions of Meshes."
                    raise ValueError(msg)

                self.textures._num_faces_per_mesh = self._num_faces_per_mesh.tolist()
                self.textures._num_verts_per_mesh = self._num_verts_per_mesh.tolist()
                self.textures.valid = self.valid

            if verts_normals is not None:
                self._set_verts_normals(verts_normals)
        else:
            pass





# overwrite pytorch3d: chamfer distance
def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    # _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    if point_reduction is None:
        return (cham_x, cham_y), (cham_norm_x, cham_norm_y)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals

