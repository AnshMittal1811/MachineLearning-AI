import numpy as np
import trimesh


def create_grid_points_from_xyz_bounds(min_x, max_x, min_y, max_y, min_z, max_z, res):
    x = np.linspace(min_x, max_x, res)
    y = np.linspace(min_y, max_y, res)
    z = np.linspace(min_z, max_z, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=False)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


# conversion of points into coordinates understood by pytorch's grid_sample function
# bbox = [min_x,max_x,min_y,max_y,min_z,max_z]
def to_grid_sample_coords(points, bbox):
    min_values, max_values = bbox[::2], bbox[1::2]
    points = 2 * 1 / (max_values - min_values) * (points - min_values) - 1
    grid_coords = points.copy()
    grid_coords[:, 0], grid_coords[:, 2] = points[:, 2], points[:, 0]
    return grid_coords


def form_grid_to_original_coords(grid_points, bbox):
    min_values, max_values = bbox[::2], bbox[1::2]
    grid_points = (grid_points + 1) / 2 * \
        (max_values - min_values) + min_values
    points = grid_points.copy()
    points[:, 0], points[:, 2] = grid_points[:, 2], grid_points[:, 0]
    return points


def barycentric_coordinates(p, q, u, v):
    """
    Calculate barycentric coordinates of the given point
    :param p: a given point
    :param q: triangle vertex
    :param u: triangle vertex
    :param v: triangle vertex
    :return: 1X3 ndarray with the barycentric coordinates of p
    """
    v0 = u - q
    v1 = v - q
    v2 = p - q
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    y = (d11 * d20 - d01 * d21) / denom
    z = (d00 * d21 - d01 * d20) / denom
    x = 1.0 - z - y
    return np.array([x, y, z])


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh
