"""
Utilities for geometry etc.
"""

from collections import OrderedDict
import numpy as np
import torch
import transforms3d
from scipy.spatial.transform import Rotation as R


def transform_vectors4(matrix: np.array, vectors4: np.array) -> np.array:
    """
    Left-multiplies 4x4 @ Nx4. Returns Nx4.
    """
    assert vectors4.shape[-1] == 4
    if len(vectors4.shape) == 1:
        vectors4 = vectors4[None, ...]
    #res = np.einsum('ij,hj->hi', matrix, vectors4)
    res = (matrix @ vectors4.swapaxes(-1, -2)).swapaxes(-1, -2)
    if len(vectors4.shape) == 1:
        res = res[0, ...]
    return res


def transform_directions(matrix: np.array, vecs: np.array, return_euclidean: bool = False) -> np.array:
    """
    Left-multiplies 4x4 @ Nx(3/4). Returns Nx4 or Nx3.
    """
    if vecs.shape[-1] == 3:
        vecs = np.concatenate((vecs, np.zeros_like(vecs[..., :1])), axis=1)
    res = transform_vectors4(matrix, vecs)
    if return_euclidean:
        res = res[..., :3]
    return res


def transform_points(matrix: np.array, points: np.array, return_euclidean: bool = False) -> np.array:
    """
    Left-multiplies 4x4 @ Nx(3/4). Returns Nx4 or Nx3.
    """
    if points.shape[-1] == 3:
        points = np.concatenate((points, np.ones_like(points[..., :1])), axis=-1)
    res = transform_vectors4(matrix, points)
    if return_euclidean:
        res = res[..., :3] / res[..., 3:].repeat(3, axis=-1)
    return res


def transform_normals(matrix: np.array, normals: np.array) -> np.array:
    """
    Left-multiplies 4x4 @ Nx(3/4). Uses T^-1 for normals.
    """
    normals = transform_directions(np.linalg.inv(matrix.T), normals, return_euclidean=True)
    normals = normalize_vecs(normals)
    return normals


def normalize_vecs(vectors: np.array) -> np.array:
    """
    Normalize vector lengths.
    """
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def decompose_matrix4x4(matrix: np.array):
    """
    Decomposes affine matrix to T @ R @ Z @ S
    https://matthew-brett.github.io/transforms3d/reference/transforms3d.affines.html#transforms3d.affines.decompose
    """
    T, R, Z, S = transforms3d.affines.decompose(matrix.astype(np.float64))
    # Now we make an affine matrix
    mat_T = np.eye(4, dtype=np.float32)
    mat_T[:3, 3] = T
    mat_R = np.eye(4, dtype=np.float32)
    mat_R[:3, :3] = R
    mat_Z = np.eye(4, dtype=np.float32)
    mat_Z[:3, :3] = np.diag(Z)
    mat_S = np.array([[1, S[0], S[1], 0],
                      [0, 1, S[2], 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], np.float32)
    matrix_test = mat_T @ mat_R @ mat_Z @ mat_S
    assert np.max(np.abs(matrix_test - matrix)) < 1e-3
    return mat_T, mat_R, mat_Z, mat_S


def getPerspectiveProjection(h_fov_deg: float, v_fov_deg: float):
    """
    Shortcut for simple perspective projection.
    """
    near = 0.1
    far = 2
    htan = np.tan(np.radians(h_fov_deg / 2)) * near
    vtan = np.tan(np.radians(v_fov_deg / 2)) * near
    return glFrustrum(-vtan, vtan, -htan, htan, near, far)


def glOrtho(b, t, l, r, n, f):
    """
    Get OpenGL ortho projection matrix.
    https://docs.microsoft.com/en-us/windows/win32/opengl/glortho
    """
    M = np.zeros((4, 4), np.float32)
    M[0, 0] = 2 / (r - l)
    M[1, 1] = 2 / (t - b)
    M[2, 2] = -2 / (f - n)

    M[0, 3] = (r + l) / (r - l)
    M[1, 3] = (t + b) / (t - b)
    M[2, 3] = (f + n) / (f - n)
    M[3, 3] = 1

    return M


def glFrustrum(b, t, l, r, n, f):
    """
    Get OpenGL projection matrix.
    """
    M = np.zeros((4, 4), np.float32)
    # set OpenGL perspective projection matrix
    M[0, 0] = 2 * n / (r - l)
    M[0, 1] = 0
    M[0, 2] = 0
    M[0, 3] = 0

    M[1, 0] = 0
    M[1, 1] = 2 * n / (t - b)
    M[1, 2] = 0
    M[1, 3] = 0

    M[2, 0] = (r + l) / (r - l)
    M[2, 1] = (t + b) / (t - b)
    M[2, 2] = -(f + n) / (f - n)
    M[2, 3] = -1

    M[3, 0] = 0
    M[3, 1] = 0
    M[3, 2] = -2 * f * n / (f - n)
    M[3, 3] = 0

    return M.T


# def glFrustrum2(b, t, l, r, n, f):
#     """
#     Get OpenGL projection matrix.
#     Corrected version based on the doc:

#     """
#     M = np.zeros((4, 4), np.float32)

#     M[0, 0] = 2 * n / (r - l)
#     M[1, 1] = 2 * n / (t - b)

#     M[0, 2] = (r + l) / (r - l)
#     M[1, 2] = (t + b) / (t - b)
#     M[2, 2] = (f + n) / (f - n)
#     M[3, 2] = -1

#     M[2, 3] = 2 * f * b / (f - n)

#     return M


def decompose_projection_matrix(projection_matrix: np.array):
    """
    Extracts near and far planes from glFrustrum matrix.
    http://docs.gl/gl3/glFrustum
    http://dougrogers.blogspot.com/2013/02/how-to-derive-near-and-far-clip-plane.html
    """

    def check_match(name, test, gt):
        if abs(gt - test) > 1e-3:
            print(f'{name}: Test = {test:.5f} vs. GT = {gt:.5f}')
            raise RuntimeError("Mismatch!")

    mat = projection_matrix
    n = mat[2, 3] / (mat[2, 2] - 1.0)
    f = mat[2, 3] / (mat[2, 2] + 1.0)

    # D
    D_ref = -2 * f * n / (f - n)
    D_in = mat[2, 3]
    check_match("D", D_in, D_ref)

    # C
    C_ref = -(f + n) / (f - n)
    C_in = mat[2, 2]
    check_match("C", C_in, C_ref)

    check_match("M32", mat[3, 2], -1)

    # RL
    L = (mat[0, 2] - 1) / mat[0, 0] * n
    R = (mat[0, 2] + 1) / mat[0, 0] * n
    B = (mat[1, 2] - 1) / mat[1, 1] * n
    T = (mat[1, 2] + 1) / mat[1, 1] * n
    # print(f'L = {L:.5f} | R = {R:.5f} | B = {B:.5f} | T = {T:.5f}')

    # print(f'\tSize = {R - L} x {T - B}')
    # fov = np.arctan2(np.array([R - L, T - B]) / 2, n) / np.pi * 180 * 2
    # print(f'\tFOV = {fov}')

    return OrderedDict([
        ('l', L),
        ('r', R),
        ('b', B),
        ('t', T),
        ('n', n),
        ('f', f),
    ])


def crop_projection_matrix(projection_matrix: np.array, viewport: np.array):
    """
    Restricts the viewport to given rectangle.
    """
    params = decompose_projection_matrix(projection_matrix)

    # Apply crop
    image_pos = np.array([params['l'], params['b']])
    image_size = np.array([params['r'] - params['l'], params['t'] - params['b']])

    # Shift.
    image_pos += image_size * viewport[: 2]

    # Scale.
    image_size *= viewport[2:]

    # Update params.
    params['l'] = image_pos[0]
    params['b'] = image_pos[1]
    params['r'] = image_pos[0] + image_size[0]
    params['t'] = image_pos[1] + image_size[1]

    return glFrustrum(**params)


def dot(x: np.array, y: np.array):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)


def dir2Angles(vector: np.array) -> np.array:
    """
    Returns yaw, pitch, radius.
    Z+ is the zero.
    """
    radius = np.linalg.norm(vector, axis=-1)
    vector = vector / radius[..., None]
    return np.stack([
        np.arctan2(vector[..., 0], vector[..., 2]),
        np.arcsin(vector[..., 1]),
        radius
    ], -1)


def angles2Dir(angles: np.array) -> np.array:
    """
    Returns yaw, pitch, radius.
    Z- is the zero.
    """
    if angles.shape[-1] == 3:
        radius = angles[..., 2]
    else:
        radius = np.ones_like(angles[..., 1])
    return np.stack([
        radius * np.sin(angles[..., 0]) * np.cos(angles[..., 1]),
        radius * np.sin(angles[..., 1]),
        radius * np.cos(angles[..., 0]) * np.cos(angles[..., 1]),
    ], -1)


def interpolate_views(mat_a: np.array, mat_b: np.array, ratio: float):
    """
    Decomposes to rot and trans and interpolates linearly.
    """
    # Decompose.
    rot_a = R.from_matrix(mat_a[:3, :3]).as_quat()
    trans_a = mat_a[:3, 3]
    rot_b = R.from_matrix(mat_b[:3, :3]).as_quat()
    trans_b = mat_b[:3, 3]

    # Interpolate.
    rot_c = rot_a + (rot_b - rot_a) * ratio
    trans_c = trans_a + (trans_b - trans_a) * ratio

    # Recompose.
    mat_c = np.eye(4, dtype=mat_a.dtype)
    mat_c[:3, :3] = R.from_quat(rot_c).as_matrix()
    mat_c[:3, 3] = trans_c
    return mat_c


def mflip(x, y, z):
    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = x
    transform[1, 1] = y
    transform[2, 2] = z
    return transform
