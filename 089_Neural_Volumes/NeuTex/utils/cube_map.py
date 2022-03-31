import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def convert_cube_uv_to_xyz(index, uvc):
    assert uvc.shape[-1] == 2
    vc, uc = uvc.unbind(-1)
    if index == 0:
        x = torch.ones_like(uc).to(uc.device)
        y = vc
        z = -uc
    elif index == 1:
        x = -torch.ones_like(uc).to(uc.device)
        y = vc
        z = uc
    elif index == 2:
        x = uc
        y = torch.ones_like(uc).to(uc.device)
        z = -vc
    elif index == 3:
        x = uc
        y = -torch.ones_like(uc).to(uc.device)
        z = vc
    elif index == 4:
        x = uc
        y = vc
        z = torch.ones_like(uc).to(uc.device)
    elif index == 5:
        x = -uc
        y = vc
        z = -torch.ones_like(uc).to(uc.device)
    else:
        raise ValueError(f"invalid index {index}")

    return F.normalize(torch.stack([x, y, z], axis=-1), dim=-1)


def load_cubemap(imgs):
    assert len(imgs) == 6
    return np.array([np.array(Image.open(img))[::-1] / 255.0 for img in imgs])


def sample_cubemap(cubemap, xyz):
    assert len(cubemap.shape) == 4
    assert cubemap.shape[0] == 6
    assert cubemap.shape[1] == cubemap.shape[2]
    assert xyz.shape[-1] == 3

    result = torch.zeros(xyz.shape[:-1] + (cubemap.shape[-1],)).float().to(xyz.device)

    x, y, z = xyz.unbind(-1)

    absX = x.abs()
    absY = y.abs()
    absZ = z.abs()

    isXPositive = x > 0
    isYPositive = y > 0
    isZPositive = z > 0

    maps = cubemap.unbind(0)
    masks = [
        isXPositive * (absX >= absY) * (absX >= absZ),
        isXPositive.logical_not() * (absX >= absY) * (absX >= absZ),
        isYPositive * (absY >= absX) * (absY >= absZ),
        isYPositive.logical_not() * (absY >= absX) * (absY >= absZ),
        isZPositive * (absZ >= absX) * (absZ >= absY),
        isZPositive.logical_not() * (absZ >= absX) * (absZ >= absY),
    ]

    uvs = []

    uc = -z[masks[0]] / absX[masks[0]]
    vc = y[masks[0]] / absX[masks[0]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = z[masks[1]] / absX[masks[1]]
    vc = y[masks[1]] / absX[masks[1]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[2]] / absY[masks[2]]
    vc = -z[masks[2]] / absY[masks[2]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[3]] / absY[masks[3]]
    vc = z[masks[3]] / absY[masks[3]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[4]] / absZ[masks[4]]
    vc = y[masks[4]] / absZ[masks[4]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = -x[masks[5]] / absZ[masks[5]]
    vc = y[masks[5]] / absZ[masks[5]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    for texture, mask, uv in zip(maps, masks, uvs):
        result[mask] = (
            F.grid_sample(
                texture.permute(2, 0, 1)[None],
                uv.view((1, -1, 1, 2)),
                padding_mode="border",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .view(uv.shape[:-1] + (texture.shape[-1],))
        )

    return result


def merge_cube_to_single_texture(cube, flip=True, rotate=True):
    """
    cube: (6,res,res,c)
    """
    assert cube.shape[0] == 6
    assert cube.shape[1] == cube.shape[2]
    res = cube.shape[1]
    result = torch.ones((3 * res, 4 * res, cube.shape[-1]))

    if flip:
        cube = cube.flip(1)
    if rotate:
        result[res : 2 * res, :res] = cube[0]
        result[res : 2 * res, res : 2 * res] = cube[5]
        result[res : 2 * res, 2 * res : 3 * res] = cube[1]
        result[res : 2 * res, 3 * res :] = cube[4]
        result[:res, res : 2 * res] = cube[2].flip(0, 1)
        result[2 * res : 3 * res, res : 2 * res] = cube[3].flip(0, 1)
    else:
        result[res : 2 * res, :res] = cube[1]
        result[res : 2 * res, res : 2 * res] = cube[4]
        result[res : 2 * res, 2 * res : 3 * res] = cube[0]
        result[res : 2 * res, 3 * res :] = cube[5]
        result[:res, res : 2 * res] = cube[2]
        result[2 * res : 3 * res, res : 2 * res] = cube[3]

    return result


def load_cube_from_single_texture(filename, rotate=True):
    img = np.array(Image.open(filename)) / 255.0
    assert img.shape[0] * 4 == img.shape[1] * 3
    res = img.shape[0] // 3
    if rotate:
        cube = [
            img[res : 2 * res, :res][::-1],
            img[res : 2 * res, 2 * res : 3 * res][::-1],
            img[:res, res : 2 * res][:, ::-1],
            img[2 * res : 3 * res, res : 2 * res][:, ::-1],
            img[res : 2 * res, 3 * res :][::-1],
            img[res : 2 * res, res : 2 * res][::-1],
        ]
    else:
        cube = [
            img[res : 2 * res, 2 * res : 3 * res][::-1],
            img[res : 2 * res, :res][::-1],
            img[:res, res : 2 * res][::-1],
            img[2 * res : 3 * res, res : 2 * res][::-1],
            img[res : 2 * res, res : 2 * res][::-1],
            img[res : 2 * res, 3 * res :][::-1],
        ]

    return cube


if __name__ == "__main__":

    cube = load_cube_from_single_texture('../run/checkpoints/114-cont-new/cube_view.png')
    # cube = load_cubemap(
    #     [
    #         "./cubemap/posx.jpg",
    #         "./cubemap/negx.jpg",
    #         "./cubemap/posy.jpg",
    #         "./cubemap/negy.jpg",
    #         "./cubemap/posz.jpg",
    #         "./cubemap/negz.jpg",
    #     ]
    # )

    cube = torch.tensor(cube).float()
    import trimesh

    mesh = trimesh.creation.icosphere(6)
    xyz = torch.tensor(mesh.vertices).float()

    colors = sample_cubemap(cube, xyz)

    import open3d

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz.data.numpy())
    pcd.colors = open3d.utility.Vector3dVector(colors.data.numpy())
    open3d.visualization.draw_geometries([pcd])
