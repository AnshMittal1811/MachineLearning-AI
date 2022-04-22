# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from IPython import display


def mask_to_bbox(mask, mode='minmax', rate=1):
    """
    Args:
        mask (H, W)
    """
    h_idx, w_idx = np.where(mask > 0)
    if len(h_idx) == 0:
        return np.array([0, 0, 0, 0])

    if mode == 'minmax':
        y1, y2 = h_idx.min(), h_idx.max()
        x1, x2 = w_idx.min(), w_idx.max()
    elif mode == 'com':
        y_c, x_c = h_idx.mean(), w_idx.mean()
        y_l, x_l = h_idx.std(), w_idx.std()

        x1, y1 = x_c - x_l, y_c - y_l
        x2, y2 = x_c + x_l, y_c + y_l
    elif mode == 'med':
        h_idx, w_idx = np.sort(h_idx), np.sort(w_idx)

        idx25 = len(h_idx) // 4
        idx75 = len(h_idx) * 3 // 4
        y_c, x_c = h_idx[len(h_idx) // 2], w_idx[len(w_idx) // 2]
        y_l, x_l = h_idx[idx75] - h_idx[idx25], w_idx[idx75] - w_idx[idx25]

        x1, y1 = x_c - rate*x_l, y_c - rate*y_l
        x2, y2 = x_c + rate*x_l, y_c + rate*y_l

    return np.array([x1, y1, x2, y2])

def joint_bbox(*bboxes):
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0].min()
    y1 = bboxes[:, 1].min()
    x2 = bboxes[:, 2].max()
    y2 = bboxes[:, 3].max()
    return np.array([x1, y1, x2, y2])


def crop_weak_cam(cam, bbox_topleft, oldo2n, 
    new_center, new_size, old_size=224, resize=224):
    """
    Args:
        cam ([type]): [description]
        bbox_topleft ([type]): [description]
        scale ([type]): [description]
        new_bbox ([type]): [description] 
    """
    cam = cam.copy()
    s, t = np.split(cam, [1, ], -1)
    prev_center = bbox_topleft + (old_size / 2) / oldo2n
    offset = (prev_center - new_center)

    newo2n = resize/new_size
    
    # t += offset / (resize / 2) / s  * oldo2n
    s *=  newo2n / oldo2n * old_size / resize
    t += 2 *  newo2n * offset / resize / s
    new_cam = np.concatenate([s, t], -1)

    new_tl = new_center - new_size / 2
    new_scale = newo2n
    return new_cam, new_tl, new_scale


def ndc_to_screen_intr(cam, H, W):
    max_size = max(H, W)
    min_size = min(H, W)
    px, py = (max_size - W) // 2, (max_size - H) // 2
    k = torch.FloatTensor([
        [max_size / 2, 0, max_size / 2],
        [0, max_size / 2, max_size / 2],
        [0, 0, 1],
    ]).to(cam)
    out = k @ cam
    return out

def screen_intr_to_ndc_fp(cam, H, W):
    """
    Args:
        cam ([type]): (3, 3)
        H ([type]): [description]
        W ([type]): [description]
    Returns:
        [type]: (N, 2) (N, 2)
    """
    device = cam.device
    k = torch.FloatTensor([
        [2 / W, 0, -1],
        [0, 2 / H, -1],
        [0, 0,      1],
    ]).to(device)
    out = k @ cam
    f = torch.diagonal(out, dim1=-1, dim2=-2)[..., :2]
    p = out[..., 0:2, 2]
    
    return f, p    


def jitter_bbox(bbox, s_stdev, t_stdev):
    x1y1, x2y2 = bbox[..., :2], bbox[..., 2:]
    center = (x1y1 + x2y2) / 2 
    ori_size = x2y2 - x1y1

    jitter_s = torch.exp(torch.rand(1) * s_stdev * 2 - s_stdev)
    new_size = ori_size * jitter_s

    jitter_t = torch.rand(2) * t_stdev * 2 - t_stdev
    jitter_t = ori_size * jitter_t

    center += jitter_t
    bbox[0:2] = center - new_size / 2
    bbox[2:4] = center + new_size / 2   
    return bbox 


def square_bbox(bbox, pad=0):
    if not torch.is_tensor(bbox):
        is_numpy = True
        bbox = torch.FloatTensor(bbox)
    else:
        is_numpy = False

    x1y1, x2y2 = bbox[..., :2], bbox[..., 2:]
    center = (x1y1 + x2y2) / 2 
    half_w = torch.max((x2y2 - x1y1) / 2, dim=-1)[0]
    half_w = half_w * (1 + 2 * pad)
    bbox = torch.cat([center - half_w, center + half_w], dim=-1)
    if is_numpy:
        bbox = bbox.cpu().detach().numpy()
    return bbox


def crop_cam_intr(cam_intr, bbox_sq, H):
    x1y1 = bbox_sq[..., 0:2]
    dxy = bbox_sq[..., 2:] - bbox_sq[..., 0:2]
    t_mat = torch.FloatTensor([
        [1, 0, -x1y1[0]],
        [0, 1, -x1y1[1]],
        [0, 0, 1],
    ]).to(cam_intr)
    s_mat = torch.FloatTensor([
        [H / dxy[0], 0, 0],
        [0, H / dxy[1], 0],
        [0, 0, 1],
    ]).to(cam_intr)
    mat = s_mat @ t_mat @ cam_intr
    return mat


def frank_pad_and_resize(img, hand_bbox, add_margin=True, final_size=224):
    """
    :param img:
    :param hand_bbox:
    :param add_margin:
    :param final_size:
    :return: scale: xscale: original -> final (224)
             bbox: x1, y1, x2, y2 in original space. might be out of screen
    """
    ori_height, ori_width = img.shape[:2]
    min_x, min_y = hand_bbox[:2].astype(np.int32)
    max_x, max_y = hand_bbox[2:].astype(np.int32)
    # width, height = hand_bbox[2:].astype(np.int32)
    # max_x = min_x + width
    # max_y = min_y + height
    width = max_x - min_x
    height = max_y - min_y

    # make it square, unless hit the boundary
    if width > height:
        margin = (width - height) // 2
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, ori_height)
    else:
        margin = (height - width) // 2
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, ori_width)

    # add additional margin
    if add_margin:
        margin = int(0.3 * (max_y - min_y))  # if use loose crop, change 0.3 to 1.0
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, ori_height)
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, ori_width)

    new_size = max(max_x - min_x, max_y - min_y)
    bbox_processed = [min_x, min_y, min_x + new_size, min_y + new_size]

    ratio = final_size / new_size
    return ratio, np.array(bbox_processed)


def crop_resize(img: np.ndarray, bbox, final_size=224, pad='constant', return_np=True, **kwargs):
    # todo: joint effect
    ndim = img.ndim
    img_y, img_x = img.shape[0:2]

    min_x, min_y, max_x, max_y = np.array(bbox).astype(int)
    # pad
    pad_x1, pad_y1 = max(-min_x, 0), max(-min_y, 0)
    pad_x2, pad_y2 = max(max_x - img_x, 0), max(max_y - img_y, 0)
    pad_dim = ((pad_y1, pad_y2), (pad_x1, pad_x2), )
    if ndim == 3:
        pad_dim += ((0, 0), )
    img = np.pad(img, pad_dim, mode=pad, **kwargs)

    min_x += pad_x1
    max_x += pad_x1
    min_y += pad_y1
    max_y += pad_y1
    
    img = Image.fromarray(img.astype(np.uint8))
    img = img.crop([min_x, min_y, max_x, max_y])
    img = img.resize((final_size, final_size))

    # img_cropped = img[int(min_y):int(max_y), int(min_x):int(max_x)]
    # new_size = max(max_x - min_x, max_y - min_y)
    #
    # if ndim == 2:
    #     new_img = np.zeros((new_size, new_size), dtype=np.uint8)
    # else:
    #     new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    # new_img[:(max_y - min_y), :(max_x - min_x)] = img_cropped

    # new_img = Image.fromarray(new_img)
    # new_img = new_img.resize((final_size, final_size), Image.BICUBIC)
    # new_img = cv2.resize(new_img, (final_size, final_size))
    if return_np:
        img = np.array(img)
    return img


def affine_image(image, res, affine_trans=None, augment=None):
    """ apply 2D afine transformation to image
    :param image: numpy / Image of (H, W, C)
    :param [augs]: [center (2), scale (1), final_size HW(2), rot (1)]
    :return: Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if affine_trans is None:
        affine_trans, _ = get_affine_trans(*augment)
    trans = np.linalg.inv(affine_trans)

    image = image.transform(
        tuple(res), Image.AFFINE, (trans[0, 0], trans[0, 1], trans[0, 2],
                                   trans[1, 0], trans[1, 1], trans[1, 2]))
    image = image.crop((0, 0, res[1], res[0]))  # x2/W, y2
    return image

def affine_coords(pts, affine_trans=None, augment=None, invert=False):
    """apply 2D affine transform to points
    :return: (N, 2)
    """
    if affine_trans is None:
        affine_trans, _ = get_affine_trans(*augment)
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows.astype(int)


def get_affine_trans(center, scale, res, rot=0):
    """
    from Yana move ROI to [0, 0, res, res]
    :param center:
    :param scale:
    :param res:
    :param rot:
    :return: the transformation matrix, post_rot (3, 3)
    """
    homo_center = np.array(center.tolist() + [1,])

    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # todo: Rotate center to obtain coordinate of center in rotated image
    ori_rot_center = rot_mat.dot(homo_center)[:2]
    t_mat = np.eye(3)
    t_mat[0:2, 2] = [-res[1] / 2, -res[0] / 2]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(homo_center)

    post_rot_trans = get_affine_trans_no_rot(ori_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2],
                                                   scale, res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(
        np.float32)


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / scale
    affinet[1, 1] = float(res[0]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet


def get_bbox(skel2d):
    x1, y1 = skel2d.min(0)
    x2, y2 = skel2d.max(0)
    bbox = np.array([x1, y1, x2, y2])
    return bbox


def iou(a, b, eps=1e-7):
    """
    :param a: (N, 4)
    :param b: (M, 4)
    :param eps:
    :return: (N, M)
    """
    a = np.array(a)
    b = np.array(b)

    area1 = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    width_height = np.minimum(a[:, None, 2:], b[:, 2:]) - np.maximum(a[:, None, :2], b[:, :2])  # N, M, 2
    width_height = np.clip(width_height, a_min=0, a_max=1e7)
    area_overlap = width_height.prod(axis=2)

    iou = area_overlap / (area1[:, None] + area2 - area_overlap + eps)
    return iou


def squre_mask(xy_center, radius, out_size):
    """
    :param xy_center: (N, 2) in NDC space
    :param radius: (N, 2) in NDC space
    :param H:
    :return:
    """
    H = W = out_size
    device = xy_center.device
    N = xy_center.size(0)

    x = torch.linspace(-1, 1, W).to(device)
    y = torch.linspace(-1, 1, H).to(device)
    y, x = torch.meshgrid(y, x)  # in shape of H, W
    coord = torch.stack([x, y], dim=-1)  # (H, W, 2)
    coord = coord.unsqueeze(0).repeat(N, 1, 1, 1) # (N, H, W, 2)

    radius = radius.view(N, 1, 1, 2)
    xy_center = xy_center.view(N, 1, 1, 2)
    mask = torch.sqrt(( ((coord - xy_center)/radius) ** 2).sum(-1)) < 1
    mask = mask.float().view(N, 1, H, W).contiguous()
    return mask


def mask_to_points(masks, P):
    """
    :param masks: (N, 1, H, W)
    :param P:
    :return: (N, P, 2) in NPC space
    """
    N, _, H, W = masks.size()
    device = masks.device
    prob = masks.view(N, H*W)
    inds = torch.multinomial(prob, P, replacement=True).unsqueeze(-1)  # (N, P, 1)

    x = torch.linspace(-1, 1, W).to(device)
    y = torch.linspace(-1, 1, H).to(device)
    y, x = torch.meshgrid(y, x)  # in shape of H, W
    coord = torch.stack([x, y], dim=-1)  # (H, W, 2)
    coord = coord.unsqueeze(0).repeat(N, 1, 1, 1).view(N, H * W, 2)  # (N, H, W, 2)

    points = torch.gather(coord, 1, inds.repeat(1, 1, 2))
    return points


# ######################## Visualization code ########################
def read_mp4(filename):
    vid = imageio.get_reader(filename,  'ffmpeg')
    image_list = []
    for image in vid.iter_data():
        image_list.append(image)
    return image_list

def merge_novel_view(gif_file, t_list=None, axis=1):
    image_list = read_mp4(gif_file + '.mp4', )
    T = len(image_list)
    if t_list is None:
        t_list = [0,
            int(T // 4),
        ]
    save_list = []
    for t in t_list:
        save_list.append(image_list[t])
    save_list = np.concatenate(save_list, axis=axis)
    imageio.imwrite(gif_file + '_merge.png', save_list)


def extract_novel_view(gif_file, t_list=None):
    image_list = read_mp4(gif_file + '.mp4', )
    T = len(image_list)
    if t_list is None:
        t_list = [0,
            int(T // 3),
            int(T // 4),
            int(T // 6),
            int(2 * T // 3),
            int(3 * T // 4),
            int(5* T // 6),
        ]

    for t in t_list:
        imageio.imwrite(gif_file + '_%d.png' % t, image_list[t])



def merge_gifs(file_list, save_file, size=None, axis=1):
    vid_list = []
    for image_file in file_list:
        ext = image_file.split('.')[0]
        if ext in ['png', 'jpg', 'jpeg']:
            image = imageio.imread(image_file)
            image_list = [image]
        else:
            image_list = read_mp4(image_file)
        vid_list.append(image_list)
    max_t = len(max(vid_list, key=len))
    if size is None:
        size = max(vid_list, key=lambda x: x[0].shape[1-axis])
        size = size[0].shape[1-axis]
    scale_list = [size / x[0].shape[1-axis] for x in vid_list]
    canvas_list = []
    for t in range(max_t):
        canvas = []
        for v, vid in enumerate(vid_list):
            if t >= len(vid):
                t_id = -1
            else:
                t_id = t
            image = vid[t_id]
            f = scale_list[v]
            image = cv2.resize(image, (0, 0), fx=f, fy=f)
            canvas.append(image)
        canvas = np.concatenate(canvas, axis=axis)
        canvas_list.append(canvas)
    write_mp4(canvas_list, save_file)
    return canvas_list


def save_images(images, fname, text_list=[None], merge=1, col=8, scale=False, bg=None, mask=None, r=0.9,
                keypoint=None, color=(0, 1, 0)):
    """
    :param it:
    :param images: Tensor of (N, C, H, W)
    :param text_list: str * N
    :param name:
    :param scale: if RGB is in [-1, 1]
    :param keypoint: (N, K, 2) in scale of [-1, 1]
    :return:
    """
    if bg is not None:
        images = blend_images(images, bg, mask, r)
    if keypoint is not None:
        images = vis_j2d(images, keypoint, -1, color=color)

    merge_image = tensor_text_to_canvas(images, text_list, col=col, scale=scale)

    if fname is not None:
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        imageio.imwrite(fname + '.png', merge_image)
    return merge_image


def save_heatmap(image, fname, merge=1, col=8, scale=False):
    image = image.cpu()
    image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])[..., 0]
    print(image.min(), image.max())
    plt.close()
    plt.figure()
    plt.imshow(image, cmap='plasma', vmax=4,vmin=-1)
    plt.axis('off')
    if fname is not None:
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        plt.savefig(fname + '.png', bbox_inches='tight',pad_inches = 0)

    # if scale:
    #     N = image.size(0)
    #     min_v = torch.min(image.view(N, -1), dim=-1)[0].view(N, 1, 1, 1)
    #     max_v = torch.max(image.view(N, -1), dim=-1)[0].view(N, 1, 1, 1)
    #     image = (image - min_v) / (max_v - min_v)
    # image = image.cpu()
    # image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    # image = image.numpy().transpose([1, 2, 0])
    # image = np.clip(255 * image, 0, 255).astype(np.uint8)
    # image = cv2.applyColorMap(image, cv2.COLORMAP_JET,)
    #
    # if fname is not None:
    #     if not os.path.exists(os.path.dirname(fname)):
    #         os.makedirs(os.path.dirname(fname))
    #     imageio.imwrite(fname + '.png', image)
    # return image



def save_dxdy(dxdy, fname, scale=False):
    """
    dxdy: (N, 2, H, W)
    """
    N, C, H, W = dxdy.size()
    if scale:
        dxdy = dxdy / torch.norm(dxdy, dim=1, keepdim=True).clamp(min=1e-6)
    if C == 2:
        dxdy = torch.cat([dxdy, torch.zeros(N, 1, H, W, device=dxdy.device)], dim=1)
    save_images(dxdy, fname, scale=True)

def draw_contour(xy, width=64, color=(0, 255, 0)):
    """
    :param xy: numpy / tensor, in range of [-1, 1]
    :return:
    """
    if torch.is_tensor(xy):
        xy = xy.cpu().detach().numpy()
    K = xy.shape[0]
    xy = (xy / 2 + 0.5) * width
    xy = xy.astype(np.uint8)
    canvas = np.zeros([width, width, 3], np.uint8)
    for k in range(K):
        center = (xy[k, 0], xy[k, 1])
        canvas = cv2.circle(canvas, center, 1, color, -1)
    return canvas


def save_contour(contours, fname, ):
    """
    :param: contours: (N, P, 2)
    :return:
    """
    image_list = []
    for n in range(contours.size(0)):
        canvas = draw_contour(contours[n])
        image_list.append(canvas)
    image_list = torch.from_numpy(np.array(image_list).transpose([0, 3, 1, 2]))
    image = vutils.make_grid(image_list)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(image, 0, 255).astype(np.uint8)

    if fname is not None:
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        imageio.imwrite(fname + '.png', image)
    return image


def save_depth(images, fname, text_list=[None], merge=1, col=8, scale=False, znear=1, zfar=100):
    """
    :param it:
    :param images: Tensor of (N, C, H, W)
    :param text_list: str * N
    :param name:
    :param scale: if RGB is in [-1, 1]
    :return:
    """
    images = images.clamp(min=znear, max=zfar)
    images = 1 - (images - znear) / (zfar - znear)
    merge_image = tensor_text_to_canvas(images, text_list, col=col, scale=False)

    merge_image = cv2.applyColorMap(merge_image[..., 0], cv2.COLORMAP_JET)
    if fname is not None:
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        imageio.imwrite(fname + '.png', merge_image)
    else:
        return merge_image

def save_gif(image_list, fname, text_list=[None], merge=1, col=8, scale=False):
    """
    :param image_list: [(N, C, H, W), ] * T
    :param fname:
    :return:
    """

    def write_to_gif(gif_name, tensor_list, batch_text=[None], col=8, scale=False):
        """
        :param gif_name: without ext
        :param tensor_list: list of [(N, C, H, W) ] of len T.
        :param batch_text: T * N * str. Put it on top of
        :return:
        """
        T = len(tensor_list)
        if batch_text is None:
            batch_text = [None]
        if len(batch_text) == 1:
            batch_text = batch_text * T
        image_list = []
        for t in range(T):
            time_slices = tensor_text_to_canvas(tensor_list[t], batch_text[t], col=col,
                                                scale=scale)  # numpy (H, W, C) of uint8
            image_list.append(time_slices)
        # write_mp4(image_list, gif_name)
        write_gif(image_list, gif_name)
    # merge write
    if len(image_list) == 0:
        print('not save empty gif list')
        return
    num = image_list[0].size(0)
    if merge >= 1:
        write_to_gif(fname, image_list, text_list, col=min(col, num), scale=scale)
    if merge == 0 or merge == 2:
        for n in range(num):
            os.makedirs(fname, exist_ok=True)
            single_list = [each[n:n+1] for each in image_list]
            write_to_gif(os.path.join(fname, '%d' % n), single_list, [text_list[n]], col=1, scale=scale)


def write_gif(image_list, gif_name):
    if not os.path.exists(os.path.dirname(gif_name)):
        os.makedirs(os.path.dirname(gif_name))
        print('## Make directory: %s' % gif_name)
    imageio.mimsave(gif_name + '.gif', image_list)
    print('save to ', gif_name + '.gif')


def write_mp4(video, save_file):
    tmp_dir = save_file + '.tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    for t, image in enumerate(video):
        imageio.imwrite(osp.join(tmp_dir, '%02d.jpg' % t), image.astype(np.uint8))

    if osp.exists(save_file + '.mp4'):
        os.system('rm %s.mp4' % (save_file))
    src_list_dir = osp.join(tmp_dir, '%02d.jpg')
    cmd = 'ffmpeg -framerate 10 -i %s -c:v libx264 -pix_fmt yuv420p %s.mp4' % (src_list_dir, save_file)
    cmd += ' -hide_banner -loglevel error'
    print(cmd)
    os.system(cmd)
    cmd = 'rm -r %s' % tmp_dir
    os.system(cmd)


def blend_images(fg, bg, mask=None, r=0.9):
    fg = fg.cpu()
    bg=bg.cpu()
    if mask is None:
        image = fg.cpu() * r + bg.cpu() * (1-r)
    else:
        mask = mask.cpu().float()
        image = bg * (1 - mask) + (fg * r + bg * (1 - r)) * mask
    return image

def inverse_transform(images):
    images = (images + 1.) / 2.
    # images = images.transpose([1, 2, 0])
    return images



def tensor_text_to_canvas(image, text=None, col=8, scale=False):
    """
    :param image: Tensor / numpy in shape of (N, C, H, W)
    :param text: [str, ] * N
    :param col:
    :return: uint8 numpy of (H, W, C), in scale [0, 255]
    """
    if scale:
        image = image / 2 + 0.5
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    image = image.cpu().detach()  # N, C, H, W

    image = write_text_on_image(image.numpy(), text)  # numpy (N, C, H, W) in scale [0, 1]
    image = vutils.make_grid(torch.from_numpy(image), nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)
    return image


def write_text_on_image(images, text):
    """
    :param images: (N, C, H, W) in scale [0, 1]
    :param text: (str, ) * N
    :return: (N, C, H, W) in scale [0, 1]
    """
    if text is None or text[0] is None:
        return images

    images = np.transpose(images, [0, 2, 3, 1])
    images = np.clip(255 * images, 0, 255).astype(np.uint8)

    image_list = []
    for i in range(images.shape[0]):
        img = images[i].copy()
        img = put_multi_line(img, text[i])
        image_list.append(img)
    image_list = np.array(image_list).astype(np.float32)
    image_list = image_list.transpose([0, 3, 1, 2])
    image_list = image_list / 255
    return image_list


def put_multi_line(img, multi_line, h=15):
    for i, line in enumerate(multi_line.split('\n')):
        img = cv2.putText(img, line, (h, h * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
    return img

def vis_bbox(image_tensor, bboxes, color=(0, 1, 0), normed=True, ):
    """boxes: x1 y1 x2 y2 in [-1, 1]? """
    if torch.is_tensor(image_tensor):
        image_tensor = image_tensor.cpu().detach().numpy().transpose([0, 2, 3, 1])  # N, H, W, C
    if torch.is_tensor(bboxes):
        bboxes = bboxes.cpu().detach().numpy()
    N, H, W, _ = image_tensor.shape
    if normed:
        bboxes = (bboxes + 1) / 2 * np.array([[[W, H]]])
    image_list = []
    for n in range(N):
        image = image_tensor[n].copy()
        image = draw_bbox(image, [bboxes[n]], color)
        image_list.append(image)
    image_list = np.array(image_list)
    image_list = torch.FloatTensor(image_list.transpose(0, 3, 1, 2))
    return image_list

def vis_pts(image_tensor, pts, color=(0, 1, 0), normed=True, subset=-1):
    if torch.is_tensor(image_tensor):
        image_tensor = image_tensor.cpu().detach().numpy().transpose([0, 2, 3, 1])  # N, H, W, C
    if torch.is_tensor(pts):
        pts = pts.cpu().detach().numpy()
    N, H, W, _ = image_tensor.shape
    if normed:
        pts = (pts + 1) / 2 * np.array([[[W, H]]])
    image_list = []
    for n in range(N):
        image = image_tensor[n].copy()
        image = draw_hand(image, pts[n], subset, color)
        image_list.append(image)
    image_list = np.array(image_list)
    image_list = torch.FloatTensor(image_list.transpose(0, 3, 1, 2))
    return image_list


def vis_j2d(image_tensor, pts, j_list=[0, 8, 11, 14, 17, 20], color=(0, 1, 0), normed=True):
    """if normed, then 2D space is in [-1,1], else: [0, HorW]
    :param: image_tensor: tensor of (N, C, H, W)
    :param: pts: (N, V, 2) of (x, y) pairs
    :return: torch.cpu.tensor (N, C, H, W) RGB range(0, 1)
    """
    return vis_pts(image_tensor, pts, color, normed, j_list)

def draw_bbox(image, x1y1x2y2, color):
    for i in range(len(x1y1x2y2)):
        print('hello')
        x1, y1, x2, y2 = x1y1x2y2[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    return image


def draw_hand(image, pts, j_list, color):
    if j_list == -1:
        j_list = range(len(pts))
    for j in j_list:
        x, y = pts[j]
        cv2.circle(image, (int(x), int(y)), 2, color, -1)
    return image


def splat_with_wgt(feat, grid,  H, W):
    """
    https://github.com/JudyYe/CVP/blob/213a4abb7cb9dae04f224bc8e2a19e3ee9a23b08/cvp/layout.py#L318
    :param feat: (N, D, ...)
    :param grid: (N, ..., 2) in range [-1, 1]?
    :return: (N, D, H, W)
    """
    grid = grid.clone()
    # print('grid', grid)
    BIG_NEG = -1e+6
    
    # N, D, h, w = feat.size()
    N, D, num_pixels_src = feat.size()
    num_pixels_trg = H * W
    device = feat.device

    grid = grid / 2 + 0.5
    grid[..., 0] = grid[..., 0] * W
    grid[..., 1] = grid[..., 1] * H

    # num_pixels_src = h * w
    grid = grid - 0.5
    x = grid[..., 0].view(N, 1, num_pixels_src)
    y = grid[..., 1].view(N, 1, num_pixels_src)
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    y_max = H - 1
    x_max = W - 1

    x0_safe = torch.clamp(x0, min=0, max=x_max)
    y0_safe = torch.clamp(y0, min=0, max=y_max)
    x1_safe = torch.clamp(x1, min=0, max=x_max)
    y1_safe = torch.clamp(y1, min=0, max=y_max)

    wt_x0 = (x1 - x) * torch.eq(x0, x0_safe).to(device)
    wt_x1 = (x - x0) * torch.eq(x1, x1_safe).to(device)
    wt_y0 = (y1 - y) * torch.eq(y0, y0_safe).to(device)
    wt_y1 = (y - y0) * torch.eq(y1, y1_safe).to(device)

    wt_tl = wt_x0 * wt_y0
    wt_tr = wt_x1 * wt_y0
    wt_bl = wt_x0 * wt_y1
    wt_br = wt_x1 * wt_y1

    eps = 1e-3
    wt_tl = torch.clamp(wt_tl, min=eps)
    wt_tr = torch.clamp(wt_tr, min=eps)
    wt_bl = torch.clamp(wt_bl, min=eps)
    wt_br = torch.clamp(wt_br, min=eps)

    values_tl = (feat * wt_tl).view(N, D, num_pixels_src)  # (N, D, h, w)
    values_tr = (feat * wt_tr).view(N, D, num_pixels_src)  # (N, D, h, w)
    values_bl = (feat * wt_bl).view(N, D, num_pixels_src)  # (N, D, h, w)
    values_br = (feat * wt_br).view(N, D, num_pixels_src)  # (N, D, h, w)

    inds_tl = (x0_safe + y0_safe * W).view(N, 1, num_pixels_src).long().expand(N, D, num_pixels_src)
    inds_tr = (x1_safe + y0_safe * W).view(N, 1, num_pixels_src).long().expand(N, D, num_pixels_src)
    inds_bl = (x0_safe + y1_safe * W).view(N, 1, num_pixels_src).long().expand(N, D, num_pixels_src)
    inds_br = (x1_safe + y1_safe * W).view(N, 1, num_pixels_src).long().expand(N, D, num_pixels_src)

    init_trg_image = torch.zeros([N, D, num_pixels_trg]).to(device)
    init_trg_image = init_trg_image.scatter_add(-1, inds_tl, values_tl)
    init_trg_image = init_trg_image.scatter_add(-1, inds_tr, values_tr)
    init_trg_image = init_trg_image.scatter_add(-1, inds_bl, values_bl)
    init_trg_image = init_trg_image.scatter_add(-1, inds_br, values_br)

    # cnt weight
    wt_tl = wt_tl.view(N, 1, num_pixels_src)
    wt_tr = wt_tr.view(N, 1, num_pixels_src)
    wt_bl = wt_bl.view(N, 1, num_pixels_src)
    wt_br = wt_br.view(N, 1, num_pixels_src)

    inds_tl = (x0_safe + y0_safe * W).view(N, 1, num_pixels_src).long().expand(N, 1, num_pixels_src)
    inds_tr = (x1_safe + y0_safe * W).view(N, 1, num_pixels_src).long().expand(N, 1, num_pixels_src)
    inds_bl = (x0_safe + y1_safe * W).view(N, 1, num_pixels_src).long().expand(N, 1, num_pixels_src)
    inds_br = (x1_safe + y1_safe * W).view(N, 1, num_pixels_src).long().expand(N, 1, num_pixels_src)

    init_trg_wgt = torch.zeros([N, 1, num_pixels_trg]).to(device) + eps
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_tl, wt_tl)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_tr, wt_tr)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_bl, wt_bl)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_br, wt_br)

    init_trg_image = init_trg_image.view(N, D, H, W)
    init_trg_wgt = init_trg_wgt.view(N, 1, H, W)
    init_trg_image = init_trg_image / init_trg_wgt

    init_trg_wgt = torch.zeros([N, 1, num_pixels_trg]).to(device)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_tl, wt_tl)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_tr, wt_tr)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_bl, wt_bl)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_br, wt_br)

    init_trg_wgt = init_trg_wgt.view(-1)
    inds = torch.nonzero(init_trg_wgt == 0)
    init_trg_wgt[inds] = BIG_NEG
    init_trg_wgt = torch.clamp(init_trg_wgt, max=0)
    init_trg_wgt = init_trg_wgt.view(N, 1, H, W)

    return init_trg_image, init_trg_wgt


#  flow vis

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def dilate(image, k):
    C = image.size(1)
    kernel_tensor = torch.ones(C, C, 2*k+1, 2*k+1).to(image) # c, c, 3, 3
    # kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)
    torch_result = torch.nn.functional.conv2d(image, kernel_tensor, padding=(k, k))
    torch_result = torch_result.clamp(0, 1).bool()
    return torch_result


def blur(image, k, mask):
    C = image.size(1)
    kernel_tensor = torch.ones(C, C, 2*k+1, 2*k+1).to(image) # c, c, 3, 3
    torch_result = torch.nn.functional.conv2d(image, kernel_tensor, padding=(k, k))
    kernel_tensor = torch.ones(1, 1, 2*k+1, 2*k+1).to(image) # c, c, 3, 3
    wgt = torch.nn.functional.conv2d(mask.float(), kernel_tensor, padding=(k, k))
    non_zero = wgt.clamp(max=1)

    eps = 1e-6
    out = (non_zero) * torch_result / (wgt + eps) + \
        (1 - non_zero) * torch_result / ((2*k+1) ** 2 * C)
    return out

########################
def display_gif(filename):
    with open(filename,'rb') as f:
        display.display(display.Image(data=f.read(), format='png'))    
    # display.Image(filename="%s.png" % filename)


def save_image(image, fname):
    os.makedirs(osp.dirname(fname), exist_ok=True)
    imageio.imwrite(fname, image)


if __name__ == '__main__':
    import os.path as osp
    N, H,  W = 1, 32, 32
    P = 16**2
    device = 'cuda:0'

    masks = torch.zeros([N, 1, H, W], device=device)
    masks[:, :, H // 4:H//4 * 3, W // 4: W // 4 * 3] = 1
    points = mask_to_points(masks, P)  # (N, P, 2)
    print(points)
    canvas = torch.zeros([N, 3, H ,W], device=device)
    canvas = vis_pts(canvas, points, (0, 255, 0))
    save_dir = '/checkpoint/yufeiy2/hoi_output/vis_mask'
    save_images(canvas, osp.join(save_dir, 'canvas'))
    save_images(masks, osp.join(save_dir, 'masks'))