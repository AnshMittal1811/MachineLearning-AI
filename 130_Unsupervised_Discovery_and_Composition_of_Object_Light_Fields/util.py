import matplotlib.colors as colors
import diff_operators
import torch.nn.functional as F
import geometry
import os, struct, math
import numpy as np
import torch
from glob import glob
import collections
import cv2

from pdb import set_trace as pdb
def analytical_depth(phis,coords,world2model,cam2world):
    with torch.enable_grad():
        depth_infos = [light_field_depth_map(coord,cam2world,phi)
                for phi,coord in zip(phis,coords)]

    mod_xyz    = torch.stack([di["points"] for di in depth_infos])
    mod_xyzh   = torch.cat((mod_xyz,torch.ones_like(mod_xyz[...,:1])),-1)
    world_xyz  = (world2model.inverse()@mod_xyzh.permute(0,1,3,2)
                                                      ).permute(0,1,3,2)[...,:3]
    valid_mask = torch.stack([di["mask"].view(di["depth"].shape) for di in depth_infos])
    est_depth = (1e-4+(geometry.get_ray_origin(cam2world)[None,:,None]-world_xyz
                    ).square().sum(-1,True)).sqrt()*valid_mask
    return est_depth,valid_mask

def get_context_cam(input):
    query_dict = input['context']
    pose = flatten_first_two(query_dict["cam2world"])
    intrinsics = flatten_first_two(query_dict["intrinsics"])
    uv = flatten_first_two(query_dict["uv"].float())
    return pose, intrinsics, uv

def get_query_cam(input):
    query_dict = input['query']
    pose = flatten_first_two(query_dict["cam2world"])
    intrinsics = flatten_first_two(query_dict["intrinsics"])
    uv = flatten_first_two(query_dict["uv"].float())
    return pose, intrinsics, uv

def get_latest_file(root_dir):
    """Returns path to latest file in a directory."""
    list_of_files = glob.glob(os.path.join(root_dir, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def parse_comma_separated_integers(string):
    return list(map(int, string.split(',')))


def scale_img(img, type):
    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img


def convert_image(img, type):
    '''Expects single batch dimesion'''
    img = img.squeeze(0)

    if not 'normal' in type:
        img = detach_all(lin2img(img, mode='np'))

    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img


def write_img(img, path):
    print(img.shape)
    img = lin2img(img)[0]

    img += 1
    img /= 2.
    img = img.detach().cpu().numpy()
    img = np.clip(img, 0., 1.)
    img *= 255

    cv2.imwrite(path, img.astype(np.uint8))


def in_out_to_param_count(in_out_tuples):
    return np.sum([np.prod(in_out) + in_out[-1] for in_out in in_out_tuples])


def flatten_first_two(tensor):
    b, s, *rest = tensor.shape
    return tensor.view(b*s, *rest)


def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        line1 = list(map(float, file.readline().split()))
        if line1[-1]==0:
            f, cx, cy, _ = line1
            fy=f
        else:
            f, fy, cx, cy, = line1
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -fy

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses

def num_divisible_by_2(number):
    i = 0
    while not number%2:
        number = number // 2
        i += 1

    return i

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pose(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d"%params)

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def encoder_load(model, path):
    if os.path.isdir(path):
        checkpoint_path = sorted(glob(os.path.join(path, "*.pth")))[-1]
    else:
        checkpoint_path = path

    whole_dict = torch.load(checkpoint_path)

    state = model.state_dict()
    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in whole_dict["model"].items() if "encoder" in k}
    # 2. overwrite entries in the existing state dict
    state.update(filtered_dict)
    # 3. load the new state dict
    model.load_state_dict(state)

def custom_load(model, path, discriminator=None, 
                            gen_optimizer=None,disc_optimizer=None):
    checkpoint_path = path

    #state = model.state_dict()
    whole_dict = torch.load(checkpoint_path)
    model.load_state_dict(whole_dict["model"])
    #state.update(whole_dict["model"])
    #model.load_state_dict(state)
    """
    if type(path)==list:
        for checkpoint_path in path:
            whole_dict = torch.load(checkpoint_path)
            # 1. filter out unnecessary keys
            filtered_dict = {k: v for k, v in whole_dict["model"].items() if k in state}
            # 2. overwrite entries in the existing state dict
            state.update(filtered_dict)
            # 3. load the new state dict
            model.load_state_dict(state)
    else:
        whole_dict = torch.load(checkpoint_path)
        # 1. filter out unnecessary keys
        filtered_dict = {k: v for k, v in whole_dict["model"].items() if k in state}
        # 2. overwrite entries in the existing state dict
        state.update(filtered_dict)
        # 3. load the new state dict
        model.load_state_dict(state)
    """


    if discriminator is not None and "disc" in whole_dict:
        discriminator.load_state_dict(whole_dict['disc'])
    else:
        print("no disc")

    if gen_optimizer is not None:
        try:
            if "gen_optimizer" in whole_dict:
                gen_optimizer.load_state_dict(whole_dict["gen_optimizer"])
            else:
                gen_optimizer.load_state_dict(whole_dict["coarse_optimizer"])
        except:
            print("optimizer load failed")
    if disc_optimizer is not None and "disc_optimizer" in whole_dict:
        disc_optimizer.load_state_dict(whole_dict["disc_optimizer"])
    else:
        print("no disc optim")


def custom_save(model, path, discriminator=None, gen_optimizer=None,disc_optimizer=None):
    whole_dict = {'model':model.state_dict()}
    whole_dict.update({'gen_optimizer':gen_optimizer.state_dict()})
    if discriminator is not None:
        whole_dict.update({'disc_optimizer':disc_optimizer.state_dict()})
        whole_dict.update({'disc':discriminator.state_dict()})
    torch.save(whole_dict, path)

def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(dict_to_gpu(k) for k in ob)
    elif isinstance(ob, list):
        return [dict_to_gpu(k) for k in ob]
    else:
        try:
            return ob.cuda()
        except:
            return ob


def add_batch_dim_to_dict(ob):
    if isinstance(ob, collections.Mapping):
        return {k: add_batch_dim_to_dict(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(add_batch_dim_to_dict(k) for k in ob)
    elif isinstance(ob, list):
        return [add_batch_dim_to_dict(k) for k in ob]
    else:
        try:
            return ob[None, ...]
        except:
            return ob


def detach_all(tensor):
    return tensor.detach().cpu().numpy()


def lin2img(tensor, image_resolution=None, mode='torch'):
    if len(tensor.shape) == 3:
        batch_size, num_samples, channels = tensor.shape
    elif len(tensor.shape) == 2:
        num_samples, channels = tensor.shape

    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    if len(tensor.shape)==3:
        if mode == 'torch':
            tensor = tensor.permute(0, 2, 1).view(batch_size, channels, height, width)
        elif mode == 'np':
            tensor = tensor.view(batch_size, height, width, channels)
    elif len(tensor.shape) == 2:
        if mode == 'torch':
            tensor = tensor.permute(1, 0).view(channels, height, width)
        elif mode == 'np':
            tensor = tensor.view(height, width, channels)

    return tensor


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def parse_intrinsics_hdf5(raw_data, trgt_sidelength=None, invert_y=False):
    s = raw_data[...].tostring()
    s = s.decode('utf-8')

    lines = s.split('\n')

    f, cx, cy, _ = map(float, lines[0].split())
    grid_barycenter = torch.Tensor(list(map(float, lines[1].split())))
    height, width = map(float, lines[3].split())

    try:
        world2cam_poses = int(lines[4])
    except ValueError:
        world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, world2cam_poses


def get_mgrid(sidelen, dim=2, flatten=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.from_numpy(pixel_coords)

    if flatten:
        pixel_coords = pixel_coords.view(-1, dim)
    return pixel_coords


def Nv(st, x, x_prim, d):
    a = x + st[..., :1] * d
    b = x_prim + st[..., 1:] * d
    v_dir = b - a
    v_mom = torch.cross(a, b, dim=-1)
    return torch.cat((v_dir, v_mom), dim=-1) / (v_dir.norm(dim=-1, keepdim=True) + 1e-9)


def horizontal_plucker_slices_thirds(cam2world, light_field_fn, sl=256):
    x = geometry.get_ray_origin(cam2world)[:1]
    right = cam2world[:1, ..., :3, 0]

    slices = []
    sts = []
    s = torch.linspace(-0.5, 0.5, 128)
    t = torch.linspace(-0.5, 0.5, 1024)
    st = torch.stack(torch.meshgrid(s, t), dim=-1).cuda().requires_grad_(True)
    for j, third in enumerate([0.0]):
    # for j, third in enumerate([-0.2, 0.05, 0.2]):
        at = cam2world[:1, ..., :3, 2] + cam2world[:1, ..., :3, 1] * third

        x_prim = x + at
        with torch.enable_grad():
            # st = get_mgrid(sl).cuda().flatten(1, 2).requires_grad_(True) * 0.5
            v_norm = Nv(st, x, x_prim, right)
            reg_model_out = light_field_fn(v_norm)
            slices.append(reg_model_out)
            sts.append(st)

    return {'slices':slices, 'st':sts, 'coords':v_norm}


def lumigraph_slice(cam2world, intrinsics, uv, light_field_fn, sl, row, parallax=0.2):
    uv_img = lin2img(uv[:1], mode='np')
    uv_slice = uv_img[:, row]
    # unproject points
    lift = geometry.lift(uv_slice[..., 0], uv_slice[..., 1], torch.ones_like(uv_slice[..., 0]), intrinsics)

    x = geometry.get_ray_origin(cam2world)[:1]
    right = cam2world[:1, ..., :3, 0]
    at = torch.einsum('...ij,...j', cam2world[0, :, :3, :3], lift[:, lift.shape[1]//2])
    at = F.normalize(at, dim=-1)

    s = torch.linspace(0, parallax, sl).cuda()
    t = torch.nn.Upsample(size=sl, mode='linear', align_corners=True)(lift.permute(0, 2, 1)[:, :1])[0,0]

    x_prim = x + at
    with torch.enable_grad():
        st = torch.stack(torch.meshgrid(s, t), dim=-1).cuda()
        st[..., 1] += torch.linspace(0., parallax, sl)[:, None].cuda()
        st = st.requires_grad_(True)
        v_norm = Nv(st, x, x_prim, right)
        reg_model_out = light_field_fn(v_norm)

    return {'slice':reg_model_out, 'st':st}


def vertical_plucker_slices_thirds(cam2world, light_field_fn, sl=256):
    x = geometry.get_ray_origin(cam2world)[:1]
    right = cam2world[:1, ..., :3, 0]
    down = cam2world[:1, ..., :3, 1]

    slices = []
    s = torch.linspace(-0.5, 0.5, 128)
    t = torch.linspace(-0.5, 0.5, 1024)
    st = torch.stack(torch.meshgrid(s, t), dim=-1).cuda().requires_grad_(True)
    for j, third in enumerate([0.]):
    # for j, third in enumerate([-0.15, 0., 0.15]):
        at = cam2world[:1, ..., :3, 2] + right * third

        x_prim = x + at
        with torch.enable_grad():
            # st = get_mgrid(sl).cuda().flatten(1, 2).requires_grad_(True) * 0.5
            v_norm = Nv(st, x, x_prim, down)
            reg_model_out = light_field_fn(v_norm)
            slices.append(reg_model_out)

    return {'slices':slices, 'st':st, 'coords':v_norm}


def get_view_grid(cam2world, grid_sl, offset=1):
    right = cam2world[:1, ..., :3, 0]
    down = cam2world[:1, ..., :3, 1]

    view_grid = []
    for row in np.linspace(1, -1, grid_sl):
        row_list = []
        for col in np.linspace(1, -1, grid_sl):
            new_cam2world = cam2world.clone()
            new_cam2world[..., :3, 3] += row * offset * down + col * offset * right
            row_list.append(new_cam2world)
        view_grid.append(row_list)
    return view_grid


def canonical_plucker_slice(cam2world, light_field_fn, sl=256):
    x = geometry.get_ray_origin(cam2world)[:1]
    right = cam2world[:1, ..., :3, 0]
    at = cam2world[:1, ..., :3, 2]

    x_prim = x + at
    with torch.enable_grad():
        st = get_mgrid(sl).cuda().flatten(1, 2).requires_grad_(True) * 0.5
        v_norm = Nv(st, x, x_prim, right)
        reg_model_out = light_field_fn(v_norm)

    return {'slice':reg_model_out, 'st':st, 'coords':v_norm}


def plucker_slice(ray_origin, right, at, light_field_fn, sl=256):
    plucker = geometry.plucker_embedding(cam2world, uv, intrinsics)
    right = cam2world[:1, ..., :3, 0]
    at = cam2world[:1, ..., :3, 2]

    x = geometry.get_ray_origin(cam2world)[:1]

    intersections = geometry.lift(uv[...,0], uv[...,1], torch.ones_like(uv[...,0]), intrinsics=intrinsics)
    s = intersections[0, ..., 0]
    t = torch.linspace(-1, 1, s.shape[0]).cuda()

    x_prim = x + at
    with torch.enable_grad():
        st = torch.stack(torch.meshgrid(s, t), dim=-1).requires_grad_(True).cuda()

        a = x + plucker[..., :3] + st[..., :1] * right
        b = x_prim + st[..., 1:] * right
        v_dir = b - a
        v_mom = torch.cross(a, b, dim=-1)
        v_norm = torch.cat((v_dir, v_mom), dim=-1) / (v_dir.norm(dim=-1, keepdim=True) + 1e-9)
        reg_model_out = light_field_fn(v_norm)

    return {'slice':reg_model_out, 'st':st, 'coords':v_norm}


def get_random_slices(light_field_fn, k=10, sl=128):
    x = torch.zeros((k, 1, 3)).cuda()
    x_prim = torch.randn_like(x).cuda()
    x_prim = F.normalize(x_prim, dim=-1)

    d = torch.normal(torch.zeros_like(x), torch.ones_like(x)).cuda()
    d = F.normalize(d, dim=-1)

    with torch.enable_grad():
        st = get_mgrid(sl).cuda().flatten(1, 2).requires_grad_(True)
        coords = Nv(st, x, x_prim, d)
        c = light_field_fn(coords)

    return {'slice':c, 'st':st, 'coords':coords}


def light_field_point_cloud(light_field_fn, num_samples=64**2, outlier_rejection=True):
    dirs = torch.normal(torch.zeros(1, num_samples, 3), torch.ones(1, num_samples, 3)).cuda()
    dirs = F.normalize(dirs, dim=-1)

    x = (torch.rand_like(dirs) - 0.5) * 2

    D = 1
    x_prim = x + D * dirs

    st = torch.zeros(1, num_samples, 2).requires_grad_(True).cuda()
    max_norm_dcdst = torch.ones_like(st) * 0
    dcdsts = []
    for i in range(5):
        d_prim = torch.normal(torch.zeros(1, num_samples, 3), torch.ones(1, num_samples, 3)).cuda()
        # d_prim = F.normalize(torch.cross(d_prim, dirs, dim=-1))
        # d_prim += torch.normal(torch.zeros(1, num_samples, 3), torch.ones(1, num_samples, 3)).cuda() * 1e-3
        d_prim = F.normalize(d_prim, dim=-1)

        a = x + st[..., :1] * d_prim
        b = x_prim + st[..., 1:] * d_prim
        v_dir = b - a
        v_mom = torch.cross(a, b, dim=-1)
        v_norm = torch.cat((v_dir, v_mom), dim=-1) / v_dir.norm(dim=-1, keepdim=True)

        with torch.enable_grad():
            c = light_field_fn(v_norm)
            dcdst = diff_operators.gradient(c, st)
            dcdsts.append(dcdst)
            criterion = max_norm_dcdst.norm(dim=-1, keepdim=True)<dcdst.norm(dim=-1, keepdim=True)
            # dir_dot = torch.abs(torch.einsum('...j,...j', d_prim, dirs))[..., None]
            # criterion = torch.logical_and(criterion, dir_dot<0.1)
            max_norm_dcdst = torch.where(criterion, dcdst, max_norm_dcdst)

    dcdsts = torch.stack(dcdsts, dim=0)
    dcdt = dcdsts[..., 1:]
    dcds = dcdsts[..., :1]

    d = D * dcdt / (dcds + dcdt)
    mask = d.std(dim=0) > 1e-2
    d = d.mean(0)
    d[mask] = 0.
    d[max_norm_dcdst.norm(dim=-1)<1] = 0.

    # if outlier_rejection:

    return {'depth':d, 'points':x + d * dirs, 'colors':c}


def get_pencil_dirs(plucker_coords, cam2world, light_field_fn):
    x = geometry.get_ray_origin(cam2world)
    at = cam2world[..., :3, 2]
    right = cam2world[..., :3, 0]
    x_prim = x + at

    st = torch.zeros_like(plucker_coords[..., :2]).requires_grad_(True).to(plucker_coords.device)
    # d_prim = torch.normal(torch.zeros_like(plucker_coords[..., :3]), torch.ones_like(plucker_coords[..., :3])).to(plucker_coords.device)
    # d_prim = F.normalize(d_prim, dim=-1)
    # d_prim = torch.normal(torch.zeros(1, 1, 3), torch.ones(1, 1, 3)).to(plucker_coords.device)
    # d_prim = F.normalize(d_prim, dim=-1)
    d_prim = right

    with torch.enable_grad():
        c = light_field_fn(Nv(st, x, x_prim, d_prim))
        dcdst = diff_operators.gradient(c, st)

    confidence = dcdst.norm(dim=-1, keepdim=True)

    dcdst = F.normalize(dcdst, dim=-1)
    J = torch.Tensor([[0, -1], [1, 0.]]).cuda()
    rot_grad = torch.einsum('ij,bcj->bci', J, dcdst)

    dcdt = dcdst[..., 1:]
    dcds = dcdst[..., :1]

    def pencil(a):
        return light_field_fn(Nv(st+a*rot_grad, x, x_prim, d_prim))

    return {'confidence':confidence, 'pencil_dir':rot_grad, 'pencil_fn':pencil}


def get_canonical_pencil_dirs(plucker_coords, light_field_fn):
    x = geometry.get_ray_origin(cam2world)
    right = cam2world[..., :3, 0]
    at = cam2world[..., :3, 2]

    x_prim = x + at
    st = torch.zeros_like(plucker_coords[..., :2]).requires_grad_(True).to(plucker_coords.device)
    # d_prim = torch.normal(torch.zeros_like(plucker_coords[..., :3]), torch.ones_like(plucker_coords[..., :3])).to(plucker_coords.device)
    # d_prim = F.normalize(d_prim, dim=-1)

    with torch.enable_grad():
        c = light_field_fn(Nv(st, x, x_prim, right))
        dcdst = diff_operators.gradient(c, st)

    J = torch.Tensor([[0, -1], [1, 0.]]).cuda()
    rot_grad = torch.einsum('ij,bcj->bci', J, dcdst)

    dcdt = dcdst[..., 1:]
    dcds = dcdst[..., :1]

    return {'confidence':torch.abs(dcds + dcdt), 'pencil_dir':rot_grad}


def depth_map(query):
    light_field_fn = model.get_light_field_function(query['z'])

    plucker_coords = geometry.plucker_embedding(cam2world, uv, intrinsics)
    return light_field_depth_map(plucker_coords, cam2world, light_field_fn)

def light_field_depth_map(plucker_coords, cam2world, light_field_fn,niter=4):
    x = geometry.get_ray_origin(cam2world)[:,None]
    D = 1
    x_prim = x + D * plucker_coords[..., :3]

    d_prim = torch.normal(torch.zeros_like(plucker_coords[..., :3]),
          torch.ones_like(plucker_coords[..., :3])).to( plucker_coords.device)
    d_prim = F.normalize(d_prim, dim=-1)

    dcdsts = []
    for i in range(niter):
        st = ((torch.rand_like(plucker_coords[..., :2]) - 0.5) * 1e-2).requires_grad_(True).to(plucker_coords.device)
        a = x + st[..., :1] * d_prim
        b = x_prim + st[..., 1:] * d_prim

        v_dir = b - a
        v_mom = torch.cross(a, b, dim=-1)
        v_norm = torch.cat((v_dir, v_mom), dim=-1) / v_dir.norm(dim=-1, keepdim=True)

        with torch.enable_grad():
            c = light_field_fn(v_norm)
            dcdst = diff_operators.gradient(c, st, create_graph=False)
            dcdsts.append(dcdst)
            del dcdst
            del c

    dcdsts = torch.stack(dcdsts, dim=0)

    dcdt = dcdsts[0, ..., 1:]
    dcds = dcdsts[0, ..., :1]

    all_depth_estimates = D * dcdsts[..., 1:] / (dcdsts.sum(dim=-1, keepdim=True))
    all_depth_estimates[torch.abs(dcdsts.sum(dim=-1)) < 5] = 0
    all_depth_estimates[all_depth_estimates<0] = 0.

    depth_var = torch.std(all_depth_estimates, dim=0, keepdim=True)

    d = D * dcdt / (dcds + dcdt)
    invalid = (
               (torch.abs(dcds + dcdt) < 5).flatten()|
               (d<0).flatten()|
               (depth_var[0, ..., 0] > 0.01).flatten()
               )
    d[invalid.view(d.shape)] = 0.
    return {'depth':d, 'points':x + d * plucker_coords[..., :3],"mask":~invalid}

def assemble_model_input(context, query, gpu=True):
    context['mask'] = torch.Tensor([1.])
    query['mask'] = torch.Tensor([1.])

    context = add_batch_dim_to_dict(context)
    context = add_batch_dim_to_dict(context)

    query = add_batch_dim_to_dict(query)
    query = add_batch_dim_to_dict(query)

    model_input = {'context': context, 'query': query, 'post_input': query}

    if gpu:
        model_input = dict_to_gpu(model_input)
    return model_input


def grads2img(mG):
    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return mGrgb
