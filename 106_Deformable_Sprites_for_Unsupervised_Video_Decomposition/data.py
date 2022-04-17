import os
import glob
import numpy as np

import cv2
import imageio
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    BatchSampler,
    RandomSampler,
    SubsetRandomSampler,
)
import torchvision.transforms as T

import utils


def split_dataset_idcs(dset, n_val):
    """
    split the video sequence either at the front or back of the video
    (don't want to split in the middle and interrupt potential pairs)
    """
    if np.random.uniform() < 0.5:
        tr_idcs = np.arange(n_val, len(dset))
        va_idcs = np.arange(n_val)
    else:
        n_train = len(dset) - n_val
        tr_idcs = np.arange(n_train)
        va_idcs = np.arange(n_train, len(dset))
    return tr_idcs, va_idcs


def get_ordered_loader(dset, batch_size, preloaded):
    num_workers = batch_size if not preloaded else 0
    print("DATALOADER NUM WORKERS", num_workers)
    persistent_workers = True if num_workers > 0 else False
    return DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=persistent_workers,
        shuffle=False,
        persistent_workers=persistent_workers,
    )


def get_subset_loader(dset, idcs, batch_size, preloaded):
    """
    get a dataloader that samples randomly from a subset of the data with replacement
    """
    ## samples same number of elements of the dataset from the subset with replacement
    ## (so each epoch through the sampler has the same number of elements)
    subset = Subset(dset, idcs)
    sampler = RandomSampler(subset, replacement=True, num_samples=len(dset))
    num_workers = batch_size if not preloaded else 0
    print("DATALOADER NUM WORKERS", num_workers)
    persistent_workers = True if num_workers > 0 else False
    return DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=persistent_workers,
        sampler=sampler,
        persistent_workers=persistent_workers,
    )


def get_validation_loader(dset, batch_size, preloaded):
    # only return the indices with ground truth
    num_workers = batch_size if not preloaded else 0
    print("DATALOADER NUM WORKERS", num_workers)
    persistent_workers = True if num_workers > 0 else False
    if dset.has_set("gt"):
        gt_dset = dset.get_set("gt")
        val_set = Subset(dset, gt_dset.val_idcs)
    else:
        # otherwise, just return all frames in sequential order
        val_set = dset

    return DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=persistent_workers,
        shuffle=False,
        persistent_workers=persistent_workers,
    )


def get_random_ordered_batch_loader(dset, batch_size, preloaded, min_batch_size=None):
    total_size = len(dset)
    if min_batch_size is None:
        min_batch_size = batch_size // 2
    idcs = list(range(total_size - min_batch_size))
    sampler = SubsetRandomSampler(idcs)  # sample randomly without replacement
    batch_sampler = OrderedBatchSampler(sampler, total_size, batch_size)
    num_workers = batch_size if not preloaded else 0
    print("DATALOADER NUM WORKERS", num_workers)
    persistent_workers = True if num_workers > 0 else False
    return DataLoader(
        dset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        pin_memory=persistent_workers,
        persistent_workers=persistent_workers,
    )


class OrderedBatchSampler(BatchSampler):
    """
    For any base sampler, make a batch with the ordered elements after the base sampled index
    Sampler -> i -> OrderedBatchSampler -> [i, i+1, ..., i+batch_size-1]
    """

    def __init__(self, sampler, total_size, batch_size):
        super().__init__(sampler, batch_size, drop_last=False)
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self):
        """
        returns an iterator returning batch indices
        """
        for idx in self.sampler:
            n_batch = min(self.batch_size, self.total_size - idx)
            yield [idx + i for i in range(n_batch)]

    def __len__(self):
        return len(self.sampler)


def get_path_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_data_subdir(dtype, root, subd, seq=None, res="480p"):
    if dtype == "custom":
        out = get_custom_dir(root, subd, seq)
    elif dtype == "davis":
        out = get_davis_dir(root, subd, seq, res=res)
    elif dtype == "stv2":
        out = get_stv2_dir(root, subd, seq)
    elif dtype == "fbms":
        out = get_fbms_dir(root, subd, seq)
    elif dtype == "sintel":
        out = get_sintel_dir(root, subd, seq)
    else:
        raise NotImplementedError

    return out


def get_davis_dir(root, subd, seq=None, res="480p"):
    seq = seq if seq else ""
    return os.path.join(root, subd, res, seq).rstrip("/")


def get_stv2_dir(root, subd, seq=None):
    seq = seq if seq else ""
    return os.path.join(root, subd, seq).rstrip("/")


def get_fbms_dir(root, subd, seq=None):
    return os.path.join(root, seq, subd) if seq else root


def get_sintel_dir(root, subd, seq=None):
    seq = seq if seq else ""
    return os.path.join(root, subd, seq).rstrip("/")


def get_custom_dir(root, subd, seq=None):
    seq = seq if seq is not None else ""
    return os.path.join(root, subd, seq).rstrip("/")


def match_custom_seq(root, subd, seq):
    """
    convenience function for matching long sequence names
    """
    matches = glob.glob(f"{root}/{subd}/{seq}*")
    if len(matches) != 1:
        print(
            "sequence name {} has {} matches in {}/{}".format(
                seq, len(matches), root, subd
            )
        )
        raise ValueError
    match = os.path.basename(matches[0])
    print(f"found matching {match} for {seq}")
    return match


def get_data_dirs(data_type, root, seq, flow_gap, res="480p"):
    if data_type == "custom":
        subds = [
            "PNGImages",
            "raw_flows_gap{}".format(flow_gap),
            "raw_flows_gap-{}".format(flow_gap),
            "masks",
        ]
        seq = match_custom_seq(root, subds[0], seq)
    elif data_type == "fbms":
        subds = [
            "",
            "raw_flows_gap{}".format(flow_gap),
            "raw_flows_gap-{}".format(flow_gap),
            "GroundTruthClean",
        ]
    elif data_type == "davis":
        subds = [
            "JPEGImages",
            "raw_flows_gap{}".format(flow_gap),
            "raw_flows_gap-{}".format(flow_gap),
            "Annotations",
        ]
    elif data_type == "stv2":
        subds = [
            "JPEGImages",
            "raw_flows_gap{}".format(flow_gap),
            "raw_flows_gap-{}".format(flow_gap),
            "GroundTruth",
        ]
    elif data_type == "sintel":
        subds = [
            "final",
            "flow",
            "back_flow",
            "masks",
        ]

    subdirs = [get_data_subdir(data_type, root, sd, seq, res=res) for sd in subds]
    return [sd if os.path.isdir(sd) else None for sd in subdirs]


def check_dims_dsets(dsets):
    test = next(iter(dsets))
    assert all(d.height == test.height for d in dsets)
    assert all(d.width == test.width for d in dsets)
    return test.height, test.width


def check_names_dsets(dsets):
    test = next(iter(dsets))
    assert all(d.names == test.names for d in dsets)
    return test.names


class CompositeDataset(Dataset):
    def __init__(self, dsets: dict, idcs=None):
        super().__init__()

        self.height, self.width = check_dims_dsets(dsets.values())
        self.names = check_names_dsets(dsets.values())
        self.dsets = dsets

        print("DATASET LENGTHS:", {k: len(v) for k, v in self.dsets.items()})
        size = min([len(d) for d in self.dsets.values()])

        if idcs is None:
            idcs = np.arange(size)
        assert all(i < size and i >= 0 for i in idcs), "invalid indices {}".format(idcs)
        self.idcs = idcs
        self.cache = [None for _ in self.idcs]
        self.device = None

    def set_device(self, device):
        self.device = device
        print("SETTING DATASET DEVICE TO {}".format(device))

    def __len__(self):
        return len(self.idcs)

    def has_set(self, name):
        return name in self.dsets

    def get_set(self, name):
        return self.dsets[name]

    def compute_item(self, idx):
        out = {name: dset[idx] for name, dset in self.dsets.items()}
        out["idx"] = torch.tensor(idx)
        return out

    def __getitem__(self, i):
        if self.cache[i] is None:
            idx = self.idcs[i]
            self.cache[i] = self.compute_item(idx)
            if self.device is not None:
                self.cache[i] = utils.move_to(self.cache[i], self.device)
        return self.cache[i]


class RGBDataset(Dataset):
    def __init__(self, src_dir, scale=1, start=0, end=-1, ext=""):
        super().__init__()
        self.src_dir = src_dir
        files = sorted(filter(is_image, glob.glob("{}/*{}".format(src_dir, ext))))
        if len(files) < 1:
            raise NotImplementedError
        names = [get_path_name(p) for p in files]

        if end < 0:  # (-1 -> all, -2 -> all but last)
            end += len(files) + 1
        self.start = start
        self.end = end
        self.names = names[start:end]
        self.files = files[start:end]

        self.scale = scale
        print(
            "FOUND {} files in {}, using range {}-{}".format(
                len(files), src_dir, start, end
            )
        )
        test = load_img_tensor(self.files[0], scale)
        self.height, self.width = test.shape[-2:]
        print("RGB SCALE {} {}x{}".format(scale, self.width, self.height))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_img_tensor(self.files[idx], self.scale)


class MaskDataset(Dataset):
    def __init__(self, gt_dir, scale=1, rgb_dset=None):
        super().__init__()

        self.gt_dir = gt_dir

        # segtrack and fbms clean labels are organized by object number
        self.subdirs = sorted(glob.glob("{}/**/".format(gt_dir)))
        if len(self.subdirs) < 1:  # just the original input directory
            self.subdirs = [gt_dir]
        subd = self.subdirs[0]

        self.n_channels = len(self.subdirs) + 1

        # find the examples that are labeled (usually all of them, except fbms)
        if rgb_dset:
            self.names = rgb_dset.names
            self.scale, self.height, self.width = (
                rgb_dset.scale,
                rgb_dset.height,
                rgb_dset.width,
            )
        else:
            files = sorted(glob.glob("{}/*.png".format(subd)))
            self.names = [get_path_name(f) for f in files]
            print("FOUND {} files in {}".format(len(files), subd))

            self.scale = scale
            test = load_img_tensor(files[0], scale)
            self.height, self.width = test.shape[-2:]
            print("MASK SCALE {} {}x{}".format(scale, self.width, self.height))

        paths = [os.path.join(subd, "{}.png".format(name)) for name in self.names]

        self.is_valid = [os.path.isfile(path) for path in paths]
        self.val_idcs = [i for i, path in enumerate(paths) if os.path.isfile(path)]
        print("FOUND {} matching masks in {}".format(len(self.val_idcs), gt_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.is_valid[idx]:
            name = self.names[idx]
            paths = [os.path.join(sd, "{}.png".format(name)) for sd in self.subdirs]
            imgs = [load_img_tensor(path, self.scale)[:1] for path in paths]
            imgs.append(sum(imgs))
            img = torch.cat(imgs, dim=0)  # object masks and background mask
            return img, torch.tensor(1, dtype=bool)

        ## return an empty mask
        img = torch.zeros(self.n_channels, self.height, self.width, dtype=torch.float32)
        return img, torch.tensor(0, dtype=bool)


class FlowDataset(Dataset):
    """
    flow dataset, indices line up with the rgb image indices
    """

    def __init__(self, flow_dir, gap, scale=1, rgb_dset=None):
        super().__init__()
        self.src_dir = flow_dir
        self.gap = gap

        if rgb_dset:
            self.names = rgb_dset.names
            self.scale, self.height, self.width = (
                rgb_dset.scale,
                rgb_dset.height,
                rgb_dset.width,
            )
        else:
            files = sorted(glob.glob("{}/*.flo".format(flow_dir)))
            self.names = [get_path_name(f) for f in files]
            print("FOUND {} files in {}".format(len(files), flow_dir))

            self.scale = scale
            test = load_flow_tensor(files[0], scale)
            self.height, self.width = test.shape[-2:]
            print("FLOW SCALE {} {}x{}".format(scale, self.width, self.height))

        if self.gap > 0:
            idcs = 0, len(self.names) - self.gap - 1
        else:
            idcs = -self.gap, len(self.names) - 1
        self.valid_idx_range = idcs
        valid_names = self.names[idcs[0] : idcs[1] + 1]

        self.files = [os.path.join(flow_dir, "{}.flo".format(n)) for n in valid_names]
        assert all(os.path.isfile(f) for f in self.files)
        print("FOUND {} corresponding files in {}".format(len(self.files), flow_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        """
        returns bool for valid flow or not, and flow map
        """
        name = self.names[idx]
        path = os.path.join(self.src_dir, "{}.flo".format(name))
        if not os.path.isfile(path):
            return torch.tensor(False), torch.zeros(
                2, self.height, self.width, dtype=torch.float32
            )
        return torch.tensor(True), load_flow_tensor(path, self.scale, normalize=True)


class OcclusionDataset(Dataset):
    def __init__(self, fwd_dset, bck_dset):
        assert abs(fwd_dset.gap) == abs(bck_dset.gap)
        assert len(fwd_dset) == len(bck_dset)
        assert fwd_dset.height == bck_dset.height
        assert fwd_dset.width == bck_dset.width

        self.fwd = fwd_dset
        self.bck = bck_dset
        self.gap = self.fwd.gap

        self.names = fwd_dset.names
        self.height, self.width = self.fwd.height, self.fwd.width
        self.valid_fwd_range = fwd_dset.valid_idx_range
        self.valid_bck_range = bck_dset.valid_idx_range
        print("VALID FWD RANGE", self.valid_fwd_range)
        print("VALID BCK RANGE", self.valid_bck_range)

    def __len__(self):
        return len(self.fwd)

    def _check_valid(self, idx):
        fwd_min, fwd_max = self.valid_fwd_range
        bck_min, bck_max = self.valid_bck_range
        bck_idx = idx + self.gap

        inval_fwd = idx < fwd_min or idx > fwd_max  # no valid forward
        inval_bck = bck_idx < bck_min or bck_idx > bck_max  # no valid backward
        if inval_fwd or inval_bck:  # just return zeros (or ones??)
            return False
        return True

    def __getitem__(self, idx):
        """
        return occlusion map, occluding pixel locs, and number of occluded pixels
        """
        if not self._check_valid(idx):
            occ_map = torch.zeros(1, self.height, self.width, dtype=torch.bool)
            occ_locs = torch.zeros(self.height, self.width, 2, dtype=torch.float32)
            return occ_map, occ_locs, torch.tensor(0)

        bck_idx = idx + self.gap
        f_ok, fwd = self.fwd[idx]  # (2, H, W)
        b_ok, bck = self.bck[bck_idx]
        if not f_ok or not b_ok:
            occ_map = torch.zeros(1, self.height, self.width, dtype=torch.bool)
            occ_locs = torch.zeros(self.height, self.width, 2, dtype=torch.float32)
            return occ_map, occ_locs, torch.tensor(0)

        # occ_map (1, 1, H, W), occ_locs (1, H, W, 2)
        occ_map, occ_locs = utils.compute_occlusion_locs(
            fwd[None], bck[None], self.gap, ret_locs=True
        )
        return occ_map[0], occ_locs[0], occ_map.sum()


class EpipolarDataset(Dataset):
    def __init__(self, flow_dset, clip=True, reject=0.5):
        self.flow_dset = flow_dset
        self.names = flow_dset.names
        self.gap = flow_dset.gap
        self.valid_idx_range = flow_dset.valid_idx_range

        self.height, self.width = flow_dset.height, flow_dset.width
        uv = utils.get_uv_grid(self.height, self.width, align_corners=False)
        self.x1 = uv.reshape(-1, 2)  # (H*W, 2)
        self.clip = clip
        self.reject = reject
        self.cache = [None for _ in self.names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self.cache[i] is None:
            self.cache[i] = self.compute_sampson_error(i)
        return self.cache[i]

    def compute_sampson_error(self, i):
        H, W = self.height, self.width
        F = torch.zeros(3, 3, dtype=torch.float32)
        err = torch.zeros(H, W, dtype=torch.float32)

        lo, hi = self.valid_idx_range
        if i < lo or i > hi:
            return torch.tensor(False), err, F

        ok, flow = self.flow_dset[i]
        if not ok:
            return ok, err, F

        ok = torch.tensor(True)
        x1 = self.x1
        flow = flow.permute(1, 2, 0)  # (H, W, 2)
        x2 = x1 + flow.view(-1, 2)  # (H*W, 2)
        F, _ = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
        #         F, _ = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_8POINT)
        F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
        err = utils.compute_sampson_error(x1, x2, F).reshape(H, W)
        fac = (H + W) / 2
        err = err * fac ** 2
        if self.clip:  # either clip
            thresh = torch.quantile(err, 0.8)
            if thresh > self.reject:
                ok = torch.tensor(False)
            err = torch.where(err <= thresh, torch.zeros_like(err), err)

        return ok, err, F

    def save_to(self, out_dir):
        """
        save the fundamental mats and sampson errors to out_dir
        """
        print("saving to", out_dir)
        N = len(self)
        err_scales = []
        f_mats = []
        oks = []
        ims_path = os.path.join(out_dir, "s_err_ims.gif")
        ims_writer = imageio.get_writer(ims_path, format="gif")
        for i in range(N):
            ok, err, F = self[i]
            emax = err.max() + 1e-6
            ims_writer.append_data((255 * err / emax).cpu().byte().numpy())
            oks.append(ok.cpu())
            err_scales.append(emax)
            f_mats.append(F.cpu())

        ims_writer.close()

        ok_path = os.path.join(out_dir, "epi_ok.txt")
        np.savetxt(ok_path, oks)

        scale_path = os.path.join(out_dir, "s_err_scales.txt")
        np.savetxt(scale_path, err_scales)

        fmat_path = os.path.join(out_dir, "fmats_flat.txt")
        np.savetxt(fmat_path, torch.stack(f_mats, dim=0).reshape(N, -1))


def is_image(path):
    ext = os.path.splitext(path.lower())[-1]
    return ext == ".png" or ext == ".jpg" or ext == ".bmp"


def load_img_tensor(path, scale=1):
    """
    Load image, rescale to [0., 1.]
    Returns (C, H, W) float32
    """
    im = read_img(path, scale)
    tensor = torch.from_numpy(np.array(im))
    if tensor.ndim < 3:
        tensor = tensor[..., None]
    return tensor.permute(2, 0, 1) / 255.0


def read_img(path, scale=1):
    im = Image.open(path)
    if scale != 1:
        W, H = im.size
        w, h = int(scale * W), int(scale * H)
        im = im.resize((w, h), Image.ANTIALIAS)
    return im


def load_flow_tensor(path, scale=1, normalize=True, align_corners=True):
    """
    Load flow, scale the pixel values according to the resized scale.
    If normalize is true, return rescaled in normalized pixel coordinates
    where pixel coordinates are in range [-1, 1].
    NOTE: RAFT USES ALIGN_CORNERS=TRUE SO WE NEED TO ACCOUNT FOR THIS
    Returns (2, H, W) float32
    """
    flow = read_flo(path).astype(np.float32)
    H, W, _ = flow.shape
    u, v = flow[..., 0], flow[..., 1]
    if normalize:
        if align_corners:
            u = 2.0 * u / (W - 1)
            v = 2.0 * v / (H - 1)
        else:
            u = 2.0 * u / W
            v = 2.0 * v / H
    else:
        u = scale * u
        v = scale * v

    if scale != 1:
        h, w = int(scale * H), int(scale * W)
        u = Image.fromarray(u).resize((w, h), Image.ANTIALIAS)
        v = Image.fromarray(v).resize((w, h), Image.ANTIALIAS)
        u, v = np.array(u), np.array(v)
    return torch.from_numpy(np.stack([u, v], axis=0))


TAG_FLOAT = 202021.25


def read_flo(filename):
    """
    returns (H, W, 2) numpy array flow field
    """
    assert type(filename) is str, "filename is not str %r" % str(filename)
    assert os.path.isfile(filename) is True, "file does not exist %r" % str(filename)
    assert filename[-4:] == ".flo", "file ending is not .flo %r" % filename[-4:]
    f = open(filename, "rb")
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, (
        "Flow number %r incorrect. Invalid .flo file" % flo_number
    )
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def write_flo(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(TAG_FLOAT, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("write_flo: flow must have two bands!")
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, "wb") as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())
