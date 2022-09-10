from PIL import Image
import sys
import numpy as np
import random

from .vid import VIDDataset, make_vid_transforms
# from mega_core.config import cfg
import datasets.transforms as T

class VIDMULTIDataset(VIDDataset):
    def __init__(self, image_set, img_dir, anno_path, img_index, transforms, is_train=True, cfg=None):
        super().__init__(image_set, img_dir, anno_path, img_index, transforms, is_train=is_train, cfg=cfg)
        self.cfg = cfg
        self.max_offset = 12
        self.min_offset = -12
        self.ref_num_local = 2

        self.test_with_one_img = False
        self.test_ref_nums = 2
        self.test_max_offset = 12
        self.test_min_offset = -12

        if cfg is not None:

            self.test_with_one_img = cfg.TEST.test_with_one_img
            self.test_ref_nums = cfg.TEST.test_ref_nums
            self.test_max_offset = cfg.TEST.test_max_offset
            self.test_min_offset = cfg.TEST.test_min_offset


        if cfg is not None:
            self.max_offset = cfg.DATASET.max_offset
            self.min_offset = cfg.DATASET.min_offset
            self.ref_num_local = cfg.DATASET.ref_num_local

        # if not self.is_train:
        #     self.start_index = []
        #     self.start_id = []
        #     # if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
        #     #     self.shuffled_index = {}
        #     for id, image_index in enumerate(self.image_set_index):
        #         frame_id = int(image_index.split("/")[-1])
        #         if frame_id == 0:
        #             self.start_index.append(id)
        #             # if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
        #             #     shuffled_index = np.arange(self.frame_seg_len[id])
        #             #     if cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
        #             #         np.random.shuffle(shuffled_index)
        #             #     self.shuffled_index[str(id)] = shuffled_index
        #
        #             self.start_id.append(id)
        #         else:
        #             self.start_id.append(self.start_index[-1])

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # if a video dataset
        img_refs_l = []
        # img_refs_m = []
        # img_refs_g = []
        if hasattr(self, "pattern"):
            offsets = np.random.choice(self.max_offset - self.min_offset + 1,
                                       self.ref_num_local, replace=False) + self.min_offset
            for i in range(len(offsets)):
                ref_id = min(max(self.frame_seg_id[idx] + offsets[i], 0), self.frame_seg_len[idx] - 1)
                ref_filename = self.pattern[idx] % ref_id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs_l.append(img_ref)

            # # memory frames
            # if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
            #     ref_id_center = max(self.frame_seg_id[idx] - cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL, 0)
            #     offsets = np.random.choice(cfg.MODEL.VID.MEGA.MAX_OFFSET - cfg.MODEL.VID.MEGA.MIN_OFFSET + 1,
            #                                cfg.MODEL.VID.MEGA.REF_NUM_MEM, replace=False) + cfg.MODEL.VID.MEGA.MIN_OFFSET
            #     for i in range(len(offsets)):
            #         ref_id = min(max(ref_id_center + offsets[i], 0), self.frame_seg_len[idx] - 1)
            #         ref_filename = self.pattern[idx] % ref_id
            #         img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
            #         img_refs_m.append(img_ref)
            #
            # # global frames
            # if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            #     ref_ids = np.random.choice(self.frame_seg_len[idx], cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL, replace=False)
            #     for ref_id in ref_ids:
            #         ref_filename = self.pattern[idx] % ref_id
            #         img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
            #         img_refs_g.append(img_ref)
        else:
            for i in range(self.ref_num_local):
                img_refs_l.append(img.copy())
            # if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
            #     for i in range(cfg.MODEL.VID.MEGA.REF_NUM_MEM):
            #         img_refs_m.append(img.copy())
            # if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            #     for i in range(cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL):
            #         img_refs_g.append(img.copy())

        target = self.get_groundtruth(idx)
        # target = target.clip_to_image(remove_empty=True)
        # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        # scales1 = [400, 500, 600]
        # p_dict = {"resize0": random.choice(scales),
        #           "resize1": random.choice(scales1),
        #           "resize2": random.choice(scales),
        #           "hflip": random.random(),
        #           # "size_crop": [random.randint(min(img.height, 384), min(img.height, 600)),
        #           #               random.randint(min(img.width, 384), min(img.width, 600))],
        #           "select": random.random()}
        p_dict = None
        #


        if self.transforms is not None:
            # print("p_dict_define", p_dict)
            img, target = self.transforms(img, target, p_dict)
            for i in range(len(img_refs_l)):
                # print("p_dict_define_ref", p_dict)
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None, p_dict)
            # for i in range(len(img_refs_m)):
            #     img_refs_m[i], _ = self.transforms(img_refs_m[i], None)
            # for i in range(len(img_refs_g)):
            #     img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img  # to make a list
        images["ref_l"] = img_refs_l
        # images["ref_m"] = img_refs_m
        # images["ref_g"] = img_refs_g

        return images, target

    def _get_test(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(filename.split("/")[-1])
        frame_category = 0
        if frame_id != 0:
            frame_category = 1

        if self.test_with_one_img:

            img_refs_l = []
            # reading other images of the queue (not necessary to be the last one, but last one here)
            ref_id = min(self.frame_seg_len[idx] - 1, frame_id +self.max_offset)
            ref_filename = self.pattern[idx] % ref_id
            img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
            img_refs_l.append(img_ref)

        else:
            img_refs_l = self.get_ref_imgs(idx)

        img_refs_g = []
        # if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
        #     size = cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == 0 else 1
        #     shuffled_index = self.shuffled_index[str(self.start_id[idx])]
        #     for id in range(size):
        #         filename = self.pattern[idx] % shuffled_index[
        #             (idx - self.start_id[idx] + cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) % self.frame_seg_len[idx]]
        #         img_g = Image.open(self._img_dir % filename).convert("RGB")
        #         img_refs_g.append(img)

        target = self.get_groundtruth(idx)
        # target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l
        # images["ref_g"] = img_refs_g
        # images["frame_category"] = frame_category
        # images["seg_len"] = self.frame_seg_len[idx]
        # images["pattern"] = self.pattern[idx]
        # images["img_dir"] = self._img_dir
        # images["transforms"] = self.transforms

        return images, target

    def get_ref_imgs(self, idx):
        filename = self.image_set_index[idx]
        # img = Image.open(self._img_dir % filename).convert("RGB")

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(filename.split("/")[-1])
        ref_id_list = []
        ref_start_id = frame_id + self.test_min_offset
        ref_end_id = frame_id + self.test_max_offset

        interval = (ref_end_id - ref_start_id) // (self.test_ref_nums - 1)

        for i in range(ref_start_id, ref_end_id + 1, interval):
            # print(i)
            ref_id_list.append(min(max(0, i), self.frame_seg_len[idx] - 1))

        # for i in range(ref_start_id, ref_end_id + 1):
        #     ref_id_list.append(min(max(0, i), self.frame_seg_len[idx] - 1))

        img_refs_l = []

        for ref_id in ref_id_list:
            # print(ref_id)
            ref_filename = self.pattern[idx] % ref_id
            img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
            img_refs_l.append(img_ref)

        return img_refs_l



def build_vitmulti_transforms(is_train):
    # todo fixme add data augmantation
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # if is_train:
    #     return T.Compose([
    #         T.RandomHorizontalFlip(),
    #         T.RandomSelect(
    #             T.RandomResize(scales, max_size=1333, id=0),
    #             T.Compose([
    #                 T.RandomResize([400, 500, 600], id=1),
    #                 # T.RandomSizeCrop(384, 600),  # todo if cropping is neccessary? 'cause it may lead to no bbox in the image. In current version, cur and ref images are cropped using different parameters.
    #                 T.RandomResize(scales, max_size=1333, id=2),
    #             ])
    #         ),
    #         normalize,
    #     ])

    # if image_set == 'val':
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    # return transform





def build_vidmulti(image_set, cfg, split_name='VID_train_15frames', transforms=build_vitmulti_transforms(True)):

    is_train = (image_set == 'train')
    if is_train:
        dataset = VIDMULTIDataset(
        image_set = split_name,
        img_dir = cfg.DATASET.img_dir[0].format(split_name.split('_')[0]),
        anno_path = cfg.DATASET.anno_path[0].format(split_name.split('_')[0]),
        img_index = cfg.DATASET.img_index[0].format(split_name),
        transforms=transforms,
        is_train=is_train,
        cfg=cfg
        )
    else:
        dataset = VIDMULTIDataset(
        image_set = split_name,
        img_dir = cfg.DATASET.img_dir[0].format(split_name.split('_')[0]),
        anno_path = cfg.DATASET.anno_path[0].format(split_name.split('_')[0]),
        img_index = cfg.DATASET.img_index[0].format(split_name),
        transforms=transforms,
        is_train=is_train,
        cfg=cfg
        )

    return dataset

def build_detmulti(image_set, cfg):

    is_train = (image_set == 'train')
    assert is_train is True  # no validation dataset
    dataset = VIDMULTIDataset(
    image_set = "DET_train_30classes",
    img_dir = "/dataset/public/ilsvrc2015/Data/DET",
    anno_path = "/dataset/public/ilsvrc2015/Annotations/DET",
    img_index = "/data1/wanghan20/Prj/VODETR/datasets/split_file/DET_train_30classes.txt",
    transforms=build_vitmulti_transforms(True),
    is_train=is_train,
    cfg=cfg
    )
    return dataset