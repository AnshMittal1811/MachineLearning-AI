import torch
import torch.nn.functional as F
from .net import PatchmatchNet
from copy import deepcopy
from .util import tocuda, tensor2numpy
import math


class Patch_Depth_ES(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # 3D Points transformer
        if opt.H is None:
            self.H = opt.W
        else:
            self.H = opt.H
        self.W = opt.W

        self.scale_factor = opt.scale_factor
        self.model = PatchmatchNet()
        self.load_state()


    def load_state(self, path=None):
        """
        Load pretrained model and freeze all the weights.
        """
        if path == None:

            path = "./models/depth_es/pretrained_depth_model.ckpt"

        state_dict = torch.load(path)
        if  self.opt.pretrained_MVS:
            self.model.load_state_dict(state_dict)
        if not self.opt.learnable_mvs:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        print("MVS model has been loaded")

    def image_to_stage(self, image):
        with torch.no_grad():
            H, W = image.shape[-2:]

            output_image = {
                "stage_3": F.interpolate(image, scale_factor=0.125) * 0.5 + 0.5,
                "stage_2": F.interpolate(image, scale_factor=0.25) * 0.5 + 0.5,
                "stage_1": F.interpolate(image, scale_factor=0.5) * 0.5 + 0.5,
                "stage_0": image * 0.5 + 0.5
            }
            return output_image

    def prepare_project(self, K, input_RTs):
        # we first rescale back the K and input_RTs

        rescale_K = deepcopy(K)

        rescale_input_RTs = []
        for input_RT in input_RTs:
            rescale_input_RT = deepcopy(input_RT)
            rescale_input_RT[:, 0:3, 3] = rescale_input_RT[:, 0:3, 3] * self.scale_factor
            rescale_input_RTs.append(rescale_input_RT)

        proj_matrices_0 = [torch.bmm(rescale_K, rescale_input_RT) for rescale_input_RT in rescale_input_RTs]

        rescale_K[:, :2, :] = rescale_K[:, :2, :] * 0.5
        proj_matrices_1 = [torch.bmm(rescale_K, rescale_input_RT) for rescale_input_RT in rescale_input_RTs]

        rescale_K[:, :2, :] = rescale_K[:, :2, :] * 0.5
        proj_matrices_2 = [torch.bmm(rescale_K, rescale_input_RT) for rescale_input_RT in rescale_input_RTs]

        rescale_K[:, :2, :] = rescale_K[:, :2, :] * 0.5
        proj_matrices_3 = [torch.bmm(rescale_K, rescale_input_RT) for rescale_input_RT in rescale_input_RTs]

        return proj_matrices_0, proj_matrices_1, proj_matrices_2, proj_matrices_3


    def prepare_data(self, batch):
        input_imgs = []
        num_inputs = self.opt.input_view_num
        batch_size = batch['images'][0].shape[0]
        for i in range(num_inputs):
            input_imgs.append(batch["images"][i])
        BS, C, H, W = input_imgs[0].shape
        if self.opt.down_sample:
            for i in range(num_inputs):
                input_imgs[i] = F.interpolate(input_imgs[i], size=(H//2, W//2), mode="area")
        BS, C, H, W = input_imgs[0].shape
        target_H = math.floor( H / 8) * 8
        target_W = math.floor(W / 8) * 8
        start_H = (H - target_H) // 2
        start_W = (W - target_W) // 2
        # H and W should be divided by 8.
        for i in range(num_inputs):
            input_imgs[i] = input_imgs[i][:,:, start_H : start_H + target_H, start_W : start_W + target_W  ]
        # Camera parameters
        K = deepcopy(batch["cameras"][0]["K"])
        K_inv = deepcopy(batch["cameras"][0]["Kinv"])
        if self.opt.down_sample:
            K[:, 0:2, 0:3] = K[:, 0:2, 0:3] / 2.0
            K_inv = torch.inverse(K)
        K[:, 0, 2] = K[:, 0, 2] - start_W
        K[:, 1, 2] = K[:, 1, 2] - start_H
        K_inv = torch.inverse(K)

        input_RTs = []
        input_RTinvs = []
        for i in range(num_inputs):
            input_RTs.append(batch["cameras"][i]["P"])
            input_RTinvs.append(batch["cameras"][i]["Pinv"])


        proj_matrices_0, proj_matrices_1, proj_matrices_2, proj_matrices_3 = self.prepare_project(K, input_RTs)

        proj_matrices_0 = torch.stack(proj_matrices_0, 0).transpose(0,1).contiguous()
        proj_matrices_1 = torch.stack(proj_matrices_1, 0).transpose(0,1).contiguous()
        proj_matrices_2 = torch.stack(proj_matrices_2, 0).transpose(0,1).contiguous()
        proj_matrices_3 = torch.stack(proj_matrices_3, 0).transpose(0,1).contiguous()

        # stage_image = self.image_to_stage(input_imgs)
        stage_0 = []
        stage_1 = []
        stage_2 = []
        stage_3 = []

        for input_img in input_imgs:

            output_img = self.image_to_stage(input_img)
            stage_0.append(output_img['stage_0'])
            stage_1.append(output_img['stage_1'])
            stage_2.append(output_img['stage_2'])
            stage_3.append(output_img['stage_3'])

        stage_0 = torch.stack(stage_0, 0).transpose(0,1).contiguous()
        stage_1 = torch.stack(stage_1, 0).transpose(0,1).contiguous()
        stage_2 = torch.stack(stage_2, 0).transpose(0,1).contiguous()
        stage_3 = torch.stack(stage_3, 0).transpose(0,1).contiguous()

        imgs = {}
        imgs['stage_0'] = stage_0
        imgs['stage_1'] = stage_1
        imgs['stage_2'] = stage_2
        imgs['stage_3'] = stage_3

        proj = {}
        proj['stage_0'] = proj_matrices_0
        proj['stage_1'] = proj_matrices_1
        proj['stage_2'] = proj_matrices_2
        proj['stage_3'] = proj_matrices_3

        return {
            "imgs": imgs,
            "proj_matrices": proj,
            "depth_min": torch.tensor([425.0] * batch_size),
            "depth_max": torch.tensor([935.0] * batch_size)
            }, start_H, start_W, target_H, target_W

    def shift_data(self, warp_data):
        """
        shift data by 1.
        """

        for stage in warp_data['imgs'].keys():
            warp_data['imgs'][stage] = torch.roll(warp_data['imgs'][stage], -1, 1)

        for stage in warp_data['proj_matrices'].keys():

            warp_data['proj_matrices'][stage] = torch.roll(warp_data['proj_matrices'][stage], -1 ,1)

        return warp_data

    def forward(self, batch, thre=0.9):
        if self.opt.learnable_mvs:
            num_inputs = self.opt.input_view_num
            depths = []
            results = []
            warp_data, start_H, start_W, target_H, target_W = self.prepare_data(batch)
            all_masks = batch['masks']
            for i in range(num_inputs):
                warp_data = tocuda(warp_data)

                output = self.model(warp_data["imgs"], warp_data["proj_matrices"],
                        warp_data["depth_min"], warp_data["depth_max"])
                warp_data =  self.shift_data(warp_data)
                depth = output["refined_depth"]['stage_0']
                new_depth = torch.nn.functional.pad(depth, ( start_W, start_W, start_H, start_H), mode='replicate' )
                new_depth = new_depth / self.opt.scale_factor
                if (new_depth != new_depth).any():
                    print("NAN in depths")
                new_depth = new_depth[:, 0:1, :, :]
                results.append(new_depth)

            return results, None, None
        else:
            num_inputs = self.opt.input_view_num
            depths = []
            confidences = []
            results = []
            warp_data, start_H, start_W, target_H, target_W = self.prepare_data(batch)
            self.model.eval()
            all_masks = batch['masks']
            H, W = all_masks[0].shape[-2:]
            if self.opt.down_sample:
                H = H // 2
                W = W // 2
                #all_masks = torch.cat(all_masks)
                for i in range(len(all_masks)):
                    all_masks[i] = F.interpolate(all_masks[i]*1.0, size=(H,W), mode="nearest")
                    all_masks[i] = all_masks[i] == 1.0
                #all_masks = [all_masks[i:i+1] for i in range(all_masks.shape[0])]
            for i in range(num_inputs):
                warp_data = tocuda(warp_data)

                output = self.model(warp_data["imgs"], warp_data["proj_matrices"],
                        warp_data["depth_min"], warp_data["depth_max"])
                warp_data =  self.shift_data(warp_data)
                depth = output["refined_depth"]['stage_0']
                confidence = output["photometric_confidence"].unsqueeze(1)
                depths.append(deepcopy(depth))
                confidences.append(confidence)
                depth[confidence<thre] = 0.0
                new_depth = torch.zeros_like(batch['images'][0]).to(depth.device)
                new_depth = torch.nn.functional.pad(depth, ( start_W, start_W, start_H, start_H), mode='constant' )
                new_depth = new_depth / self.opt.scale_factor
                mask = all_masks[i].to(new_depth.device)
                if (new_depth != new_depth).any():
                    print("NAN in depths")
                new_depth = new_depth[:, 0:1, :, :]
                new_depth[~mask[:,:,0:H,0:W]] = 0.0
                results.append(new_depth)

            return results, depths, confidences


