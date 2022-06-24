import numpy as np
import os
import argparse
import torch
import random
import config
import json
from src.dataset.dataloader import img_to_tensor, uv_map_to_tensor
from src.dataset.dataloader import make_data_loader, make_dataset, ImageData
from src.model.loss import *
from PIL import Image
from src.util.printer import DecayVarPrinter
from src.visualize.render_mesh import render_face_orthographic, render_uvm
from src.visualize.plot_verts import plot_kpt, compare_kpt
from src.dataset.uv_face import uvm2mesh
import matplotlib.pyplot as plt


class BasePredictor:
    def __init__(self, weight_path):
        self.model = self.get_model(weight_path)

    def get_model(self, weight_path):
        raise NotImplementedError

    def predict(self, img):
        raise NotImplementedError


class SADRNPredictor(BasePredictor):
    def __init__(self, weight_path):
        super(SADRNPredictor, self).__init__(weight_path)

    def get_model(self, weight_path):
        from src.model.SADRN import get_model
        model = get_model()
        pretrained = torch.load(weight_path, map_location=config.DEVICE)
        model_dict = model.state_dict()
        match_dict = {k: v for k, v in pretrained.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(match_dict)
        model.load_state_dict(model_dict)
        model = model.to(config.DEVICE)
        model.eval()
        return model

    def predict(self, img):
        # TODO normalize
        image = (img / 255.0).astype(np.float32)
        for ii in range(3):
            image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
                image[:, :, ii].var() + 0.001)
        image = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
        with torch.no_grad():
            out = self.model({'img': image}, {}, 'predict')
        out['face_uvm'] *= config.POSMAP_FIX_RATE
        out['kpt_uvm'] *= config.POSMAP_FIX_RATE

        out['face_uvm'] = out['face_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
        out['kpt_uvm'] = out['kpt_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
        out['offset_uvm'] = out['offset_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
        out['attention_mask'] = out['attention_mask'].cpu().permute(0, 2, 3, 1).numpy()[0]
        return out


class SADRNv2Predictor(SADRNPredictor):
    def __init__(self, weight_path):
        super(SADRNv2Predictor, self).__init__(weight_path)

    def get_model(self, weight_path):
        from src.model.SADRNv2 import get_model
        model = get_model()
        pretrained = torch.load(weight_path, map_location=config.DEVICE)
        model_dict = model.state_dict()
        match_dict = {k: v for k, v in pretrained.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(match_dict)
        model.load_state_dict(model_dict)
        model = model.to(config.DEVICE)
        model.eval()
        return model


class Evaluator:
    def __init__(self):
        self.all_eval_data = None

        self.metrics = {"nme3d": NME(),
                        "nme2d": NME2D(),
                        "kpt2d": KptNME2D(),
                        "kpt3d": KptNME(),
                        "rec": RecLoss(),
                        }

        self.printer = DecayVarPrinter()

    def get_data(self):
        val_dataset = make_dataset(config.VAL_DIR, 'val')
        self.all_eval_data = val_dataset.val_data

    def get_example_data(self):
        val_dataset = make_dataset(['data/example'], 'val')
        self.all_eval_data = val_dataset.val_data

    def show_face_uvm(self, face_uvm, img, gt_uvm=None, is_show=True):
        ret = render_uvm(face_uvm, img)
        if is_show:
            plt.imshow(ret)
            plt.show()
        ret_kpt = plot_kpt(img, face_uvm[uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]])
        if is_show:
            plt.imshow(ret_kpt)
            plt.show()
        if gt_uvm is not None:
            ret_cmp = compare_kpt(face_uvm, gt_uvm, img)
            if is_show:
                plt.imshow(ret_cmp)
                plt.show()
            return ret, ret_kpt, ret_cmp
        else:
            return ret, ret_kpt

    def evaluate(self, predictor, is_visualize=False):
        with torch.no_grad():
            predictor.model.eval()
            for i in range(len(self.all_eval_data)):
                item = self.all_eval_data[i]
                init_img = item.get_image()
                image = (init_img / 255.0).astype(np.float32)
                for ii in range(3):
                    image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
                        image[:, :, ii].var() + 0.001)
                image = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
                out = predictor.model({'img': image}, 'predict')

                init_attention_mask = item.get_attention_mask() / 255.0
                attention_mask = Image.fromarray(init_attention_mask)
                attention_mask = attention_mask.resize((32, 32), Image.BILINEAR)
                attention_mask = np.array(attention_mask)
                attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).unsqueeze(0)
                init_pos_map = (item.get_pos_map())
                pos_map = pos_map = init_pos_map / config.POSMAP_FIX_RATE
                pos_map = uv_map_to_tensor(pos_map).float().to(config.DEVICE).unsqueeze(0)

                for key in self.metrics:
                    func = self.metrics[key]
                    error = func(pos_map, out['face_uvm']).cpu().numpy()
                    self.printer.update_variable_avg(key, error)
                    print(f'{key}:{error:05f},', end=' ')
                print(f'{i}/{len(self.all_eval_data)}')

                if is_visualize:
                    face_uvm_out = out['face_uvm'][0].cpu().permute(1, 2, 0).numpy() * config.POSMAP_FIX_RATE
                    self.show_face_uvm(face_uvm_out, init_img)

        print('Dataset Results')
        for key in self.metrics:
            print(self.printer.get_variable_str(key))

    def evaluate_aflw(self, predictor, is_visualize=False):
        pose_list = np.load('data/uv_data/AFLW2000-3D.pose.npy')
        with torch.no_grad():
            predictor.model.eval()
            for i in range(len(self.all_eval_data)):
                yaw_angle = np.abs(pose_list[i])
                if yaw_angle <= 30:
                    angle_str = '[0,30]'
                elif yaw_angle <= 60:
                    angle_str = '[30,60]'
                elif yaw_angle <= 90:
                    angle_str = '[60,90]'

                item = self.all_eval_data[i]
                init_img = item.get_image()
                image = (init_img / 255.0).astype(np.float32)
                for ii in range(3):
                    image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
                        image[:, :, ii].var() + 0.001)
                image = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
                out = predictor.model({'img': image}, 'predict')

                init_attention_mask = item.get_attention_mask() / 255.0
                attention_mask = Image.fromarray(init_attention_mask)
                attention_mask = attention_mask.resize((32, 32), Image.BILINEAR)
                attention_mask = np.array(attention_mask)
                attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).unsqueeze(0)
                init_pos_map = (item.get_pos_map())
                pos_map = init_pos_map / config.POSMAP_FIX_RATE
                pos_map = uv_map_to_tensor(pos_map).float().to(config.DEVICE).unsqueeze(0)

                for key in self.metrics:
                    func = self.metrics[key]
                    error = func(pos_map, out['face_uvm']).cpu().numpy()
                    self.printer.update_variable_avg(angle_str + key, error)
                    print(f'{angle_str}{key}:{error:05f},', end=' ')
                print(f'{i}/{len(self.all_eval_data)}')

                if is_visualize:
                    face_uvm_out = out['face_uvm'][0].cpu().permute(1, 2, 0).numpy() * config.POSMAP_FIX_RATE
                    self.show_face_uvm(face_uvm_out, init_img)

        print('Dataset Results')
        for key in self.metrics:
            for angle_str in ["[0,30]", "[30,60]", "[60,90]"]:
                print(self.printer.get_variable_str(angle_str + key))

    def evaluate_example(self, predictor, is_visualize=True, output_folder='data/output/SADRN-out'):
        with torch.no_grad():
            predictor.model.eval()
            for i in range(len(self.all_eval_data)):
                item = self.all_eval_data[i]
                init_img = item.get_image()
                image = (init_img / 255.0).astype(np.float32)
                for ii in range(3):
                    image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
                        image[:, :, ii].var() + 0.001)
                image = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
                out = predictor.model({'img': image}, 'predict')

                print(f'{i}/{len(self.all_eval_data)}')

                if is_visualize:
                    face_uvm_out = out['face_uvm'][0].cpu().permute(1, 2, 0).numpy() * config.POSMAP_FIX_RATE
                    ret, ret_kpt = self.show_face_uvm(face_uvm_out, init_img, None, True)
                    # io.imsave(f'{output_folder}/{i}_cmp.jpg', ret_cmp)
                    io.imsave(f'{output_folder}/{i}_kpt.jpg', ret_kpt)
                    io.imsave(f'{output_folder}/{i}_face.jpg', ret)
                    io.imsave(f'{output_folder}/{i}_img.jpg', init_img)

        print('Dataset Results')


if __name__ == '__main__':
    if config.NET == 'SADRNv2':
        predictor_1 = SADRNv2Predictor(config.PRETAINED_MODEL)
        evaluator = Evaluator()
        # evaluator.get_data()
        # evaluator.evaluate(predictor_1)
        # evaluator.evaluate_aflw(predictor_1)
        evaluator.get_example_data()
        evaluator.evaluate_example(predictor_1)
