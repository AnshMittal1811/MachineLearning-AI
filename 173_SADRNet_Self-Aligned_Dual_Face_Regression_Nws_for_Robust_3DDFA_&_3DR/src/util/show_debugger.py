import config
from src.dataset.uv_face import mean_shape_map_np, uvm2mesh
import matplotlib.pyplot as plt
from src.visualize.render_mesh import render_face_orthographic
import numpy as np


def show_batch(batch_data):
    print('**********')
    image_t, pos_map_t, offset_map_t, attention_mask_t = batch_data

    for idx in range(4):
        image = image_t[idx].permute(1, 2, 0).cpu().numpy()
        pos_map = pos_map_t[idx].permute(1, 2, 0).cpu().numpy() * config.POSMAP_FIX_RATE
        offset_map = offset_map_t[idx].permute(1, 2, 0).cpu().numpy()
        shape = offset_map + mean_shape_map_np
        attention = attention_mask_t[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(image)
        plt.show()
        shapem = shape * 50 + 128
        face_mesh = uvm2mesh(shapem / 128)
        ret = render_face_orthographic(face_mesh)
        plt.imshow(ret)
        face_mesh = uvm2mesh(pos_map / 128)
        ret = render_face_orthographic(face_mesh, (image * 255).astype(np.uint8))
        plt.imshow(ret)
        plt.show()
        plt.imshow(attention)
        plt.show()
    print('############')


def show_batch_out(ipt, out):
    print('**********')
    image_t = ipt
    pos_map_t, offset_map_t, attention_mask_t, kpt_uvm_t = out['face_uvm'], out['offset_uvm'], out['attention_mask'], \
                                                           out['kpt_uvm']

    for idx in range(2):
        image = image_t[idx].permute(1, 2, 0).cpu().numpy()
        pos_map = pos_map_t[idx].permute(1, 2, 0).cpu().numpy() * config.POSMAP_FIX_RATE
        offset_map = offset_map_t[idx].permute(1, 2, 0).cpu().numpy()
        kpt_map = kpt_uvm_t[idx].permute(1, 2, 0).cpu().numpy() * config.POSMAP_FIX_RATE
        shape = offset_map + mean_shape_map_np
        attention = attention_mask_t[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(image)
        plt.show()
        image[image < 0] = 0
        image[image > 1] = 1
        shapem = shape * 50 + 128
        face_mesh = uvm2mesh(shapem / 128)
        ret = render_face_orthographic(face_mesh)
        plt.imshow(ret)
        plt.show()

        face_mesh = uvm2mesh(pos_map / 128)
        ret = render_face_orthographic(face_mesh, (image * 255).astype(np.uint8))
        plt.imshow(ret)
        plt.show()

        face_mesh = uvm2mesh(kpt_map / 128)
        ret = render_face_orthographic(face_mesh, (image * 255).astype(np.uint8))
        plt.imshow(ret)
        plt.show()

        plt.imshow(attention)
        plt.show()
    print('############')
