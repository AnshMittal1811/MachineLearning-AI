import socket
import torch
import numpy as np
import cv2
from absl import app
from absl import flags
from jax import config
from tqdm import tqdm
import time
import quaternion
from scipy.spatial.transform import Rotation

from octree.nerf import models
from octree.nerf import utils
from octree.nerf import datasets
import svox
from db_utils import *


FLAGS = flags.FLAGS
utils.define_flags()
flags.DEFINE_string("input", "./tree.npz", "Input octree npz")
flags.DEFINE_float("trim", 0.8, "trim size")
flags.DEFINE_float("dist", 1.2, "default dist in dataset")
flags.DEFINE_integer("size", 256, "render size")
flags.DEFINE_integer("port", 6006, "Port number for tcp")
config.parse_flags_with_absl()


def f_frame(img):
    img = (img.clamp_(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    # img[0], img[-1], img[:,0], img[:,-1] = 255, 255, 255, 255
    return img


def xyzquat2c2w(xyz, quat):
    c2w = np.zeros([4,4])
    c2w[:3,:3] = quaternion.as_rotation_matrix(np.quaternion(*quat))
    c2w[:,3] = np.concatenate([xyz, np.array([1.,])])
    c2w[:3,:3] = c2w[:3,:3].dot(Rotation.from_euler('xyz', (0,0,90), degrees=True).as_matrix())
    return c2w


def main(unused_argv):
    utils.update_flags(FLAGS)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### TODO
    w, h, focal = 800, 800, 1111.11  # dataset.w, dataset.h, dataset.focal    
    size = FLAGS.size
    focal = focal / (w / size)

    tree = svox.N3Tree.load(FLAGS.input, map_location=device)
    tree = lighten_tree(tree, FLAGS.trim)
    r = svox.VolumeRenderer(tree, step_size=FLAGS.renderer_step_size, ndc=None)
    print("I'm Ready!")

    # https://docs.python.org/ja/3/library/socket.html
    # https://docs.python.org/ja/3/library/socket.html#example
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', FLAGS.port))
                s.listen(1)
                conn, addr = s.accept()
                print(addr)
            except OSError as e:
                print(e); time.sleep(1); continue
            break

        while True:
            with conn:
                time.sleep(0.03)
                while True:
                    try:
                        data = conn.recv(1024)
                    except OSError as e:
                        print(e); time.sleep(1); continue
                    if not data:
                        print("no data"); break
                    try:
                        decode_data = data.decode("utf-8").split(",")
                        decode_data = [float(s) for s in decode_data]
                    except ValueError as e:
                        print(e); break
                    if not len(decode_data) == 7:
                        print("invalid data"); break

                    xyz = np.array(decode_data[:3])
                    quat = np.array(decode_data[3:])
                    dist = np.sum(np.power(xyz, 2)) ** 0.5
                    print(np.round(dist, 3), np.round(xyz, 3), np.round(quat, 3))

                    c2w = xyzquat2c2w(xyz, quat)
                    ### TODO
                    if 'IMG_' in FLAGS.input:
                        c2w[:3,3] = Rotation.from_euler(
                            'xyz', (90,-90,0), degrees=True).as_matrix().dot(c2w[:3,3])
                        c2w[:3,:3] = Rotation.from_euler(
                            'xyz', (90,-90,0), degrees=True).as_matrix().dot(c2w[:3,:3])

                    with torch.no_grad():
                        c2w = torch.from_numpy(c2w).float()
                        img = r.render_persp(c2w.to(device), size, size,
                                             fx=focal*(dist/FLAGS.dist), fast=True)
                    img = f_frame(img)

                    view = np.zeros([6])  # dummy. not used
                    view = ("{} " * 6)[:-1].format(*view).encode()
                    img = cv2.imencode(".png", img)[1].tobytes()
                    view = "{:04}    ".format(len(view)).encode() + view
                    conn.sendall(view + img)


if __name__ == "__main__":
    app.run(main)