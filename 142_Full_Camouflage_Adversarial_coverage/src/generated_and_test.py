import torch
from torch.utils.data import DataLoader
from data_loader import MyDatasetTestAdv
from tqdm import tqdm
import numpy as np
import sys
import argparse
from PIL import Image
import os


sys.path.append("./neural_renderer/")
import neural_renderer

parser = argparse.ArgumentParser()

parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--obj", type=str, default='carassets/audi_et_te.obj')
parser.add_argument("--faces", type=str, default='carassets/exterior_face.txt') # exterior_face   all_faces
parser.add_argument("--textures", type=str, default='textures/texture_camouflage.npy')
parser.add_argument("--datapath", type=str, default="carla_dataset/")
args = parser.parse_args()


BATCH_SIZE = args.batchsize
mask_dir = os.path.join(args.datapath, 'masks/')

obj_file =args.obj
texture_size = 6

vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)


# Camouflage Textures
texture_content_adv = torch.from_numpy(np.load(args.textures)).cuda(device=0)

texture_origin =textures[None, :, :, :, :, :].cuda(device=0)
texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
with open(args.faces, 'r') as f:
    face_ids = f.readlines()
    for face_id in face_ids:
        if face_id != '\n':
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)



def cal_texture(texture_content, CONTENT=False):
    textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


@torch.no_grad()
def run_cam(data_dir, batch_size=BATCH_SIZE):
    print(data_dir)
    dataset = MyDatasetTestAdv(data_dir, input_size, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=2,
    )

    print(len(dataset))
    tqdm_loader = tqdm(loader)
    
    textures_adv = cal_texture(texture_content_adv, CONTENT=True)
    dataset.set_textures(textures_adv)
    for i, (index, total_img, texture_img, _,  filename) in enumerate(tqdm_loader):
            texture_img_np = total_img.data.cpu().numpy()[0]
            texture_img_np = Image.fromarray(np.transpose(texture_img_np, (1, 2, 0)).astype('uint8'))
            filename = filename[0].split('.')[0]
            save_path = 'savedImage'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            texture_img_np.save(fr'{save_path}/{filename}.png')
            
            # Yolo-v5 detection
            results = net(texture_img_np)
            results.save(fr'{save_path}/{filename}_pred.png')
            results.show()



if __name__ == "__main__":
    data_dir = f"{args.datapath}/test/"
    batch_size = 1
    input_size = 800
    net = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5m, yolov5x, custom
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()


    run_cam(data_dir)