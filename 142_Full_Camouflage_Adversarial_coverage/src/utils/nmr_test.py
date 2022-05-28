from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ntpath

import numpy as np
import scipy.misc
import math

# import chainer
import torch
import neural_renderer
from glob import glob
import os
#############
### Utils ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

# get_params(((-25, 16, 20), (-45, 180, 0)), ((-45, 3, 0.8), (0, 0, 0)))  # test example
def get_params(carlaTcam, carlaTveh):  # carlaTcam: tuple of 2*3
    scale = 0.40
    # scale = 0.38
    # calc eye
    eye = [0, 0, 0]
    for i in range(0, 3):
        # eye[i] = (carlaTcam[0][i] - carlaTveh[0][i]) * scale
        eye[i] = carlaTcam[0][i] * scale

    # calc camera_direction and camera_up
    pitch = math.radians(carlaTcam[1][0])
    yaw = math.radians(carlaTcam[1][1])
    roll = math.radians(carlaTcam[1][2])
    # 需不需要确定下范围？？？
    cam_direct = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    cam_up = [math.cos(math.pi / 2 + pitch) * math.cos(yaw), math.cos(math.pi / 2 + pitch) * math.sin(yaw),
              math.sin(math.pi / 2 + pitch)]

    # 如果物体也有旋转，则需要调整相机位置和角度，和物体旋转方式一致
    # 先实现最简单的绕Z轴旋转
    p_cam = eye
    p_dir = [eye[0] + cam_direct[0], eye[1] + cam_direct[1], eye[2] + cam_direct[2]]
    p_up = [eye[0] + cam_up[0], eye[1] + cam_up[1], eye[2] + cam_up[2]]
    p_l = [p_cam, p_dir, p_up]
    trans_p = []
    for p in p_l:
        if math.sqrt(p[0] ** 2 + p[1] ** 2) == 0:
            cosfi = 0
            sinfi = 0
        else:
            cosfi = p[0] / math.sqrt(p[0] ** 2 + p[1] ** 2)
            sinfi = p[1] / math.sqrt(p[0] ** 2 + p[1] ** 2)
        cossum = cosfi * math.cos(math.radians(carlaTveh[1][1])) + sinfi * math.sin(math.radians(carlaTveh[1][1]))
        sinsum = math.cos(math.radians(carlaTveh[1][1])) * sinfi - math.sin(math.radians(carlaTveh[1][1])) * cosfi
        trans_p.append([math.sqrt(p[0] ** 2 + p[1] ** 2) * cossum, math.sqrt(p[0] ** 2 + p[1] ** 2) * sinsum, p[2]])

    return trans_p[0], \
           [trans_p[1][0] - trans_p[0][0], trans_p[1][1] - trans_p[0][1], trans_p[1][2] - trans_p[0][2]], \
           [trans_p[2][0] - trans_p[0][0], trans_p[2][1] - trans_p[0][1], trans_p[2][2] - trans_p[0][2]]
    

########################################################################
############ Wrapper class for the chainer Neural Renderer #############
##### All functions must only use numpy arrays as inputs/outputs #######
########################################################################
class NMR(object):
    def __init__(self):
        # setup renderer
        renderer = neural_renderer.Renderer(camera_mode='look')
        self.renderer = renderer

    def to_gpu(self, device=0):
        # self.renderer.to_gpu(device)
        self.cuda_device = device

    def forward_mask(self, vertices, faces):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
        Returns:
            masks: B X 256 X 256 numpy array
        '''
        # self.faces = torch.autograd.Variable(chainer.cuda.to_gpu(faces, self.cuda_device))
        # self.vertices = torch.autograd.Variable(chainer.cuda.to_gpu(vertices, self.cuda_device))
        self.faces = torch.autograd.Variable(faces.cuda())
        self.vertices = torch.autograd.Variable(vertices.cuda())

        self.masks = self.renderer.render_silhouettes(self.vertices, self.faces)

        masks = self.masks.data.get()
        return masks
    #
    # def backward_mask(self, grad_masks):
    #     ''' Compute gradient of vertices given mask gradients.
    #     Args:
    #         grad_masks: B X 256 X 256 numpy array
    #     Returns:
    #         grad_vertices: B X N X 3 numpy array
    #     '''
    #     self.masks.grad = grad_masks.cuda()
    #     self.masks.backward()
    #     return self.vertices.grad.get()

    def forward_img(self, vertices, faces, textures):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
            textures: B X F X T X T X T X 3 numpy array
        Returns:
            images: B X 3 x 256 X 256 numpy array
        '''
        # self.faces = torch.autograd.Variable(chainer.cuda.to_gpu(faces, self.cuda_device))
        # self.vertices = torch.autograd.Variable(chainer.cuda.to_gpu(vertices, self.cuda_device))
        # self.textures = torch.autograd.Variable(chainer.cuda.to_gpu(textures, self.cuda_device))
        self.faces = faces
        self.vertices = vertices
        self.textures = textures
        self.images,_,_ = self.renderer.render(self.vertices, self.faces, self.textures)

        # images = self.images.data.get()
        # images = self.images[0]
        return self.images


    # def backward_img(self, grad_images):
    #     ''' Compute gradient of vertices given image gradients.
    #     Args:
    #         grad_images: B X 3? X 256 X 256 numpy array
    #     Returns:
    #         grad_vertices: B X N X 3 numpy array
    #         grad_textures: B X F X T X T X T X 3 numpy array
    #     '''
    #     # self.images.grad = chainer.cuda.to_gpu(grad_images, self.cuda_device)
    #     self.images.grad = grad_images.cuda()
    #     self.images.backward()
    #     return self.vertices.grad.get(), self.textures.grad.get()

########################################################################
################# Wrapper class a rendering PythonOp ###################
##### All functions must only use torch Tensors as inputs/outputs ######
########################################################################
class Render(torch.autograd.Function):
    # TODO(Shubham): Make sure the outputs/gradients are on the GPU
    def __init__(self, renderer):
        super(Render, self).__init__()
        self.renderer = renderer

    def forward(self, vertices, faces, textures=None):
        # B x N x 3
        # Flipping the y-axis here to make it align with the image coordinate system!
        # vs = vertices.cpu().numpy()
        vs = vertices
        vs[:, :, 1] *= -1
        # fs = faces.cpu().numpy()
        fs = faces
        if textures is None:
            self.mask_only = True
            masks = self.renderer.forward_mask(vs, fs)
            return masks   # , convert_as(torch.Tensor(masks), vertices)
        else:
            self.mask_only = False
            # ts = textures.cpu().data.numpy()
            ts = textures
            imgs = self.renderer.forward_img(vs, fs, ts)
            # return convert_as(torch.Tensor(imgs), vertices)
            return imgs

    # def backward(self, grad_out):
    #     # g_o = grad_out.cpu().numpy()
    #     if self.mask_only:
    #         grad_verts = self.renderer.backward_mask(grad_out)
    #         # grad_verts = convert_as(torch.Tensor(grad_verts), grad_out)
    #         grad_tex = None
    #     else:
    #         grad_verts, grad_tex = self.renderer.backward_img(grad_out)
    #         # grad_verts = convert_as(torch.Tensor(grad_verts), grad_out)
    #         # grad_tex = convert_as(torch.Tensor(grad_tex), grad_out)
    #
    #     grad_verts[:, :, 1] *= -1
    #     return grad_verts, None, grad_tex


########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, img_size=720):
        super(NeuralRenderer, self).__init__()
        self.renderer = NMR()
        
        # rendering
        self.renderer.renderer.image_size = img_size
        
        # camera
        self.renderer.renderer.camera_mode = 'look'
        self.renderer.renderer.viewing_angle = 45
        # test example
        eye, camera_direction, camera_up = get_params(((-25, 16, 20), (-45, 180, 0)), ((-45, 3, 0.8), (0, 0, 0)))
        # [8.2, 5.33, 7.871999999999999] eye
        # [-0.7071067811865479, 0.0, 0.707106781186547] camera up
        # [-1.0, 0.0, -1.0] camera direction

        self.renderer.renderer.eye = eye
        self.renderer.renderer.camera_direction = camera_direction
        self.renderer.renderer.camera_up = camera_up 
        
        # light
        self.renderer.renderer.light_intensity_ambient = 0.5
        self.renderer.renderer.light_intensity_directional = 0.5
        self.renderer.renderer.light_color_ambient = [1, 1, 1]  # white
        self.renderer.renderer.light_color_directional = [1, 1, 1]  # white
        self.renderer.renderer.light_direction = [0, 0, 1]  # up-to-down
        
                      
            
        self.renderer.to_gpu()

        self.proj_fn = None
        self.offset_z = 5.
        
        self.RenderFunc = Render(self.renderer)

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.renderer.light_intensity_ambient = 1
        self.renderer.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        self.renderer.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, textures=None):
        if textures is not None:
            return self.RenderFunc.forward(vertices, faces, textures)
        else:
            return self.RenderFunc.forward(vertices, faces)


def example():
    obj_file = 'audi_et_te.obj'
    data_path = '../data/phy_attack/train/'
    img_save_dir = '../data/phy_attack/render_test_res'
    
    vertices, faces = neural_renderer.load_obj(obj_file)
    
    texture_mask = np.zeros((faces.shape[0], 2, 2, 2, 3), 'int8')
    with open('./all_faces.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)
    print(texture_mask.size())
    mask_renderer = NeuralRenderer()
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
    vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
    # Textures
    texture_size = 2
    textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
    textures = torch.from_numpy(textures).cuda(device=0)
    print(textures.size())
    textures = textures * texture_mask
    
    
    # data = np.load(data_path)
    data_lsit = glob(os.path.join(data_path, "*.npy"))
    for data_path in data_lsit:
        data = np.load(data_path)
        img = data['img']
        veh_trans = data['veh_trans']
        cam_trans = data['cam_trans']
        eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)
        mask_renderer.renderer.renderer.eye = eye
        mask_renderer.renderer.renderer.camera_direction = camera_direction
        mask_renderer.renderer.renderer.camera_up = camera_up

        imgs_pred = mask_renderer.forward(vertices_var, faces_var, textures)
        im_rendered = imgs_pred.data.cpu().numpy()[0]
        im_rendered = np.transpose(im_rendered, (1,2,0))



        print(im_rendered.shape)
        print(np.max(im_rendered), np.max(img))
        scipy.misc.imsave(img_save_dir + 'test_render.png', im_rendered)
        scipy.misc.imsave(img_save_dir + 'test_origin.png', img)
    scipy.misc.imsave(img_save_dir + 'test_total.png', np.add(img, 255 * im_rendered))
    
# def parse_npz():
#     obj_file = 'audi_et.obj'
#     data_path = '../data/phy_attack/train/'
    
#     vertices, faces = neural_renderer.load_obj(obj_file)
#     mask_renderer = NeuralRenderer()
#     faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
#     vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
#     # Textures
#     texture_size = 2
#     textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
#     textures = torch.from_numpy(textures).cuda(device=0)
    
#     names = os.listdir(data_path)
#     ind = 0
#     for name in names:
#         path = data_path + name
#         data = np.load(path)
#         img = data['img']
#         veh_trans = data['veh_trans']
#         cam_trans = data['cam_trans']
#         cam_trans = cam_trans.astype(np.float64)
#         print('before modify')
#         print(veh_trans)
#         print(cam_trans)
#         # modify
#         cam_trans[0][0] = cam_trans[0][0] + veh_trans[0][0]
#         cam_trans[0][1] = cam_trans[0][1] + veh_trans[0][1]
#         cam_trans[0][2] = cam_trans[0][2] + veh_trans[0][2]
        
#         veh_trans[0][2] = veh_trans[0][2] + 0.2
#         print('after modify')
#         print(veh_trans)
#         print(cam_trans)
               
#         eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)
#         mask_renderer.renderer.renderer.eye = eye
#         mask_renderer.renderer.renderer.camera_direction = camera_direction
#         mask_renderer.renderer.renderer.camera_up = camera_up        
        
#         imgs_pred = mask_renderer.forward(vertices_var, faces_var, textures)
#         im_rendered = imgs_pred.data.cpu().numpy()[0]
#         im_rendered = np.transpose(im_rendered, (1,2,0))  # 800*800*3
        
#         for i in range(0, 800):
#             for j in range(0, 800):
#                 if not (im_rendered[i][j][0] == 0 and im_rendered[i][j][1] == 0 and im_rendered[i][j][2] == 0):
#                     img[i][j][0] = 0
#                     img[i][j][1] = 0
#                     img[i][j][2] = 0
        
#         # save back
#         np.savez(path, img=img, veh_trans=veh_trans, cam_trans=cam_trans)
#         print('done for file', name)
#         ind = ind + 1


def run(data_path, file_name):
    obj_file = 'audi_et.obj'
    # data_path = '../data/phy_attack/train/data132.npz'
    img_save_dir = '../data/submasks/'
    
    vertices, faces = neural_renderer.load_obj(obj_file)
    
    texture_mask = np.zeros((faces.shape[0], 2, 2, 2, 3), 'int8')
    with open('./all_faces.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            texture_mask[int(face_id) - 1, :, :, :, :] = 1;
    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)
    # print(texture_mask.size())
    mask_renderer = NeuralRenderer(608)
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
    vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
    # Textures
    texture_size = 2
    textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
    textures = torch.from_numpy(textures).cuda(device=0)
    # print(textures.size())
    textures = textures * texture_mask
    
    
    data = np.load(data_path)
    img = data['img']
    veh_trans = data['veh_trans']
    cam_trans = data['cam_trans']
    eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)
    mask_renderer.renderer.renderer.eye = eye
    mask_renderer.renderer.renderer.camera_direction = camera_direction
    mask_renderer.renderer.renderer.camera_up = camera_up 
  
    imgs_pred = mask_renderer.forward(vertices_var, faces_var, textures)
    im_rendered = imgs_pred.data.cpu().numpy()[0]
    im_rendered = np.transpose(im_rendered, (1,2,0))
    
    im_rendered = np.logical_or(im_rendered[:, :, 0], im_rendered[:, :, 1], im_rendered[:, :, 2])
    # print(im_rendered.shape)
    # print(np.max(im_rendered), np.max(img))
    # scipy.misc.imsave(img_save_dir + 'test_render.png', im_rendered)
    scipy.misc.imsave(img_save_dir + file_name + '.png', im_rendered)

if __name__ == '__main__':
    example()
    '''
    train_dir = '../data/phy_attack/test/'
    files = os.listdir(train_dir)
    for file in tqdm.tqdm(files):
        # print(file[:-4])
        run(os.path.join(train_dir, file), file[:-4])
    
    train_dir = '../data/phy_attack/train/'
    files = os.listdir(train_dir)
    for file in tqdm.tqdm(files):
        # print(file[:-4])
        run(os.path.join(train_dir, file), file[:-4])
    '''

