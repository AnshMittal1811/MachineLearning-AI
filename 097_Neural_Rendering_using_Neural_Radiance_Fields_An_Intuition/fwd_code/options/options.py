import os

def get_model(opt):
    print("Loading model %s ... ")

    if opt.model_type == "multi_z_transformer":
        from models.multi_z_depth_cas import Multi_Z_Transformer
        model = Multi_Z_Transformer(opt)
    else:
        raise NotImplementedError("no such model")
    return model


def get_dataset(opt):

    print("Loading dataset %s ..." % opt.dataset)
    
    if opt.dataset == 'dtu':
        opt.path = os.path.join(opt.dataset_path, "DTU/")
        opt.camera_path = os.path.join(opt.dataset_path, "camera.npy")
        opt.depth_path = os.path.join(opt.dataset_path, "Depths_2/")
        opt.list_prefix = "new_"
        opt.image_size = (opt.H, opt.W)
        opt.scale_focal = False
        opt.max_imgs = 100000
        opt.z_near = 0.1
        opt.z_far = 20.0
        opt.skip_step = None
        # opt.suffix = ['_3_r5000']
        opt.suffix = ['']

        from data.DTU_dataset import DTU_Dataset
        return DTU_Dataset

    elif opt.dataset == 'shapenet':

        opt.path = os.path.join(opt.dataset_path, "NMR_Dataset")
        opt.list_prefix = "softras_"
        opt.scale_focal = True
        opt.max_imgs = 100000
        opt.z_near = 1.2
        opt.z_far = 4.0
        opt.image_size = (opt.H, opt.W)

        from data.shapenet import ShapeNet
        return ShapeNet
    else:
        raise NotImplementedError("no such dataset")

    return Dataset
