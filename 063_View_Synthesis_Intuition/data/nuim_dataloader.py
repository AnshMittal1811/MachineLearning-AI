from util.camera_transformations import *
import torchvision

from data.disk_dataloader import DiskDataset


class ICLNUIMDataset(DiskDataset):
    '''
    Loads samples from the pre-rendered NUIM dataset: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
    Adapted from c++ code in https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
    '''

    # intrinsic camera matrix taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
    cam_K = {
        'fx': 481.2,
        'fy': -480.0,
        'cx': 319.5,
        'cy': 239.5
    }

    # intrinsic camera matrix as torch tensor
    K = torch.from_numpy(np.zeros((4,4)).astype(np.float32))
    K[0,0] = cam_K['fx']
    K[1,1] = cam_K['fy']
    K[0,2] = cam_K['cx']
    K[1,2] = cam_K['cy']
    K[2,2] = 1
    K[3,3] = 1 # we use 4x4 matrix for easier backward-calculations without removing indices, see projection/z_buffer_manipulator.py

    # and inverted matrix as well
    Kinv = torch.from_numpy(np.zeros((4,4)).astype(np.float32))
    Kinv[:3,:3] = invert_K(K[:3,:3])
    Kinv[3,3] = 1

    # max depth that we accept for ICL - can be bigger but we cut it at this value
    max_depth = 10.0

    # shape from ICL images in its original resolution. Format: HxW
    image_shape = (480, 640)

    def __init__(self,
                 path,
                 sampleOutput=True,
                 inverse_depth=False,
                 input_as_segmentation=False,
                 cacheItems=False,
                 transform=None,
                 out_shape=(480,640)):
        '''

        :param path: path/to/NUIM/files. Needs to be a directory with .png, .depth and .txt files, as can be obtained from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
        :param sampleOutput: whether or not to uniformly sample a second image + extrinsic camera pose (R|T) in the neighborhood of each accessed item.
                neighborhood is currently defined as: select uniformly at random any camera in the range [index-30, index+30) where index is the accessed item index.
                For example: If the 500. item is accessed, the second camera pose (R|T) will be from any of the poses of the items 470-530 (excluding 500).
        :param inverse_depth: If true, depth.pow(-1) is returned for the depth file (changing depth BEFORE applying transform object).
        :param transform: transform that should be applied to the input image AND the target depth
        :param out_shape: the output shape of images. If you apply transform that changes the output shape, we need to reflect this in a modified K matrix. Thus,
                if a "Resize" can be found in transform, then K will be recalculated w.r.t. the out_shape provided here. TODO: handle more than just "Resize".
        '''
        self.out_shape = out_shape

        DiskDataset.__init__(self,
                             path=path,
                             maxDepth=ICLNUIMDataset.max_depth,
                             imageInputShape=ICLNUIMDataset.image_shape,
                             sampleOutput=sampleOutput,
                             inverse_depth=inverse_depth,
                             input_as_segmentation=input_as_segmentation,
                             cacheItems=cacheItems,
                             transform=transform)

    def create_input_to_output_sample_map(self):
        inputToOutputIndex = []
        for idx in range(self.size):
            # sample second idx in [idx-30, idx+30) interval
            low = idx - 30 if idx >= 30 else 0
            high = idx + 30 if idx <= self.size - 30 else self.size
            output_idx = np.random.randint(low, high, 1)[0]  # high is exclusive

            # never return the same idx, default handling: just use +1 or -1 idx
            if output_idx == idx and self.size > 1:  # if we only have one sample, we can do nothing about this.
                output_idx = idx + 1 if idx < self.size - 1 else idx - 1

            inputToOutputIndex.append(output_idx)

        return inputToOutputIndex

    def modify_depth(self, depth):
        #return depth
        return np.fromfunction(lambda y, x: self.toImagePlane(depth, x, y), depth.shape, dtype=depth.dtype)

    def load_int_cam(self):
        if self.transform is not None and 'Resize' in str(self.transform):
            K2 = torch.from_numpy(np.zeros((4,4)).astype(np.float32))
            K2[0,0] = 0.751875 * self.out_shape[1]      #cam_K2['fx']
            K2[1,1] = -1.0 * self.out_shape[0]          #cam_K2['fy']
            K2[0,2] = 0.5 * self.out_shape[1]           #cam_K2['cx']
            K2[1,2] = 0.5 * self.out_shape[0]           #cam_K2['cy']
            K2[2,2] = 1
            K2[3,3] = 1 # we use 4x4 matrix for easier backward-calculations without removing indices, see projection/z_buffer_manipulator.py

            K2inv = torch.from_numpy(np.zeros((4,4)).astype(np.float32))
            K2inv[:3,:3] = invert_K(K2[:3,:3])
            K2inv[3,3] = 1

            return K2, K2inv
        else:
            return ICLNUIMDataset.K, ICLNUIMDataset.Kinv

    def toImagePlane(self, depth, x, y):

        # taken from the figure in: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
        #z = ICLNUIMDataset.cam_K['fx'] * np.sqrt( (depth**2) / (x**2 + y**2 + ICLNUIMDataset.cam_K['fx']**2))

        # taken from the c++ code implementation at https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html in file VaFRIC.cpp#getEuclidean2PlanarDepth
        x_plane = (x - ICLNUIMDataset.cam_K['cx']) / ICLNUIMDataset.cam_K['fx']
        y_plane = (y - ICLNUIMDataset.cam_K['cy']) / ICLNUIMDataset.cam_K['fy']
        z = depth / np.sqrt(x_plane ** 2 + y_plane ** 2 + 1)

        return z

    def load_data(self, dir_content):
        img = sorted([f for f in dir_content if f.endswith('.png') and not f.endswith('.seg.png')])
        img_seg = sorted([f for f in dir_content if f.endswith('.seg.png')])
        depth = sorted([f for f in dir_content if f.endswith('.depth') and not f.endswith('.gl.depth')])
        has_depth = len(depth) > 0
        depth_binary = sorted([f for f in dir_content if f.endswith('.depth.npy') and not f.endswith('.gl.depth.npy')])
        has_binary_depth = len(depth_binary) > 0
        cam = sorted([f for f in dir_content if f.endswith('.txt')])

        return img, img_seg, depth, has_depth, depth_binary, has_binary_depth, cam, len(img), None, img


def getEulerAngles(R):
    ry = np.arcsin(R[0,2])
    rz = np.arccos(R[0,0] / np.cos(ry))
    rx = np.arccos(R[2,2] / np.cos(ry))

    return rx, ry, rz


def test():

    size = 256

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
    ])

    dataset = ICLNUIMDataset("/home/lukas/Desktop/datasets/ICL-NUIM/prerendered_data/living_room_traj2_loop",
                             sampleOutput=True,
                             inverse_depth=False,
                             cacheItems=False,
                             transform=transform,
                             out_shape=(size, size))
    #dataset = ICLNUIMDataset("sample", sampleOutput=True, transform=transform);

    print("Length of dataset: {}".format(len(dataset)))

    # Show first item in the dataset
    i = 400
    item = dataset.__getitem__(i)

    #print(item["depth"].numpy().flags)


    print(item["image"].shape)
    print(item["depth"].shape)

    print("RT1:\n{}". format(item['cam']['RT1']))
    print("RT1 euler angles in radians: {}".format(getEulerAngles(item['cam']['RT1'])))
    print("RT2:\n{}".format(item['cam']['RT2']))
    print("RT2 euler angles in radians: {}".format(getEulerAngles(item['cam']['RT2'])))
    print("K:\n{}".format(item['cam']['K']))

    print("RT1inv:\n{}". format(item['cam']['RT1inv']))
    print("RT2inv:\n{}".format(item['cam']['RT2inv']))
    print("Kinv:\n{}".format(item['cam']['Kinv']))

    print("K*Kinv:\n{}".format(item['cam']['K'].matmul(item['cam']['Kinv'])))
    print("RT1*RT1inv:\n{}".format(item['cam']['RT1'].matmul(item['cam']['RT1inv'])))
    print("RT2*RT2inv:\n{}".format(item['cam']['RT2'].matmul(item['cam']['RT2inv'])))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=item['image'].shape[1:])
    fig.suptitle("Sample " + str(i), fontsize=16)
    img = np.moveaxis(item['image'].numpy(), 0, -1)
    depth = np.moveaxis(item['depth'].numpy(), 0, -1).squeeze()
    out_img = np.moveaxis(item['output']['image'].numpy(), 0, -1)
    out_seg = np.moveaxis(item['output']['seg'].numpy(), 0, -1)
    out_idx = item['output']['idx']
    fig.add_subplot(1, 4, 1)
    plt.title("Image")
    plt.imshow(img)
    fig.add_subplot(1, 4, 2)
    plt.title("Depth Map")
    plt.imshow(depth, cmap='gray')
    fig.add_subplot(1, 4, 3)
    plt.title("Output Image " + str(out_idx))
    plt.imshow(out_img)
    fig.add_subplot(1, 4, 4)
    plt.title("Output Seg " + str(out_idx))
    plt.imshow(out_seg)

    plt.show()

    print(np.min(depth), np.max(depth))



if __name__ == "__main__":
    # execute only if run as a script
    test()
