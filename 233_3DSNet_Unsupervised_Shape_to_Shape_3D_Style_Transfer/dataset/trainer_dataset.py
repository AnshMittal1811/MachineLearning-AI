import torch
import dataset.dataset_shapenet as dataset_shapenet
import dataset.dataset_smxl as dataset_smxl
import dataset.augmenter as augmenter
from easydict import EasyDict


class TrainerDataset(object):
    def __init__(self, opt):
        super(TrainerDataset, self).__init__()
        self.opt = opt
        self.SVR_0 = opt.SVR_0
        self.SVR_1 = opt.SVR_1
        self.family = opt.family
        if opt.dataset == 'SMXL':
            self.dataset_class = dataset_smxl.SMXL
        elif opt.dataset == 'ShapeNet':
            self.dataset_class = dataset_shapenet.ShapeNet

    def build_dataset(self):
        """
        Create dataset
        """

        self.datasets = EasyDict()

        # Create Datasets
        # Fixing class names in case they contain '.' so that they are valid keys for nn.DataParallel
        self.classes = [c.replace('.', ' ') for c in self.classes]
        ###
        # Please notice that self.classes and self.opt.class_choice are now different
        self.datasets.dataset_train = {
            self.classes[0]: self.dataset_class(self.opt, self.family, self.opt.class_choice[0], self.SVR_0, train=True),
            self.classes[1]: self.dataset_class(self.opt, self.family, self.opt.class_choice[1], self.SVR_1, train=True)
        }

        self.datasets.dataset_test = {
            self.classes[0]: self.dataset_class(self.opt, self.family, self.opt.class_choice[0], self.SVR_0, train=False),
            self.classes[1]: self.dataset_class(self.opt, self.family, self.opt.class_choice[1], self.SVR_1, train=False)
        }

        if not self.opt.demo or not self.opt.use_default_demo_samples:
            # Create dataloaders
            self.datasets.dataloader_train = {}
            self.datasets.dataloader_train[self.classes[0]] = torch.utils.data.DataLoader(
                self.datasets.dataset_train[self.classes[0]],
                batch_size=self.opt.batch_size,
                shuffle=True,
                num_workers=int(self.opt.workers)
            )
            self.datasets.dataloader_train[self.classes[1]] = torch.utils.data.DataLoader(
                self.datasets.dataset_train[self.classes[1]],
                batch_size=self.opt.batch_size,
                shuffle=True,
                num_workers=int(self.opt.workers)
            )

            self.datasets.dataloader_test = {}
            self.datasets.dataloader_test[self.classes[0]] = torch.utils.data.DataLoader(
                self.datasets.dataset_test[self.classes[0]],
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=int(self.opt.workers)
            )
            self.datasets.dataloader_test[self.classes[1]] = torch.utils.data.DataLoader(
                self.datasets.dataset_test[self.classes[1]],
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=int(self.opt.workers)
            )

            axis = []
            if self.opt.data_augmentation_axis_rotation:
                axis = [1]

            flips = []
            if self.opt.data_augmentation_random_flips:
                flips = [0, 2]

            # Create Data Augmentation
            self.datasets.data_augmenter = augmenter.Augmenter(translation=self.opt.random_translation,
                                                               rotation_axis=axis,
                                                               anisotropic_scaling=self.opt.anisotropic_scaling,
                                                               rotation_3D=self.opt.random_rotation,
                                                               flips=flips)

            self.datasets.len_dataset = {
                self.classes[0]: len(self.datasets.dataset_train[self.classes[0]]),
                self.classes[1]: len(self.datasets.dataset_train[self.classes[1]])
            }
            self.datasets.min_len_dataset = min(self.datasets.len_dataset[self.classes[0]],
                                                self.datasets.len_dataset[self.classes[1]])

            self.datasets.len_dataset_test = {
                self.classes[0]: len(self.datasets.dataset_test[self.classes[0]]),
                self.classes[1]: len(self.datasets.dataset_test[self.classes[1]])
            }
            self.datasets.min_len_dataset_test = min(self.datasets.len_dataset_test[self.classes[0]],
                                                     self.datasets.len_dataset_test[self.classes[1]])

