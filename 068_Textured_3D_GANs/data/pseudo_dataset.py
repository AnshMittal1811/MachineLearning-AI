import torch
import torch.nn.functional as F
import os
import numpy as np
import glob

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, args, augment=True):
        self.args = args
        self.cache_dir = os.path.join('cache', args.dataset)
    
        self.data = np.load(os.path.join(self.cache_dir, f'poses_metadata_{args.mode}.npz'), allow_pickle=True)
        self.data = self.data['data'].item()
        num_images = len(self.data['path'])
        self.augment = augment
        
        # Check if category information is available
        if 'category' in self.data:
            self.classes = torch.LongTensor(self.data['category'].unsqueeze(1))
            self.n_classes = (self.classes.max().item()+1,)
            print('# classes:', self.n_classes)

            # Compute sample weights for balancing
            class_lengths = [(self.classes == k).sum().item() for k in range(self.n_classes[0])]
            print('Class sizes:', class_lengths)
            self.class_weights = torch.FloatTensor(class_lengths)
            self.class_weights = self.class_weights.sum() / (self.class_weights * len(self.class_weights))
            print('Class weights:', self.class_weights)
            self.image_weights = []
            for i, weight in enumerate(self.class_weights):
                weight = weight.item()
                self.image_weights += [weight]*class_lengths[i]
            assert len(self.image_weights) == len(self)

        else:
            self.n_classes = (1,)
        args.n_classes = self.n_classes

        
        # Check if pseudo-ground-truth is available
        pseudogt_files = glob.glob(os.path.join(self.cache_dir,
                                   f'pseudogt_{args.texture_resolution}x{args.texture_resolution}_{args.mode}',
                                   '*.npz'))
        if len(pseudogt_files) == 0:
            print('Pseudo-ground-truth not found, only "Full FID" evaluation is available')
            self.has_pseudogt = False
        elif len(pseudogt_files) == num_images:
            print(f'Pseudo-ground-truth found! ({len(pseudogt_files)} images)')
            self.has_pseudogt = True
        else:
            raise ValueError('Found pseudo-ground-truth directory, but number of files does not match! '
                            f'Expected {num_images}, got {len(pseudogt_files)}. '
                             'Please check your dataset setup.')
            
        if not self.has_pseudogt and not args.evaluate:
            raise ValueError('Training a model requires the pseudo-ground-truth to be setup beforehand.')
    
        
    def suggest_truncation_sigma(self):
        return 1.0
        
    def suggest_num_discriminators(self):
        if self.args.texture_resolution >= 512:
            return 3
        else:
            return 2

        
    def suggest_mesh_template(self):
        return 'mesh_templates/uvsphere_31rings.obj'
    
    def __len__(self):
        return len(self.data['path'])
    
    def _load_pseudogt(self, idx, train=False):
        tex_res = self.args.texture_resolution
        with np.load(os.path.join(self.cache_dir,
                       f'pseudogt_{tex_res}x{tex_res}_{self.args.mode}',
                       f'{idx}.npz'), allow_pickle=True) as fdata:
            data = fdata['data'].item()
        
        gt_dict = {
            'texture': data['texture'].float(),
            'texture_alpha': data['texture_alpha'].float(),
            'mesh': data['mesh']
        }
        
        if not train:
            gt_dict['image'] = data['image'][:3].float()/2 + 0.5
        
        if self.args.conditional_semantics:
            assert 'seg' in data
            gt_dict['seg'] = F.softmax(data['seg']*100, dim=0) # Argmax (one-hot vector)
        
        return gt_dict
    
    def __getitem__(self, idx):
        gt_dict = self._load_pseudogt(idx, train=True)
        
        # "Virtual" mirroring in UV space
        # A very simple form of data augmentation that does not require re-rendering
        if self.augment and not self.args.evaluate:
            if torch.randint(0, 2, size=(1,)).item() == 1:
                for k, v in gt_dict.items():
                    gt_dict[k] = PseudoDataset.mirror_tex(v)
        
        if self.args.conditional_class:
            gt_dict['class'] = self.classes[idx]
        
        gt_dict['idx'] = idx
        return gt_dict
    
    @staticmethod
    def mirror_tex(tr):
        # "Virtually" flip a texture or displacement map of shape (nc, H, W)
        # This is achieved by mirroring the image and shifting the u coordinate,
        # which is consistent with reprojecting the mirrored 2D image.
        tr = torch.flip(tr, dims=(2,))
        tr = torch.cat((tr, tr), dim=2)
        tr = tr[:, :, tr.shape[2]//4:-tr.shape[2]//4]
        return tr
    



class PseudoDatasetForEvaluation(torch.utils.data.Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        gt_dict = {
            'scale': self.dataset.data['scale'][idx],
            'translation': self.dataset.data['translation'][idx],
            'rotation': self.dataset.data['rotation'][idx],
            'idx': idx,
        }
        
        if self.dataset.args.conditional_class:
            gt_dict['class'] = self.dataset.classes[idx]
        
        if self.dataset.args.conditional_text:
            gt_dict['caption'] = self.dataset.index_captions[idx] # Tuple (padded tokens, lengths)
            
        if self.dataset.has_pseudogt:
            # Add pseudo-ground-truth entries
            gt_dict.update(self.dataset._load_pseudogt(idx))

        return gt_dict