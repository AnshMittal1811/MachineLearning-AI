import torch
import numpy as np

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, cmr_dataset, img_size):
        self.cmr_dataset = cmr_dataset
        self.paths = cmr_dataset.get_paths()
        
        self.extra_img_keys = []
        if isinstance(img_size, list):
            for res in img_size[1:]:
                self.extra_img_keys.append(f'img_{res}')

    def __len__(self):
        return len(self.cmr_dataset)

    def __getitem__(self, idx):
        item = self.cmr_dataset[idx]

        # Rescale img to [-1, 1]
        img = item['img'].astype('float32')*2 - 1
        mask = item['mask'].astype('float32')
        seg = item['seg'].astype('float32')
        img *= mask[np.newaxis, :, :]

        img = torch.FloatTensor(img)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        seg = torch.FloatTensor(seg)
        ind = torch.LongTensor([idx])
        if item['mirrored']:
            # Indices from 0 to N-1 are straight, from N to 2N-1 are mirrored
            ind += len(self.cmr_dataset)

        scale = torch.FloatTensor(item['sfm_pose'][:1])
        translation = torch.FloatTensor([item['sfm_pose'][1], item['sfm_pose'][2], 0])
        rot = torch.FloatTensor(item['sfm_pose'][-4:])
        z0 = torch.FloatTensor(item['z0'])
        w = torch.FloatTensor(item['w'])
        semi_mask = torch.FloatTensor(item['semi_mask'])

        output = torch.cat((img, mask), dim=0)
        
        extra_imgs = []
        for k in self.extra_img_keys:
            img_k, mask_k = item[k]
            img_k = img_k.astype('float32')*2 - 1
            mask_k = mask_k.astype('float32')[np.newaxis, :, :]
            img_k *= mask_k
            img_k = torch.FloatTensor(img_k)
            mask_k = torch.FloatTensor(mask_k)
            extra_imgs.append(torch.cat((img_k, mask_k), dim=0))

        return (output, seg, *extra_imgs, scale, translation, rot, z0, w, semi_mask, ind)
    
    
# If the total number of images is not divisible by the batch size,
# this sampler ensures that the last "batch" of images is split
# into smaller batches of size 1, in order to avoid issues with multi-gpu settings
class AdjustedBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            for elem in batch:
                yield [ elem ]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            base_len = len(self.sampler) // self.batch_size
            base_len += len(self.sampler) % self.batch_size
            return base_len