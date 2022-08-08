import torch

from inference_memory_bank import MemoryBank
from model.eval_network import PCVOS
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by
import numpy as np

class InferenceCore_Per_Clip:
    def __init__(self, prop_net:PCVOS, images, num_objects, top_k=20, 
                    mem_every=5, include_last=False, req_frames=None, clip_length=1, overlap_length=0):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last
        self.clip_length = clip_length
        self.overlap_length = overlap_length

        # We HAVE to get the output for these frames
        # None if all frames are required
        self.req_frames = req_frames

        self.top_k = top_k

        # True dimensions
        b = images.shape[0]
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.b, self.t, self.h, self.w = b, t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        # list of objects with usable memory
        self.enabled_obj = []

        self.mem_banks = dict()

        self.processed_frames = []

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].cuda())
        return result

    # Task list of index as idx
    def encode_key_batch(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].flatten(start_dim=0, end_dim=1).cuda())
        result = [x.view(self.b, len(idx), *x.shape[-3:]) for x in result]
        return result

    def do_pass(self, key_k, key_v, idx, end_idx):
        closest_ti = end_idx

        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_banks:
                self.mem_banks[oi] = MemoryBank(k=1, top_k=self.top_k, refine_clip=self.prop_net.refine_clip)
            self.mem_banks[oi].add_memory(key_k, key_v[i:i+1])

        last_ti = idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        step = +1
        end = closest_ti - 1

        if self.req_frames is None:
            this_req_frames = np.array(list(range(idx+1, closest_ti)))
        else:
            # Note: Assume the mem_every is multiple of 5 (interval of gt)
            #       Thus, we only consider req frames
            np_req_frames = np.array(self.req_frames)
            this_idx = (np_req_frames > idx) & (np_req_frames < closest_ti)
            this_req_frames = np_req_frames[this_idx]
        
        # list of first frame index of each clip
        B_first_list = list(range(0, len(this_req_frames), self.clip_length))
        B_len = len(B_first_list)
        B_end = B_len - 1
        
        for B_idx in range(B_len):
            if B_idx != B_len - 1:
                ti = this_req_frames[list(range(B_first_list[B_idx], B_first_list[B_idx+1]))]
                aug_ti = this_req_frames[list(range(B_first_list[B_idx], min(B_first_list[B_idx+1]+self.overlap_length, len(this_req_frames))))]
            else:
                ti = this_req_frames[list(range(B_first_list[B_idx], len(this_req_frames)))]
                aug_ti = ti

            self.processed_frames.append(ti)

            k16, qv16, qf16, qf8, qf4 = self.encode_key_batch(aug_ti)
            k16_flatten = k16.transpose(1,2).flatten(start_dim=2, end_dim=3)
            
            # After this step all keys will have the same size
            out_mask = torch.cat([
                self.prop_net.segment_with_query_per_clip(self.mem_banks[oi], qf16, qf8, qf4, k16_flatten, qv16, len(aug_ti))
            for oi in self.enabled_obj], 0)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[0,aug_ti] = out_mask[0]
            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,aug_ti] = out_mask[i+1]
            
            for fid in range(len(ti)):
                is_mem_frame = (abs(ti[fid]-last_ti) >= self.mem_every)
                is_last_frame = (fid == len(ti)-1)
                if (self.include_last and is_last_frame) or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:,ti[fid]].cuda(), qf16[:,fid], out_mask[1:,fid])
                    prev_key = k16[:,fid].unsqueeze(2)
                    for i, oi in enumerate(self.enabled_obj):
                        self.mem_banks[oi].add_memory(prev_key, prev_value[i:i+1], is_temp=not is_mem_frame)

                    if is_mem_frame:
                        last_ti = ti[fid]

        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)
        
        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx]
        
        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].cuda(), qf16, self.prob[self.enabled_obj,frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)
