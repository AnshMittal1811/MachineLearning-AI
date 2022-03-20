import torch
import torch.nn as nn
import torch.nn.functional as F

from rendering.utils import circpad, symmetrize_texture, adjust_poles

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, norm_g, ch, emb_dim):
        super().__init__()
        
        if emb_dim is None or emb_dim == 0:
            # Disable (use standard BN)
            self.affine = True
        else:
            # Enable
            self.affine = False
        
        if norm_g == 'syncbatch':
            from sync_batchnorm import SynchronizedBatchNorm1d
            self.norm = SynchronizedBatchNorm1d(ch, affine=self.affine)
        elif norm_g == 'batch':
            self.norm = nn.BatchNorm1d(ch, affine=self.affine)
        elif norm_g == 'instance':
            self.norm = nn.InstanceNorm1d(ch, affine=self.affine)
        elif norm_g == 'none':
            self.norm = lambda x: x # Identity
        else:
            raise
            
        if not self.affine:
            self.fc_gamma = nn.Embedding(emb_dim, ch)
            self.fc_gamma.weight.data[:] = 1
            self.fc_beta = nn.Embedding(emb_dim, ch)
            self.fc_beta.weight.data[:] = 0
        
    def forward(self, x, z):
        x = self.norm(x)
        if not self.affine:
            gamma = self.fc_gamma(z)
            beta = self.fc_beta(z)
            return x * gamma + beta
        else:
            return x

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, norm_g, ch, emb_dim):
        super().__init__()
        
        if emb_dim is None or emb_dim == 0:
            # Disable (use standard BN)
            self.affine = True
        else:
            # Enable
            self.affine = False
        
        if norm_g == 'syncbatch':
            from sync_batchnorm import SynchronizedBatchNorm2d
            self.norm = SynchronizedBatchNorm2d(ch, affine=self.affine)
        elif norm_g == 'batch':
            self.norm = nn.BatchNorm2d(ch, affine=self.affine)
        elif norm_g == 'instance':
            self.norm = nn.InstanceNorm2d(ch, affine=self.affine)
        elif norm_g == 'none':
            self.norm = lambda x: x # Identity
        else:
            raise
            
        if not self.affine:
            self.fc_gamma = nn.Embedding(emb_dim, ch)
            self.fc_gamma.weight.data[:] = 1
            self.fc_beta = nn.Embedding(emb_dim, ch)
            self.fc_beta.weight.data[:] = 0
        
    def forward(self, x, z):
        x = self.norm(x)
        if not self.affine:
            gamma = self.fc_gamma(z).unsqueeze(-1).unsqueeze(-1)
            beta = self.fc_beta(z).unsqueeze(-1).unsqueeze(-1)
            return x * gamma + beta
        else:
            return x
    
class ResBlock(nn.Module):
    def __init__(self, norm_g, ch_in, ch_out, class_dim, pad_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_in, 3, padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(ch_in, ch_out, 3, padding=(1, 0), bias=False)
        self.bn1 = ConditionalBatchNorm2d(norm_g, ch_in, class_dim)
        self.bn2 = ConditionalBatchNorm2d(norm_g, ch_out, class_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pad_fn = pad_fn
        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        else:
            self.shortcut = lambda x: x
        
    def forward(self, x, cl):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(self.pad_fn(x, 1)), cl))
        x = self.relu(self.bn2(self.conv2(self.pad_fn(x, 1)), cl))
        return x + shortcut
    

class ReconstructionNetwork(nn.Module):
    def __init__(self, num_classes, test_mode, symmetric=True, texture_res=64, mesh_res=32, interpolation_mode='nearest',
                 prediction_type='texture', num_parts=None, norm_g='syncbatch'):
        super().__init__()
        
        self.symmetric = symmetric
        self.test_mode = test_mode
        assert prediction_type in ['none', 'texture', 'semantics', 'both']
        self.prediction_type = prediction_type
        if prediction_type in ['semantics', 'both']:
            assert num_parts is not None
            print('Enabled semantics, num parts:', num_parts)
        
        if symmetric:
            self.pad = lambda x, amount: F.pad(x, (amount, amount, 0, 0), mode='replicate')
        else:
            self.pad = lambda x, amount: circpad(x, amount)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.num_classes = num_classes
        if num_classes > 1:
            class_dim = num_classes
        else:
            class_dim = None
        
        if interpolation_mode == 'nearest':
            self.up = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        elif interpolation_mode == 'bilinear':
            self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            raise
            
        assert mesh_res >= 32
        assert texture_res >= 64
        
        self.conv1e = nn.Conv2d(4, 64, 5, stride=2, padding=2, bias=False) # 128 -> 64
        self.bn1e = ConditionalBatchNorm2d(norm_g, 64, class_dim)
        self.conv2e = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False) # 64 > 32
        self.bn2e = ConditionalBatchNorm2d(norm_g, 128, class_dim)
        self.conv3e = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False) # 32 -> 16
        self.bn3e = ConditionalBatchNorm2d(norm_g, 256, class_dim)
        self.conv4e = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False) # 16 -> 8
        self.bn4e = ConditionalBatchNorm2d(norm_g, 512, class_dim)
        
        bottleneck_dim = 256
        self.conv5e = nn.Conv2d(512, 64, 3, stride=2, padding=1, bias=False) # 8 -> 4
        self.bn5e = ConditionalBatchNorm2d(norm_g, 64, class_dim)
        self.fc1e = nn.Linear(64*8*8, bottleneck_dim, bias=False)
        self.bnfc1e = ConditionalBatchNorm1d(norm_g, bottleneck_dim, class_dim)
            
        self.fc3e = nn.Linear(bottleneck_dim, 1024, bias=False)
        self.bnfc3e = ConditionalBatchNorm1d(norm_g, 1024, class_dim)
        
        # Texture/mesh generation
        self.base_res_h = 4
        self.base_res_w = 2 if symmetric else 4
            
        self.fc1_tex = nn.Linear(1024, self.base_res_h*self.base_res_w*256)
        self.blk1 = ResBlock(norm_g, 256, 512, class_dim, self.pad) # 4 -> 8
        self.blk2 = ResBlock(norm_g, 512, 256, class_dim, self.pad) # 8 -> 16
        self.blk3 = ResBlock(norm_g, 256, 256, class_dim, self.pad) # 16 -> 32 (k=1)
        
        if prediction_type != 'none':
            assert texture_res in [64, 128, 256]
            self.texture_res = texture_res
            if texture_res >= 128:
                self.blk3b_tex = ResBlock(norm_g, 256, 256, class_dim, self.pad) # k = 2
            if texture_res >= 256:
                self.blk3c_tex = ResBlock(norm_g, 256, 256, class_dim, self.pad) # k = 4

            self.blk4_tex = ResBlock(norm_g, 256, 128, class_dim, self.pad) # k*32 -> k*64
            self.blk5_tex = ResBlock(norm_g, 128, 64, class_dim, self.pad) # k*64 -> k*64 (no upsampling)
            if prediction_type in ['texture', 'both']:
                self.conv_tex = nn.Conv2d(64, 3, 5, padding=(2, 0))
            if prediction_type in ['semantics', 'both']:
                self.conv_seg = nn.Conv2d(64, num_parts, 5, padding=(2, 0))
        
        # Mesh generation
        self.blk4_mesh = ResBlock(norm_g, 256, 64, class_dim, self.pad) # 32 -> 32 (no upsampling)
        self.conv_mesh = nn.Conv2d(64, 3, 5, padding=(2, 0))
        
        # Zero-initialize mesh output layer for stability (avoids self-intersections)
        self.conv_mesh.bias.data[:] = 0
        self.conv_mesh.weight.data[:] = 0
            
        total_params = 0
        for param in self.parameters():
            total_params += param.nelement()
        print('Model parameters: {:.2f}M'.format(total_params/1000000))
    
    def random_flip(self, img):
        # In-model data augmentation
        with torch.no_grad():
            img_flipped = torch.flip(img, dims=(3,))
            flip_mask = torch.randint(0, 2, size=(img.shape[0],), device=img.device).float()
            lerped = torch.lerp(img, img_flipped, flip_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        return lerped
    
    def forward(self, x, C, M):
        
        if not self.test_mode:
            x = self.random_flip(x)
            
        if self.num_classes > 1:
            cl = C
        else:
            cl = None
        
        # Generate latent code
        x = self.relu(self.bn1e(self.conv1e(x), cl))
        x = self.relu(self.bn2e(self.conv2e(x), cl))
        x = self.relu(self.bn3e(self.conv3e(x), cl))
        x = self.relu(self.bn4e(self.conv4e(x), cl))
        x = self.relu(self.bn5e(self.conv5e(x), cl))
        
        x = x.view(x.shape[0], -1) # Flatten
        z = self.relu(self.bnfc1e(self.fc1e(x), cl))
        z = self.relu(self.bnfc3e(self.fc3e(z), cl))
        
        bb = self.fc1_tex(z).view(z.shape[0], -1, self.base_res_h, self.base_res_w)
        bb = self.up(self.blk1(bb, cl))
        bb = self.up(self.blk2(bb, cl))
        bb = self.up(self.blk3(bb, cl))
        bb_mesh = bb
        mesh_map = self.blk4_mesh(bb_mesh, cl)
        mesh_map = self.conv_mesh(self.pad(self.relu(mesh_map), 2))
        mesh_map = adjust_poles(mesh_map)
        
        tex, seg = None, None
        if self.prediction_type != 'none':
            if self.texture_res >= 128:
                bb = self.up(self.blk3b_tex(bb, cl))
            if self.texture_res >= 256:
                bb = self.up(self.blk3c_tex(bb, cl))

            tex = self.up(self.blk4_tex(bb, cl))
            tex = self.blk5_tex(tex, cl)
            tex = self.pad(self.relu(tex), 2)
            tex_ = tex
            tex = None
            if self.prediction_type in ['semantics', 'both']:
                softmax_mask = (1 - M.sum(dim=-2)) * -10000
                seg = F.softmax(self.conv_seg(tex_) + softmax_mask.unsqueeze(-1).unsqueeze(-1), dim=1)
            if self.prediction_type in ['texture', 'both']:
                tex = self.conv_tex(tex_).tanh_()
        
        if self.symmetric:
            if self.prediction_type in ['texture', 'both']:
                tex = symmetrize_texture(tex)
            if self.prediction_type in ['semantics', 'both']:
                seg = symmetrize_texture(seg)
            mesh_map = symmetrize_texture(mesh_map)      

        return tex, mesh_map, seg