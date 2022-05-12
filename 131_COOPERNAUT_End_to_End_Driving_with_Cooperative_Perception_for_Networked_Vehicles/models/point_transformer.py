import torch
import torch.nn as nn
import random
import open3d.ml.torch as ml3d
from .pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from .transformer_block import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        
simple=False
class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = config.npoints, config.nblocks, config.nneighbor, config.num_output, config.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, config.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        if not simple:
            for i in range(nblocks):
                channel = 32 * 2 ** (i + 1)
                self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
                self.transformers.append(TransformerBlock(channel, config.transformer_dim, nneighbor))
        else:
            for i in range(nblocks):
                channel = 32
                self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel + 3, channel, channel]))
                self.transformers.append(TransformerBlock(channel, config.transformer_dim, nneighbor))

        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            #from IPython import embed; embed()
            #xyz_and_feats.append((xyz, points))
        return points, xyz, xyz_and_feats

class PointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Backbone(config)
        npoints, nblocks, nneighbor, n_c, d_points = config.npoints, config.nblocks, config.nneighbor, config.num_output, config.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks+8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
        dim_meta = 1
        self.fc_meta = nn.Sequential(
            nn.Linear(dim_meta, 16),
            nn.ReLU(),
            nn.Linear(16,8)
            )
    def forward(self, ego_lidar, ego_speed):
        points, xyz, _ = self.backbone(ego_lidar)
        ego_meta = self.fc_meta(ego_speed[:,None])
        #points = points.mean(1)
        points = points.max(1).values
        concat_output = torch.cat((points, ego_meta),dim=-1)
        pred_control = self.fc2(concat_output)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]

        return pred_throttle, pred_brake, pred_steer

class CooperativePointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Backbone(config)
        self.backbone_other = Backbone(config)
        #self.backbone_other.register_backward_hook(hook_fn)
        npoints, nblocks, nneighbor, n_c, d_points = config.npoints, config.nblocks, config.nneighbor, config.num_output, config.input_dim
        if not simple:
            channel = 32 * 2 ** (nblocks+1)
            self.aggr_transition_down = TransitionDown(npoints // 4 ** (nblocks+1) * 2, nneighbor, [channel //2 +3, channel, channel]) #(1+config.max_num_neighbors),
        else:
            channel = 32
            self.aggr_transition_down = TransitionDown(npoints // 4 ** (nblocks+1) * 2, nneighbor, [channel +3, channel, channel]) #(1+config.max_num_neighbors),
        self.aggr_transformer = TransformerBlock(channel , config.transformer_dim, nneighbor)#channel was  divided  by 2
        if not simple:
            self.fc2 = nn.Sequential(
                nn.Linear(32 * 2 ** (nblocks+1) + 8, 256), # was  divided by 2 , was +8
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, n_c)
            )
        else:
            self.fc2 = nn.Sequential(
                nn.Linear(channel + 8, 256), # was  divided by 2 , was +8
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, n_c)
            )

        self.nblocks = nblocks
        self.npoints = npoints
        dim_meta = 1
        self.fc_meta = nn.Sequential(
            nn.Linear(dim_meta, 16),
            nn.ReLU(),
            nn.Linear(16,8)
            )
        self.voxel_pooling = ml3d.layers.VoxelPooling(position_fn='average', feature_fn='max')

    def forward(self, ego_lidar, ego_speed, other_lidar, other_transform):
        
        points, xyz, _ = self.backbone(ego_lidar)
        other_lidars = torch.split(other_lidar, 1, dim=1)
        other_transforms = torch.split(other_transform, 1, dim=1)
        
        for i in range(len(other_lidars)):
            lidar = torch.squeeze(other_lidars[i], dim=1)
            trans = torch.squeeze(other_transforms[i], dim=1)
            other_points, other_xyz, _ = self.backbone_other(lidar)
            # Transform the xyz from other frame into ego frame
            other_xyz = torch.transpose(torch.matmul(trans[:,:3,:3],torch.transpose(other_xyz, 1, 2)), 1,2)+torch.cat(other_points.shape[1]*[trans[:,:3,3].unsqueeze(1)],dim=1)
            if i==0:
                others_xyz = other_xyz
                others_points = other_points
            else:
                others_xyz = torch.cat((others_xyz, other_xyz), dim=1)
                others_points = torch.cat((others_points, other_points), dim=1)
        #Filter Out points outside x[-70,70] y[-70,70], z[-2, -0.25]
        #mask = (others_xyz[:,:,0]>-70)&(others_xyz[:,:,0]<70)&(others_xyz[:,:,1]>-70)&(others_xyz[:,:,1]<70)&(others_xyz[:,:,2]>-2)&(others_xyz[:,:,2]<-0.25)        
        #from IPython import embed; embed() 
        #others_xyz = others_xyz[mask]
        #others_points = others_points[mask]
        #from IPython import embed; embed()
        for i in range(ego_lidar.size(0)): #Batch dimension
            pooled_voxels = self.voxel_pooling(others_xyz[i].to('cpu'), others_points[i].to('cpu'), voxel_size=0.5)
            o_xyz = pooled_voxels.pooled_positions
            o_points = pooled_voxels.pooled_features
            #Filter Out points outside x[-70,70] y[-70,70], z[-2, -0.25]
            mask = (o_xyz[:,0]>=-70)&(o_xyz[:,0]<=70)&(o_xyz[:,1]>=-70)&(o_xyz[:,1]<=70)&(o_xyz[:,2]>-2)&(o_xyz[:,2]<=0.25)        
            #from IPython import embed; embed()
            o_xyz = o_xyz[mask]
            o_points = o_points[mask]

            desired_npoints = xyz.shape[1]#self.npoints // 4 ** (self.nblocks+1) * 2
            if o_xyz.size(0) < desired_npoints:
                #print(o_xyz)
                indice = list(range(max(o_xyz.size(0),1)))
                indice.extend(list(random.choices(list(range(max(o_xyz.size(0),1))), k=int(desired_npoints-o_xyz.size(0)))))
                #print(indice)
            else:
                indice = random.sample(range(o_xyz.size(0)), desired_npoints)
                #print(indice)
            indice = torch.tensor(indice)
            if len(o_xyz.shape) < 3:
                try:
                    o_xyz = o_xyz[indice].unsqueeze(0)
                    o_points = o_points[indice].unsqueeze(0)
                except:
                    from IPython import embed; embed()
            if i==0:
                subsample_others_points = o_points
                subsample_others_xyz = o_xyz
            else:
                try:
                    subsample_others_points = torch.cat((subsample_others_points,o_points), dim=0)
                    subsample_others_xyz = torch.cat((subsample_others_xyz,o_xyz), dim=0)
                except:
                    from IPython import embed; embed()
                    print(o_points)
                    print("===")
                    print(subsample_others_points)
        subsample_others_points = subsample_others_points.to('cuda')
        subsample_others_xyz = subsample_others_xyz.to('cuda')

        points = torch.cat((points, subsample_others_points), dim=1)
        xyz = torch.cat((xyz, subsample_others_xyz), dim=1)
        
        # How to handle the dynamic number of points?
        xyz, points = self.aggr_transition_down(xyz, points)
        #from IPython import embed; embed()
        points = self.aggr_transformer(xyz, points)[0]
        ego_meta = self.fc_meta(ego_speed[:,None])
        #points = points.mean(1)
        points = points.max(1).values
        concat_output = torch.cat((points, ego_meta),dim=-1)
        pred_control = self.fc2(concat_output)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]

        return pred_throttle, pred_brake, pred_steer

class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        pred_control = self.fc2(points.mean(1))
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]

        return pred_throttle, pred_brake, pred_steer


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, config.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, config.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
            
        return self.fc3(points)


def hook_fn(m, i, o):
  print(m)
  print("------------Input Grad------------")

  for grad in i:
    try:
      print(grad.shape)
    except AttributeError: 
      print ("None found for Gradient")

  print("------------Output Grad------------")
  for grad in o:  
    try:
      print(grad.shape)
    except AttributeError: 
      print ("None found for Gradient")
  print("\n")    
