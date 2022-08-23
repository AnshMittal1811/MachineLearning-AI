import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import torch.nn.functional as F
import torchvision.models as models
import trimesh
from model.model_blocks import get_num_ada_norm_params


class ODEFunc(nn.Module):
    '''
    This refers to the dynamics function f(x,t) in a IVP defined as dh(x,t)/dt = f(x,t).
    For a given location (t) on point (x) trajectory, it returns the direction of 'flow'.
    Refer to Section 3 (Dynamics Equation) in the paper for details.
    '''

    def __init__(self, num_hidden=512, latent_len=512):
        '''
        Initialization.
        num_hidden: number of nodes in a hidden layer
        latent_len: size of the latent code being used
        '''

        super(ODEFunc, self).__init__()

        self.l1 = nn.Linear(3, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_hidden)
        self.l3 = nn.Linear(num_hidden, num_hidden)
        self.l4 = nn.Linear(num_hidden, 3)

        self.cond = nn.Linear(latent_len, num_hidden)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.nfe = 0
        self.zeros = torch.zeros((1, 1, latent_len))
        self.latent_dyn_zeros = None

    #         self.tw = nn.Linear(1, num_hidden)

    def forward(self, t, xz):
        '''
        t: Torch tensor of shape (1,)
        xz: Torch tensor of shape (N, #pts, 3+zdim). Along dimension 2, the point and shape embeddings are concatenated.

        **NOTE**
        For the uniqueness property to hold, a single dynamics function (operating in 3D) must be used to compute
        trajectories pertaining to points of a single shape.

        Here, the shape encoding (same for all points of a shape) is used to choose a function which is applied over all the shape points.
        Hence, even though the input xz appears to be a 3+zdim dimensional state, the ODE is still restricted to a 3D state-space.
        The concatenation is purely to make programming simpler without affecting the underlying theory.

        '''
        point_features = self.relu(self.l1((xz[..., :3])))  # Extract point features Nx#ptsx3 -> Nx#ptsx512
        shape_features = self.tanh(self.cond(xz[..., 3:]))  # Extract shape features Nx#ptsxzdim -> Nx#ptsx512

        point_shape_features = point_features * shape_features  # Compute point-shape features by elementwise multiplication
        # [Insight :]  Conditioning is critical to allow for several shapes getting learned by same NeuralODE.
        #              Note that under current formulation, all points belonging to a shape share a common dynamics function.

        # Two residual blocks
        point_shape_features = self.relu(self.l2(point_shape_features)) + point_shape_features
        point_shape_features = self.relu(self.l3(point_shape_features)) + point_shape_features
        # [Insight :] Using less residual blocks leads to drop in performance
        #             while more residual blocks make model heavy and training slow due to more complex trajectories being learned.

        dyns_x_t = self.tanh(self.l4(point_shape_features))  # Computed dynamics of point x at time t
        # [Insight :] We specifically choose a tanh activation to get maximum expressivity as observed by He, et.al and Massaroli, et.al

        self.nfe += 1  # To check #ode evaluations

        # To prevent updating of latent codes during ODESolver calls, we simply make their dynamics all zeros.
        if (self.latent_dyn_zeros is None or
                self.latent_dyn_zeros.shape[0] != dyns_x_t.shape[0] or
                self.latent_dyn_zeros.shape[1] != dyns_x_t.shape[1]):
            self.latent_dyn_zeros = self.zeros.repeat(dyns_x_t.shape[0], dyns_x_t.shape[1], 1).type_as(dyns_x_t)

        return torch.cat([dyns_x_t, self.latent_dyn_zeros],
                         dim=2)  # output is therefore like [dyn_x, dyn_y, dyn_z, 0,0..,0] for a point


class NODEBlock(nn.Module):
    '''
    Function to solve an IVP defined as dh(x,t)/dt = f(x,t).
    We use the differentiable ODE Solver by Chen et.al used in their NeuralODE paper.
    '''

    def __init__(self, odefunc, tol):
        '''
        Initialization.
        odefunc: The dynamics function to be used for solving IVP
        tol: tolerance of the ODESolver
        '''
        super(NODEBlock, self).__init__()
        self.odefunc = odefunc
        self.cost = 0
        self.rtol = tol
        self.atol = tol

    def forward(self, x, time):
        '''
        Solves the ODE in the forward time.
        '''
        self.odefunc.nfe = 0  # To check #ode evaluations
        self.forward_time = torch.tensor([0, time]).float().type_as(
            x)  # Time of integration (must be monotinically increasing!)
        # Solve the ODE with initial condition x and interval time.
        out = odeint(self.odefunc.to(x), x, self.forward_time, rtol=self.rtol, atol=self.atol)
        self.cost = self.odefunc.nfe  # Number of evaluations it took to solve it
        return out[1]

    def invert(self, x, time):
        '''
        Solves the ODE in the reverse time.
        '''
        self.inverse_time = torch.tensor([time, 0]).float().type_as(x)
        out = odeint(self.odefunc, x, self.inverse_time, rtol=self.rtol, atol=self.atol)
        self.cost = self.odefunc.nfe
        return out[1]


class DeformBlock(nn.Module):
    '''
    A single DeformBlock is made up of two NODE Blocks. Refer secion 3 (Overall Architecture)
    '''

    def __init__(self, time=0.2, num_hidden=512, latent_len=512, tol=1e-5):
        super(DeformBlock, self).__init__()
        '''
        Initialization.
        time: some number 0-1
        num_hidden: Number of hidden nodes in the MLP of dynamics
        latent_len: Length of shape embeddings
        tol: tolerance of the ODE Solver
        '''

        # Two NODE Blocks
        self.l1 = NODEBlock(ODEFunc(num_hidden, latent_len), tol=tol)
        self.l2 = NODEBlock(ODEFunc(num_hidden, latent_len), tol=tol)

        self.time = time

    def forward_(self, x, code, time=None):
        '''
        Forward flow method

        x: BxNx3 input tensor
        code: Bxzdim tensor embedding
        time: some number 0-1

        y: BxNx3 output tensor
        '''

        if time is None:
            time = self.time

        xz = torch.cat([x, code.repeat(1, x.shape[1], 1)], dim=2)
        # Note: To enable condioned flows, we concatenate points with their corresponding shape embeddings.
        #       Refer to code comments in ODEFunc.forward() for more details about this choice.

        x = self.l1(xz, time)
        x = self.l2(x, time)

        y = x[..., :3]  # output the corresponding 'flown' points.
        return y

    def backward_(self, x, code, time=None):
        '''
        Backward flow method

        x: BxNx3 input tensor
        code: Bxzdim tensor embedding
        time: some number 0-1

        y: BxNx3 output tensor

        **NOTE** We do not use this method in the main NMF pipeline, but may come handy for things like inverting the NMF!

        '''
        if time is None:
            time = self.time

        xz = torch.cat([x, code.repeat(1, x.shape[1], 1)], dim=2)
        # Note: To enable condioned flows, we concatenate points with their corresponding shape embeddings.
        #       Refer to code comments in ODEFunc.forward() for more details about this choice.

        x = self.l2.invert(xz, time)
        x = self.l1.invert(x, time)

        y = x[..., :3]

        return y

    def forward(self, code, x, y=None, time=None):
        '''
        code: Bxzdim tensor embedding
        x: BxNx3 input tensor
        y: BxNx3 output tensor
        time: some number 0-1
        '''

        # Calculate forward flow
        pred_y = self.forward_(x, code, time)

        if y is not None:
            # Calculate backward flow if required
            pred_x = self.backward_(y, code, time)
            return pred_y, pred_x

        return pred_y, None


class InstanceNorm(nn.Module):
    '''
    Instance Normalization. Refer Section 3 in paper (Instance Normalization) for more details and Fig 5 for visual intuition.
    This is crucial for cross-category training.
    '''

    def __init__(self, zdim):
        super(InstanceNorm, self).__init__()
        '''
        Initialization.
        '''
        self.num_features = 3  # TODO(msegu): does it make sense to normalize only on channels?
        self.norm = nn.Sequential(nn.Linear(zdim, 256), nn.ReLU(), nn.Linear(256, 3), nn.Sigmoid()).float()

    def forward(self, input, code):
        '''
        input: point cloud of shape BxNx3
        code: shape embedding of shape Bx1xzdim
        '''
        centered_input = input - torch.mean(input, axis=1, keepdim=True)  # Center the point cloud
        centered_input = centered_input * (
                    1 - self.norm(code))  # anisotropic scaling the point cloud w.r.t. the shape embedding

        return centered_input


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, zdim):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = InstanceNorm(zdim)

    def forward(self, input, content_code, params):
        a = params[:, :params.size(1) // 2].unsqueeze(1)
        b = params[:, params.size(1) // 2:].unsqueeze(1)
        return a + b * self.norm(input, content_code)


'''
The following is adapted from the official Pytorch implementation of Neural Mesh Flow.
Instead of applying simple Instance Normalization, we leverage Adaptive Instance Normalization to inject the style code 
into the network. For architectural details, please refer to the original paper by Gupta et al.:
https://arxiv.org/abs/2007.10973
'''


class NeuralMeshFlow(nn.Module):
    '''
    Implementation of the Neural Mesh Flow pipeline. Refer Section 3 in the paper for more details.
    '''

    def __init__(self, opt, time=0.2, tol=1e-5):
        super(NeuralMeshFlow, self).__init__()
        '''
        Initialization
        encoder_type: 'image' or 'point' for SVR and shape completion tasks repectively
        PATH_svr: model file for trained PointsSVR
        zdim: length of latent embedding
        time: some number 0-1
        tol: tolerance of ODESolver.
        '''

        '''
        **** NOTE on design choices  ******

        zdim : We did not observe much benefit of increased latent embedding size (i.e. >1000) and it simply increases memory requirement
        time : time of integration (set to 0.2) is chosen since it is long enough for effective integration but not too large to cause complex dynamics
        tol  : A lower tol is always better but comes at a cost of inference time. Refer to ablation in Supplementary for this.

        '''

        self.bottleneck_size = opt.bottleneck_size
        print("Neural Mesh Flow with {} length embedding initialized".format(self.bottleneck_size))
        # self.hidden_neurons = opt.hidden_neurons
        self.adaptive = opt.adaptive

        # Three deform blocks to cause successive refinements. Refer Section 3 (Overal architecture)
        self.db1 = DeformBlock(time, num_hidden=512, latent_len=self.bottleneck_size, tol=tol)
        self.db2 = DeformBlock(time, num_hidden=512, latent_len=self.bottleneck_size, tol=tol)
        self.db3 = DeformBlock(time, num_hidden=512, latent_len=self.bottleneck_size, tol=tol)

        # Template spheres for training/testing.

        # [Insight :] While trainig with smaller (#vertices) sphere is faster, inference with larger sphere is more accurate.
        # Choosing spheres with even lower vertices can cause drop in performance due to unstable optimization.
        # Using #vertices >2520 doesn't yeild much benefit and takes more training time.

        # These spheres are generated using Pymesh library : pymesh.generate_icosphere(radius=1, center=(0,0,0), refinement_order=3 or 4)

        self.sph_train = trimesh.load('./aux_models/meshflow/mypymeshsph_3.obj')  # Unit sphere with 642 vertices
        self.sph_test = trimesh.load('./aux_models/meshflow/mypymeshsph_4.obj')  # Unit sphere with 2520 vertices

        # Convert faces and vertices to torch. Face information will later be used for generating differentiable meshes.
        self.f = [torch.from_numpy(self.sph_train.faces).int(), torch.from_numpy(self.sph_test.faces).int()]  # Faces
        self.v = [torch.from_numpy(self.sph_train.vertices), torch.from_numpy(self.sph_test.vertices)]  # Vertices

        self.time = time

        # Initialize InstanceNorm layers

        self.norm0 = AdaptiveInstanceNorm(self.bottleneck_size) if self.adaptive else InstanceNorm(self.bottleneck_size)
        self.norm1 = AdaptiveInstanceNorm(self.bottleneck_size) if self.adaptive else InstanceNorm(self.bottleneck_size)
        self.norm2 = AdaptiveInstanceNorm(self.bottleneck_size) if self.adaptive else InstanceNorm(self.bottleneck_size)
        self.norm3 = AdaptiveInstanceNorm(self.bottleneck_size) if self.adaptive else InstanceNorm(self.bottleneck_size)

    def forward(self, content_latent_vector, adain_params, time=None, train=False):
        '''
        input: BxNx3 tensor
        pred_y1: BxNx3 tensor; vertices after first deformation block
        pred_y2: BxNx3 tensor; vertices after second deformation block
        pred_y3: BxNx3 tensor; vertices after third deformation block
        face: BxKx3; faces to be used for constructing differentiable meshes
        '''
        batch_size = content_latent_vector.shape[0]
        assert content_latent_vector.device != 0, print(" Input is on CPU")
        content_latent_vector = content_latent_vector.unsqueeze(1)

        # Use smaller sphere (622 vertices) for training otherwise use a larger sphere (2520 vertices) for testing
        sphidx = 0 if train else 1
        sph = self.v[sphidx].unsqueeze(0).repeat(batch_size, 1, 1).type_as(content_latent_vector)
        face = self.f[sphidx].unsqueeze(0).repeat(batch_size, 1, 1).type_as(content_latent_vector)

        # Apply instance norm to input sphere
        sph = self.norm0(sph, content_latent_vector, adain_params[:, 0:3 * 2])

        # First Deform Block computation and its instance norm
        pred_y1, _ = self.db1(content_latent_vector, sph, None, time)
        if self.adaptive:
            pred_y1 = self.norm1(pred_y1, content_latent_vector,
                                 adain_params[:, 3 * 2:3 * 4])
        else:
            pred_y1 = self.norm1(pred_y1, content_latent_vector)

        # Second Deform Block computation and its instance norm
        pred_y2, _ = self.db2(content_latent_vector, pred_y1, None, time)
        if self.adaptive:
            pred_y2 = self.norm1(pred_y2, content_latent_vector,
                                 adain_params[:, 3 * 4:3 * 6])
        else:
            pred_y2 = self.norm1(pred_y2, content_latent_vector)

        # Third Deform Block computation and its instance norm
        pred_y3, _ = self.db3(content_latent_vector, pred_y2, None, time)
        if self.adaptive:
            pred_y3 = self.norm1(pred_y3, content_latent_vector,
                                 adain_params[:, 3 * 6:3 * 8])
        else:
            pred_y3 = self.norm1(pred_y3, content_latent_vector)

        output = {
            'faces': face,
            'points_1': pred_y1,
            'points_2': pred_y2,
            'points_3': pred_y3,
        }
        return output

    def generate_mesh(self, content_latent_vector, adain_params, time=None, train=False):
        '''
        input: BxNx3 tensor
        pred_y1: BxNx3 tensor; vertices after first deformation block
        pred_y2: BxNx3 tensor; vertices after second deformation block
        pred_y3: BxNx3 tensor; vertices after third deformation block
        face: BxKx3; faces to be used for constructing differentiable meshes
        '''
        batch_size = content_latent_vector.shape[0]
        assert content_latent_vector.device != 0, print(" Input is on CPU")
        content_latent_vector = content_latent_vector.unsqueeze(1)

        # Use smaller sphere (622 vertices) for training otherwise use a larger sphere (2520 vertices) for testing
        sphidx = 0 if self.training else 1
        sph = self.v[sphidx].unsqueeze(0).repeat(batch_size, 1, 1).type_as(content_latent_vector)
        face = self.f[sphidx].unsqueeze(0).repeat(batch_size, 1, 1).type_as(content_latent_vector)

        # Apply instance norm to input sphere
        sph = self.norm0(sph, content_latent_vector, adain_params[:, 0:3 * 2])

        # First Deform Block computation and its instance norm
        pred_y1, _ = self.db1(content_latent_vector, sph, None, time)
        if self.adaptive:
            pred_y1 = self.norm1(pred_y1, content_latent_vector,
                                 adain_params[:, 3 * 2:3 * 4])
        else:
            pred_y1 = self.norm1(pred_y1, content_latent_vector)

        # Second Deform Block computation and its instance norm
        pred_y2, _ = self.db2(content_latent_vector, pred_y1, None, time)
        if self.adaptive:
            pred_y2 = self.norm1(pred_y2, content_latent_vector,
                                 adain_params[:, 3 * 4:3 * 6])
        else:
            pred_y2 = self.norm1(pred_y2, content_latent_vector)

        # Third Deform Block computation and its instance norm
        pred_y3, _ = self.db3(content_latent_vector, pred_y2, None, time)
        if self.adaptive:
            pred_y3 = self.norm1(pred_y3, content_latent_vector,
                                 adain_params[:, 3 * 6:3 * 8])
        else:
            pred_y3 = self.norm1(pred_y3, content_latent_vector)

        return make_mesh(pred_y3, face)


def make_mesh(pred, face):
    vertices = pred[0, ...].detach().cpu().numpy()
    faces = face[0, ...].detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh