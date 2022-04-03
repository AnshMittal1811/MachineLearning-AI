import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import training
from torchvision.utils import make_grid
import skimage.measure
import skvideo
import matplotlib
import skimage.io
from tqdm import tqdm
import forward_models
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')


def to_numpy(x):
    return x.detach().cpu().numpy()


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_tomography2D_summary(logging_root_path, case, additional_info, image_resolution, model, model_input, gt,
                               model_output, writer, total_steps, prefix='train_'):

    rho = gt['rho'][0].cpu()
    theta = gt['theta'][0].cpu()
    rho, theta = torch.meshgrid(rho, theta)
    img_shape = rho.shape

    rho = rho.reshape(-1)[None, :, None, None]
    theta = theta.reshape(-1)[None, :, None, None]

    min_t = -torch.sqrt(1. - rho**2)
    max_t = torch.sqrt(1. - rho**2)
    ray_len = max_t - min_t

    if model.use_grad:
        t = torch.cat((min_t, max_t), dim=-2)
        grad = True
    else:
        grad = False
        t = torch.linspace(0, 1, 128)[None, :, None]
        t = min_t * (1 - t) + max_t * t

    rho = rho.expand(t.shape)
    theta = theta.expand(t.shape)
    input_dict = {'rho': rho, 'theta': theta, 't': t}

    with torch.no_grad():
        model.use_grad = False
        pred_img = process_batch_in_chunks(input_dict, model)['model_out']['output'][0]
        if grad:
            pred_img = (pred_img[:, 1, :] - pred_img[:, 0, :]).squeeze().reshape(img_shape)
        else:
            pred_img = ray_len.squeeze().reshape(img_shape) * torch.mean(pred_img, dim=1).squeeze().reshape(img_shape)

        model.use_grad = grad

    gt_img = gt['radon_img'][0].float().cpu()

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-2)
    output_vs_gt = torch.nn.functional.upsample(output_vs_gt[None, ...], scale_factor=2)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True, nrow=1),
                     global_step=total_steps)

    fig = plt.figure()
    plt.plot(gt_img[:, gt_img.shape[1]//2])
    plt.plot(pred_img[:, pred_img.shape[1]//2])
    plt.ylim([-0.05, 0.8])
    writer.add_figure(prefix + 'gt_vs_pred_line', fig, global_step=total_steps)

    def rot_matrix(theta):
        rot = torch.zeros(2, 2)
        rot[0, 0] = torch.cos(theta)
        rot[1, 1] = torch.cos(theta)
        rot[0, 1] = -torch.sin(theta)
        rot[1, 0] = torch.sin(theta)
        return rot

    x = torch.linspace(-1, 1, image_resolution[0])
    y = torch.linspace(-1, 1, image_resolution[0])
    Y, X = torch.meshgrid(y, x)

    # select only X, Y values within the circle of the radon transform
    mask = (torch.sqrt(X**2 + Y**2) < 1.0).cuda()
    coords_grad_xy = torch.stack((Y.reshape(-1), X.reshape(-1)), dim=1)

    lin_theta = gt['theta'][0].squeeze()
    theta_idx = torch.linspace(0, len(lin_theta)-1, 180).long()
    out = []
    for idx in theta_idx:
        theta = lin_theta[idx]
        rot = rot_matrix(theta)

        coords_rho_t = torch.matmul(coords_grad_xy, rot.T).cuda()
        model_input = {'rho': coords_rho_t[None, :, 0:1].cuda(),
                       'theta': (theta * torch.ones_like(coords_rho_t[None, :, 0:1])).cuda(),
                       't': coords_rho_t[None, :, 1:2].cuda()}
        tmp = model(model_input)['model_out']['output'].detach().cpu()

        out.append(tmp.clone())
    out = torch.cat(out, dim=0)

    out = out.reshape(len(theta_idx), 1, mask.shape[0], mask.shape[1])

    nangles_to_show = 5
    out_grid = out[::180//nangles_to_show, :, :, :].cuda()

    for i in range(nangles_to_show):
        out_grid[i] *= mask
    out_grid = out_grid.transpose(-1, -2)
    out_grid = torch.nn.functional.upsample(out_grid, scale_factor=2)

    gt = torch.nn.functional.upsample(gt['iradon_img'][0][None, None, ...].float(), scale_factor=2)
    out_grid = torch.cat((gt, out_grid), dim=0)
    writer.add_image(prefix + 'grad', make_grid(out_grid, scale_each=False, normalize=True), global_step=total_steps)

    min_p_img = torch.min(out.view(-1, out.shape[-1]*out.shape[-2]))
    max_p_img = torch.max(out.view(-1, out.shape[-1]*out.shape[-2]))
    writer.add_video(prefix + 'grad_vid', (out[None, :, :, :, :]-min_p_img)/(max_p_img-min_p_img), global_step=total_steps, fps=30)


def write_simple_1D_function_summary(dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train'):

    jitter_bak = dataset.jitter
    dataset.jitter = False
    model_input, gt = dataset[0]
    model_input = {k: v.unsqueeze(0) for k, v in model_input.items()}
    model_input = training.dict2cuda(model_input)
    model_output = model(model_input)
    dataset.jitter = jitter_bak

    pred_func = to_numpy(model_output['model_out']['output'].squeeze())   # B, Samples, DimOut
    coords = to_numpy(gt['coords'].squeeze())       # B, Samples, DimIn

    val_coords = coords
    val_pred_func = pred_func
    val_gt_func = to_numpy(gt['func'].squeeze())                    # B, Samples, DimOut

    idx = model_input['idx'].cpu().long().detach().numpy().squeeze()
    train_coords = coords[idx]
    train_pred_func = pred_func[idx]

    fig = plt.figure()
    plt.plot(val_coords, val_gt_func, label='GT', linewidth=2)
    plt.plot(val_coords, val_pred_func, label='Val')
    plt.plot(train_coords, train_pred_func, '.', label='Train', markersize=8)

    plt.ylim([-1, 1])
    plt.legend()
    plt.tight_layout()
    writer.add_figure(prefix + '/gt_vs_pred', fig, global_step=total_steps)

    if model.use_grad and gt['integral'] is not None:
        # plot integral
        model.use_grad = False
        model_output = model(model_input)
        model.use_grad = True

        pred_integral = to_numpy(model_output['model_out']['output'].squeeze())
        val_pred_integral = pred_integral
        train_pred_integral = pred_integral[idx]
        val_gt_integral = to_numpy(gt['integral'].squeeze())

        fig = plt.figure()
        plt.plot(val_coords, val_gt_integral, label='GT', linewidth=2)
        plt.plot(val_coords, val_pred_integral, label='Val')
        plt.plot(train_coords, train_pred_integral, '.', label='Train', markersize=8)

        plt.ylim([-1, 1])
        plt.legend()
        plt.tight_layout()
        writer.add_figure(prefix + '/gt_vs_pred_integral', fig, global_step=total_steps)


def process_batch_in_chunks(in_dict, model, max_chunk_size=1024, progress=None):
    in_chunked = []
    for key in in_dict:
        num_views, num_rays, num_samples_per_rays, num_dims = in_dict[key].shape
        chunks = torch.split(in_dict[key].view(-1, num_samples_per_rays, num_dims), max_chunk_size)
        in_chunked.append(chunks)

    list_chunked_batched_in = \
        [{k: v for k, v in zip(in_dict.keys(), curr_chunks)} for curr_chunks in zip(*in_chunked)]
    del in_chunked

    list_chunked_batched_out_out = {}
    list_chunked_batched_out_in = {}
    for chunk_batched_in in list_chunked_batched_in:
        chunk_batched_in = {k: v.cuda() for k, v in chunk_batched_in.items()}
        tmp = model(chunk_batched_in)
        tmp = training.dict2cpu(tmp)

        for key in tmp['model_out']:
            if tmp['model_out'][key] is None:
                continue

            out_ = tmp['model_out'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_out.setdefault(key, []).append(out_)

        for key in tmp['model_in']:
            if tmp['model_in'][key] is None:
                continue

            in_ = tmp['model_in'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_in.setdefault(key, []).append(in_)

        del tmp, chunk_batched_in

        if progress is not None:
            progress.update(1)

    # Reassemble the output chunks in a batch
    batched_out = {}
    shape_out = list([num_views, num_rays, num_samples_per_rays, num_dims])
    for key in list_chunked_batched_out_out:
        batched_out_lin = torch.cat(list_chunked_batched_out_out[key], dim=0)
        shape_out[-1] = batched_out_lin.shape[-1]
        shape_out[-2] = -1
        batched_out[key] = batched_out_lin.reshape(shape_out)

    batched_in = {}
    shape_in = list([num_views, num_rays, num_samples_per_rays, num_dims])
    for key in list_chunked_batched_out_in:
        batched_in_lin = torch.cat(list_chunked_batched_out_in[key], dim=0)
        shape_in[-1] = batched_in_lin.shape[-1]
        shape_in[-2] = -1
        batched_in[key] = batched_in_lin.reshape(shape_in)

    # print(f"batched_out={batched_out.shape}")
    return {'model_in': batched_in, 'model_out': batched_out}


def peak_signal_noise_ratio(gt, pred):
    ''' Calculate PSNR using GT and predicted image (assumes valid values between 0 and 1 '''
    pred = torch.clamp(pred, 0, 1)
    return 10 * torch.log10(1 / torch.mean((gt - pred)**2))


def subsample_dict(in_dict, num_views):
    return {key: value[0:num_views, ...] for key, value in in_dict.items()}


def write_tomo_radiance_summary(models, train_dataloader, val_dataloader, loss_fn, optims,
                                meta, gt, misc, writer, total_steps,
                                chunk_size_eval, num_views_to_disp_at_training,
                                use_piecewise_model, num_cuts, use_coarse_fine):
    print('Running validation and logging...')

    chunk_size = chunk_size_eval

    '''' Log training set '''
    # sample rays across the whole image
    train_dataloader.dataset.toggle_logging_sampling()

    in_dict, meta_dict, gt_dict, misc_dict = next(iter(train_dataloader))
    in_dict = subsample_dict(in_dict, num_views_to_disp_at_training)

    # show progress
    samples_per_view = train_dataloader.dataset.samples_per_view
    num_chunks = num_views_to_disp_at_training * samples_per_view // chunk_size
    pbar = tqdm(total=2*len(models)*int(num_chunks))

    # Here, the number of images we get depend on the batch_size which is likely not going to be 1
    # so, be aware that we are processing multiple images
    with torch.no_grad():
        out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                    for key, model in models.items()}

    # Plot the sampling
    fig_sampling = plot_samples(out_dict)
    writer.add_figure('samples', fig_sampling, global_step=total_steps)
    gt_view = misc_dict['views'][0:num_views_to_disp_at_training, :, :, :3].detach().cpu()  # Views,H,W,C

    t_intervals = out_dict['sigma']['model_in']['t_intervals']
    if 'combined' in out_dict:
        if use_piecewise_model:
            pred_weights = forward_models.compute_transmittance_weights_piecewise(out_dict['combined']['model_out']['output'][..., -1:],
                                                                                  t_intervals, num_cuts)
            pred_pixels = forward_models.compute_tomo_radiance_piecewise(pred_weights, out_dict['combined']['model_out']['output'][..., :-1],
                                                                         num_cuts)
        else:
            pred_weights = forward_models.compute_transmittance_weights(out_dict['combined']['model_out']['output'][..., -1:],
                                                                        t_intervals)
            pred_pixels = forward_models.compute_tomo_radiance(pred_weights, out_dict['combined']['model_out']['output'][..., :-1])

    else:
        if use_piecewise_model:
            pred_weights = forward_models.compute_transmittance_weights_piecewise(out_dict['sigma']['model_out']['output'],
                                                                                  t_intervals, num_cuts)
            pred_pixels = forward_models.compute_tomo_radiance_piecewise(pred_weights, out_dict['rgb']['model_out']['output'],
                                                                         num_cuts)
        else:
            pred_weights = forward_models.compute_transmittance_weights(out_dict['sigma']['model_out']['output'],
                                                                        t_intervals)
            pred_pixels = forward_models.compute_tomo_radiance(pred_weights, out_dict['rgb']['model_out']['output'])

    # log the images
    pred_view = pred_pixels.view(gt_view.shape).detach().cpu()  # Views,H,W,C
    pred_view = torch.clamp(pred_view, 0, 1)
    train_psnr = peak_signal_noise_ratio(gt_view[0], pred_view[0])
    writer.add_scalar(f"train: PSNR", train_psnr, global_step=total_steps)

    # add videos takes B,T,C,H,W and we simply use it here to tile images T=1
    writer.add_video(f"train: GT", gt_view.permute(0, 3, 1, 2)[:, None, :, :, :], global_step=total_steps)
    writer.add_video(f"train: Pred", pred_view.permute(0, 3, 1, 2)[:, None, :, :, :], global_step=total_steps)

    # reset sampling back to defaults
    train_dataloader.dataset.toggle_logging_sampling()

    # Free by hand to be sure
    del in_dict, meta_dict, gt_dict, misc_dict
    del pred_view, pred_pixels, pred_weights,

    ''' Log Validation images '''
    num_samples = 1
    poses = []
    rays = []
    views = []

    val_dataloader.dataset.toggle_logging_sampling()

    for n in range(num_samples):  # we run a for loop of num_samples instead of a batch to use less cuda mem
        in_dict, meta_dict, gt_dict, misc_dict = next(iter(val_dataloader))

        with torch.no_grad():
            out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                        for key, model in models.items()}

        losses = loss_fn(out_dict, gt_dict)
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            writer.add_scalar('val_' + loss_name, single_loss, total_steps)

        t_intervals = out_dict['sigma']['model_in']['t_intervals']
        if 'combined' in out_dict:
            if use_piecewise_model:
                pred_weights = forward_models.compute_transmittance_weights_piecewise(
                                out_dict['combined']['model_out']['output'][..., -1:], t_intervals)
                pred_pixels = forward_models.compute_tomo_radiance_piecewise(pred_weights, out_dict['combined']['model_out']['output'][..., :-1])
            else:
                pred_weights = forward_models.compute_transmittance_weights(out_dict['combined']['model_out']['output'][..., -1:],
                                                                            t_intervals)
                pred_pixels = forward_models.compute_tomo_radiance(pred_weights, out_dict['combined']['model_out']['output'][..., :-1])
        else:
            if use_piecewise_model:
                pred_weights = forward_models.compute_transmittance_weights_piecewise(out_dict['sigma']['model_out']['output'],
                                                                                      t_intervals, num_cuts)
                pred_pixels = forward_models.compute_tomo_radiance_piecewise(pred_weights, out_dict['rgb']['model_out']['output'],
                                                                             num_cuts)
                pred_weights = forward_models.compute_transmittance_weights(out_dict['sigma']['model_out']['output'],
                                                                            t_intervals)
                pred_depth = forward_models.compute_tomo_depth(pred_weights, meta_dict['zs'])
                pred_disp = forward_models.compute_disp_from_depth(pred_depth, pred_weights)
            else:
                pred_weights = forward_models.compute_transmittance_weights(out_dict['sigma']['model_out']['output'],
                                                                            t_intervals)
                pred_pixels = forward_models.compute_tomo_radiance(pred_weights, out_dict['rgb']['model_out']['output'])

                pred_depth = forward_models.compute_tomo_depth(pred_weights, meta_dict['zs'])
                pred_disp = forward_models.compute_disp_from_depth(pred_depth, pred_weights)

        gt_view = misc_dict['views'][0, :, :, :3].detach().cpu()
        pred_view = pred_pixels.view(gt_view.shape).detach().cpu().permute(2, 0, 1)
        pred_view = torch.clamp(pred_view, 0, 1)
        pred_disp_view = pred_disp.view(gt_view[:, :, 0:1].shape).detach().cpu().permute(2, 0, 1)
        gt_view = gt_view.permute(2, 0, 1)

        val_psnr = peak_signal_noise_ratio(gt_view, pred_view)
        writer.add_scalar(f"val: PSNR", val_psnr, global_step=total_steps)

        # nearest neighbor upsample image for easier viewing
        if gt_view.shape[1] < 512:
            scale = 512 // gt_view.shape[1]
            gt_view = torch.nn.functional.interpolate(gt_view.unsqueeze(0), scale_factor=scale, mode='nearest')
            pred_view = torch.nn.functional.interpolate(pred_view.unsqueeze(0), scale_factor=scale, mode='nearest')
            pred_disp_view = torch.nn.functional.interpolate(pred_disp_view.unsqueeze(0), scale_factor=scale, mode='nearest')
            gt_view = gt_view.squeeze(0)
            pred_view = pred_view.squeeze(0)
            pred_disp_view = pred_disp_view.squeeze(0)

        writer.add_image(f"val: GT {n}", gt_view, global_step=total_steps)
        writer.add_image(f"val: Pred {n}", pred_view, global_step=total_steps)
        writer.add_image(f"val: Pred disp {n}", pred_disp_view, global_step=total_steps)

        # calculate samples along ray for the ray visualization
        t = in_dict['t']
        num_samples = t.shape[-2]
        origins = in_dict['ray_origins'].repeat(1, 1, num_samples, 1)
        directions = in_dict['ray_directions'].repeat(1, 1, num_samples, 1)

        ray_samples = origins + t * directions
        coords = torch.cat((ray_samples, origins), dim=-1)

        rays.append(coords[:, ::73, :, :])
        poses.append(misc_dict['poses'])
        view_shape = misc_dict['views'].shape[1]
        views.append(misc_dict['views'][0].detach().cpu())

    val_dataloader.dataset.toggle_logging_sampling()

    poses_batched = torch.cat(poses, dim=0).cpu()
    rays_batched = torch.cat(rays, dim=0).cpu()
    all_poses = misc_dict['all_poses'][0]

    # Nice 3D Visualization of the setup
    focal = val_dataloader.dataset.camera_params['focal'] / view_shape
    fig = visualize(poses_batched, focal, rays_batched, view_pose=None,
                    view_img=views[0].permute(1, 0, 2),
                    all_poses=all_poses)

    writer.add_figure(f"val: geometry", fig, global_step=total_steps)

    # close progress bar
    pbar.close()


def plot_samples(out_dict, num_rays_to_visu=10, xlim=(0, 6)):
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = plt.subplot(2, 1, 1)
    plt.title('sigma ray samples')
    t_transformed = torch.cumsum(out_dict['sigma']['model_in']['t_intervals'][0, ..., 0],
                                 dim=-1).cpu().detach()  # we could have t but it requires many more changes

    num_rays = t_transformed.shape[0]
    ts = t_transformed[num_rays//2:num_rays//2+num_rays_to_visu, :-1]
    num_samples = ts.shape[1]
    idcs = torch.arange(0, num_rays_to_visu).reshape(-1, 1).repeat(1, num_samples).float()
    idcs2 = torch.arange(0, num_samples).repeat(num_rays_to_visu).float()
    plt.scatter(ts.reshape(-1), idcs.reshape(-1), marker='|', c=idcs2.reshape(-1)/num_samples, cmap='prism')
    ax.set_ylabel('ray idx')
    ax.set_xlabel('sample position')
    ax.set_yticklabels([])
    plt.xlim(xlim)

    ax = plt.subplot(2, 1, 2)
    plt.title('rgb ray samples')
    t_transformed = torch.cumsum(out_dict['rgb']['model_in']['t_intervals'][0, ..., 0],
                                 dim=-1).cpu().detach()  # we could have t but it requires many more changes
    num_rays = t_transformed.shape[0]
    ts = t_transformed[num_rays//2:num_rays//2+num_rays_to_visu, :-1]
    num_samples = ts.shape[1]
    idcs = torch.arange(0, num_rays_to_visu).reshape(-1, 1).repeat(1, num_samples).float()
    idcs2 = torch.arange(0, num_samples).repeat(num_rays_to_visu).float()
    plt.scatter(ts.reshape(-1), idcs.reshape(-1), marker='|', c=idcs2.reshape(-1)/num_samples, cmap='prism')
    ax.set_ylabel('ray idx')
    ax.set_xlabel('sample position')
    ax.set_yticklabels([])
    plt.xlim(xlim)

    return fig


def visualize(camera_poses, focal, rays, view_pose=None, view_img=None, lims=((-4, 4), (-4, 4), (0, 4)),
              all_poses=None):
    '''Generates and returns a figure that illustrates camera & sampling geometry

    Parameters
    ----------
    camera_poses : array of size [batch_size, 3, 4]
        contains the camera rotation and translation matrix for each camera to be plotted in the
        visualization
    focal : float
        focal length of cameras in world units (not pixel)
    rays : array of shape [batch_size, num_rays, samples_per_ray, 6]
        gives the x,y,z,ox,oy,oz points along each ray, and the function will plot
        the line connecting the first and last samples per ray
    view_pose : 3 x 4 matrix (optional)
        contains the rotation and translation wrt world coordinates to show the scene
    view_img : array of shape [Nx, Ny, 4] (optional)
        an image shown at the origin of the coordinate system from the perspective defined by view_pose
    lims : 3-tuple of ((-xlim, xlim), (-ylim, ylim), (-zlim, zlim))
    all_poses : array of size [num_camera_poses, 3, 4]
        if not None, plot all camera poses with current poses indicated
    '''

    # make compound plot
    if all_poses is not None:
        matplotlib.rcParams['figure.figsize'] = [3, 3]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        fig = plt.figure()
        ax = plt.gca(projection='3d')

    camera_poses = [camera_poses[i] for i in range(camera_poses.shape[0])]
    rays = [rays[i] for i in range(rays.shape[0])]

    width = 1.0  # these are always fixed for our models
    height = 1.0

    for idx, (camera_pose, ray) in enumerate(zip(camera_poses, rays)):
        X_cam = create_camera_model(focal, focal, width, height)
        X_cam = [torch.Tensor(X) for X in X_cam]
        color = next(ax._get_lines.prop_cycler)['color']
        for i in range(len(X_cam)):
            X = np.zeros(X_cam[i].shape)
            for j in range(X_cam[i].shape[1]):
                X[0:4, j] = transform_to_matplotlib_frame(camera_pose, X_cam[i][0:4, j])

            ax.plot3D(X[0, :], X[1, :], X[2, :], color=color, linewidth=1, zorder=20)

        # iterate over rays per frustrum
        for ray_idx in range(ray.shape[0]):
            x0 = (ray[ray_idx, 0, 0]).detach().numpy().squeeze()
            y0 = (ray[ray_idx, 0, 1]).detach().numpy().squeeze()
            z0 = (ray[ray_idx, 0, 2]).detach().numpy().squeeze()
            x1 = (ray[ray_idx, -1, 0]).detach().numpy().squeeze()
            y1 = (ray[ray_idx, -1, 1]).detach().numpy().squeeze()
            z1 = (ray[ray_idx, -1, 2]).detach().numpy().squeeze()

            ax.plot3D(np.hstack((x0, x1)), np.hstack((y0, y1)), np.hstack((z0, z1)), color=color, zorder=10)

    if view_img is not None:
        # generate ray directions
        x = torch.linspace(-0.5, 0.5, view_img.shape[0]) / focal
        y = -torch.linspace(-0.5, 0.5, view_img.shape[1]) / focal
        X, Y = torch.meshgrid(x, y)
        Z = -torch.ones_like(X)

        # send rays out a distance equal to the camera distance from the origin
        dist = torch.sqrt(torch.sum(camera_poses[0][:, 3]**2))
        img_coords = torch.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), dim=0)
        img_coords = camera_poses[0][:3, :3].matmul(img_coords).permute(1, 0)
        img_coords = camera_poses[0][:3, 3][None, :] + dist * img_coords

        # plot the image as a pointcloud at that point
        ax.scatter(img_coords[:, 0], img_coords[:, 1], img_coords[:, 2], c=view_img.reshape(-1, 4), zorder=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    set_axes_equal(ax)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    def pose_to_xyz(pose, dir='z', vector=False):
        x = torch.Tensor([1.]) if dir == 'x' else torch.Tensor([0.])
        y = torch.Tensor([1.]) if dir == 'y' else torch.Tensor([0.])
        z = torch.Tensor([-1.]) if dir == 'z' else torch.Tensor([0.])
        view_dir = pose[:3, :3].matmul(torch.stack((x, y, z), dim=0))
        x, y, z = view_dir[0], view_dir[1], view_dir[2]

        if vector:
            x0, y0, z0 = pose[:3, 3]
            x1 = x0 + x/2
            y1 = y0 + y/2
            z1 = z0 + z/2
            return np.hstack((x0, x1)), np.hstack((y0, y1)), np.hstack((z0, z1))
        else:
            return x, y, z

    if view_pose is not None:
        # get viewing direction from the rotation
        x, y, z = pose_to_xyz(view_pose)
        el = torch.atan2(z, torch.sqrt(x**2 + y**2)) / np.pi * 180
        az = torch.atan2(y, x) / np.pi * 180
        ax.view_init(elev=el, azim=az)

    if all_poses is not None:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        all_poses = [all_poses[i] for i in range(all_poses.shape[0])]

        for pose in all_poses:
            x, y, z = pose_to_xyz(pose, vector=True)
            ax2.plot3D(x, y, z, color='black', linewidth=1)
            x, y, z = pose_to_xyz(pose, dir='y', vector=True)
            ax2.plot3D(x, y, z, color='black', linewidth=1)
            x, y, z = pose_to_xyz(pose, dir='x', vector=True)
            ax2.plot3D(x, y, z, color='black', linewidth=1)

        for pose in camera_poses:
            x, y, z = pose_to_xyz(pose, vector=True)
            ax2.plot3D(x, y, z, color='red', linewidth=1)
            x, y, z = pose_to_xyz(pose, dir='y', vector=True)
            ax2.plot3D(x, y, z, color='red', linewidth=1)
            x, y, z = pose_to_xyz(pose, dir='x', vector=True)
            ax2.plot3D(x, y, z, color='red', linewidth=1)
        set_axes_equal(ax2)

    return fig


# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# use the following functions from OpenCV example code
# https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py
def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = torch.eye(4)
    return M.matmul(cMo.matmul(X))


def create_camera_model(fx, fy, width, height, scale_focal=True, draw_frame_axis=False):
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, -f_scale]
    X_img_plane[0:3, 1] = [width, height, -f_scale]
    X_img_plane[0:3, 2] = [width, -height, -f_scale]
    X_img_plane[0:3, 3] = [-width, -height, -f_scale]
    X_img_plane[0:3, 4] = [-width, height, -f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, height, -f_scale]
    X_triangle[0:3, 1] = [0, 2*height, -f_scale]
    X_triangle[0:3, 2] = [width, height, -f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, -f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, -f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, -f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, -f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, -f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]


def render_views(output_path, models, dataset, num_cuts=32, use_sampler=True, integral_render=True, use_piecewise_model=False, chunk_size=1024, video=True):
    if video:
        writer = skvideo.io.FFmpegWriter(output_path + '.mp4', outputdict={
            '-vcodec': 'libx265', '-b': '30000000'}, verbosity=1)
    elif '.png' not in output_path:
        cond_mkdir(output_path)

    if integral_render:
        models['rgb'].set_mode('integral')
        models['sigma'].set_mode('integral')

    print('Rendering trajectory')
    for idx, (in_dict, meta_dict, _, misc_dict) in enumerate(tqdm(dataset)):

        # start timer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # unsqueeze batch dimension
        for key in in_dict:
            in_dict[key] = in_dict[key].unsqueeze(0)

        if not use_sampler:
            _, _, num_samples, _ = in_dict['t'].shape
            in_dict['t'] = in_dict['t'][:, :, ::num_samples//num_cuts, :]
            in_dict['t'][:, :, 0, :] = 2.0
            in_dict['t'] = torch.cat((in_dict['t'], 6.0 * torch.ones_like(in_dict['t'][:, :, 0:1, :])), dim=-2)

        out_dict = {}

        out_dict['sigma'] = process_batch_in_chunks(in_dict, models['sigma'], chunk_size)
        out_dict['rgb'] = process_batch_in_chunks(in_dict, models['rgb'], chunk_size)

        if integral_render:
            # evaluate integral
            out_dict['rgb']['model_out']['output'] = out_dict['rgb']['model_out']['output'][:, :, 1:, :] - out_dict['rgb']['model_out']['output'][:, :, 0:-1, :]
            out_dict['sigma']['model_out']['output'] = out_dict['sigma']['model_out']['output'][:, :, 1:, :] - out_dict['sigma']['model_out']['output'][:, :, 0:-1, :]

            # calculate average value over interval
            out_dict['rgb']['model_out']['output'] = out_dict['rgb']['model_out']['output'] / (out_dict['rgb']['model_in']['t_intervals'][..., :-1, :] / out_dict['rgb']['model_in']['ray_directions'].norm(p=2, dim=-1)[..., None])
            out_dict['sigma']['model_out']['output'] = out_dict['sigma']['model_out']['output'] / (out_dict['sigma']['model_in']['t_intervals'][..., :-1, :] / out_dict['sigma']['model_in']['ray_directions'].norm(p=2, dim=-1)[..., None])

            # last t_interval value should be infinite
            out_dict['sigma']['model_in']['t_intervals'][..., -2, :] = out_dict['sigma']['model_in']['t_intervals'][..., -1, :]
            out_dict['sigma']['model_in']['t_intervals'] = out_dict['sigma']['model_in']['t_intervals'][..., :-1, :]
            out_dict['rgb']['model_in']['t_intervals'][..., -2, :] = out_dict['rgb']['model_in']['t_intervals'][..., -1, :]
            out_dict['rgb']['model_in']['t_intervals'] = out_dict['rgb']['model_in']['t_intervals'][..., :-1, :]

        # run forward model
        if use_piecewise_model:
            pred_weights = forward_models.compute_transmittance_weights_piecewise(out_dict['sigma']['model_out']['output'], out_dict['sigma']['model_in']['t_intervals'], ncuts=num_cuts)
            pred_pixels = forward_models.compute_tomo_radiance_piecewise(pred_weights, out_dict['rgb']['model_out']['output'], ncuts_per_ray=num_cuts)
        else:
            pred_weights = forward_models.compute_transmittance_weights(out_dict['sigma']['model_out']['output'], out_dict['sigma']['model_in']['t_intervals'])
            pred_pixels = forward_models.compute_tomo_radiance(pred_weights, out_dict['rgb']['model_out']['output'])

        # composite onto white background
        # pred_pixels = pred_pixels + (1 - torch.sum(pred_weights, dim=-2, keepdim=False))

        pred_view = pred_pixels.view(*dataset.img_shape[:2], 3).detach().cpu()
        pred_view = torch.clamp(pred_view, 0, 1).numpy() * 255
        pred_view = pred_view.astype(np.uint8)

        gt_view = misc_dict['views'].detach().cpu()[:, :, 0:3]
        gt_view = torch.clamp(gt_view, 0, 1).numpy() * 255
        gt_view = gt_view.astype(np.uint8)

        # stop timer
        end.record()
        torch.cuda.synchronize()
        print(f'Elapsed time: {start.elapsed_time(end)}')

        if video:
            writer.writeFrame(pred_view)
        else:
            if '.png' in output_path:
                skimage.io.imsave(output_path, pred_view)
            else:
                skimage.io.imsave(output_path + f'/img_{idx:03d}.png', pred_view)

    if video:
        writer.close()
