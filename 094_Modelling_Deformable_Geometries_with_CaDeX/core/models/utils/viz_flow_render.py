# use pyrender to viz cdc
import numpy as np
from core.models.utils.pyrender_helper import render
from copy import deepcopy
import imageio
from transforms3d.euler import euler2mat


def viz_flow(
    mesh_list,
    input_pc,
    corr_pc,
    corr_pc_pred=None,
    object_T=None,
    interval=1,
    query=None,
    query_occ=None,
    use_color=True,
    cam_dist=1.0
):
    # prepare viz
    T = len(mesh_list)

    if object_T is None:
        object_T = np.eye(4)
    viz_m_list, viz_pc_list = [], []
    viz_corr_list = []
    viz_corr_pred_list = []

    color = deepcopy(mesh_list[0].vertices) + 0.5

    inv_T = np.linalg.inv(object_T)
    for t in range(T):
        m = deepcopy(mesh_list[t])
        m.apply_transform(inv_T)
        if use_color:
            m.visual.vertex_colors = color
        viz_m_list.append(m)
        pc = input_pc[t]  # N,3
        viz_pc_list.append((inv_T[:3, :3] @ pc.T + inv_T[:3, 3:4]).T)
        corr = corr_pc[t]
        viz_corr_list.append((inv_T[:3, :3] @ corr.T + inv_T[:3, 3:4]).T)
        if corr_pc_pred is not None:
            corr_pred = corr_pc_pred[t]
            viz_corr_pred_list.append((inv_T[:3, :3] @ corr_pred.T + inv_T[:3, 3:4]).T)

    cam_list = []
    for R in [
        euler2mat(0, 0, -np.pi / 6, "rzyx"),
        euler2mat(0, np.pi / 4, -np.pi / 6, "rzyx"),
        euler2mat(0, np.pi / 2, -np.pi / 6, "rzyx"),
    ]:
        cam = np.eye(4)
        cam[:3, :3] = R
        cam[:3, 3] = np.array([0.0, 0.0, cam_dist])
        cam[:3, 3:4] = R @ cam[:3, 3:4]
        cam_list.append(cam)

    query_viz_list = []
    if query is not None:
        assert query_occ is not None
        for _q, _o in zip(query, query_occ):
            _q = (inv_T[:3, :3] @ _q.T + inv_T[:3, 3:4]).T
            in_q = _q[_o > 0.5]
            out_q = _q[_o <= 0.5]
            _view_list = []
            for cam in cam_list:
                rgb_query, _ = render(
                    point_cloud=[in_q, out_q],
                    point_cloud_r=0.008,
                    point_cloud_material_color=[[1.0, 0.0, 0.6, 1.0], [0.0, 0.39, 1.0, 0.4]],
                    light_intensity=4.0,
                    camera_pose=cam,
                )
                rgb_query2, _ = render(
                    point_cloud=in_q,
                    point_cloud_r=0.008,
                    point_cloud_material_color=[1.0, 0.0, 0.6, 1.0],
                    light_intensity=4.0,
                    camera_pose=cam,
                )
                _view_list.append(np.concatenate([rgb_query, rgb_query2], axis=0))
            query_viz_list.append(np.concatenate(_view_list, axis=0))
    # render
    fig_t_list = []

    for t in range(T):
        if t % interval != 0:
            continue
        # render input pc
        rgb_input, _ = render(
            point_cloud=viz_pc_list[t],
            point_cloud_r=0.008,
            point_cloud_material_color=[0.0, 0.39, 1.0],
            light_intensity=4.0,
            cam_dst_default=cam_dist,
        )
        # render corr pc
        rgb_corr, _ = render(
            point_cloud=viz_corr_list[t],
            point_cloud_r=0.008,
            point_cloud_material_color=[1.0, 0.0, 0.6],
            light_intensity=4.0,
            cam_dst_default=cam_dist,
        )
        if corr_pc_pred is not None:
            rgb_corr_pred, _ = render(
                point_cloud=viz_corr_pred_list[t],
                point_cloud_r=0.008,
                point_cloud_material_color=[1.0, 0.6, 0.0],
                light_intensity=4.0,
                cam_dst_default=cam_dist,
            )
        # render resulting mesh + input pc
        rgb_input_result, _ = render(
            mesh=viz_m_list[t],
            mesh_material_color=[0.0, 0.0, 1.0, 0.4],
            use_mesh_material=True,
            point_cloud=viz_pc_list[t],
            point_cloud_r=0.008,
            point_cloud_material_color=[0.0, 0.39, 1.0],
            light_intensity=3.0,
            cam_dst_default=cam_dist,
        )
        # render resulting mesh
        _view_list = []
        for cam in cam_list:
            rgb_result, _ = render(
                mesh=viz_m_list[t], camera_pose=cam, use_mesh_material=not use_color
            )
            _view_list.append(rgb_result)
        if corr_pc_pred is not None:
            fig_t_list.append(
                np.concatenate(
                    [rgb_input, rgb_corr, rgb_corr_pred, rgb_input_result] + _view_list, axis=1
                )
            )
        else:
            fig_t_list.append(
                np.concatenate([rgb_input, rgb_corr, rgb_input_result] + _view_list, axis=1)
            )
    return fig_t_list, query_viz_list
