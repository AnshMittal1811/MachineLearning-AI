# use pyrender to viz cdc
import numpy as np
from core.models.utils.pyrender_helper import render
from copy import deepcopy
import imageio
from matplotlib import cm


def align(p0, p1):
    # R rotate p1 tp p0
    x0 = deepcopy(p0)
    y0 = deepcopy(p1)
    q0 = x0 - x0.mean(axis=0)
    q1 = y0 - y0.mean(axis=0)
    W = (q1[..., None] @ q0[:, None, :]).sum(axis=0)
    U, s, VT = np.linalg.svd(W)
    R = VT.T @ U.T
    return R


def viz_cdc(
    mesh_list,
    cdc_mesh,
    input_pc,
    input_cdc,
    corr_pc,
    corr_cdc,
    object_T=None,
    scale_cdc=True,
    interval=1,
    query=None,
    query_occ=None,
    align_cdc=False,
    cam_dst_default=1.0,
):
    # prepare viz
    T = len(mesh_list)

    if object_T is None:
        object_T = np.eye(4)
    viz_m_list, viz_pc_list, viz_pc_cdc_list = [], [], []
    viz_corr_list, viz_corr_cdc_list = [], []
    viz_cdc_m = deepcopy(cdc_mesh)

    cdc_vtx = deepcopy(cdc_mesh.vertices)
    cdc_center = cdc_vtx.mean(axis=0)
    cdc_vtx = cdc_vtx - cdc_center
    cdc_r = np.linalg.norm(cdc_vtx, axis=1).max()
    if scale_cdc:
        cdc_vtx = cdc_vtx / (cdc_r + 1e-8) * 0.4
    viz_cdc_m.vertices = cdc_vtx
    viz_cdc_m.update_vertices(np.ones((cdc_vtx.shape[0])) > 0.0)
    for t in range(T):
        pc_cdc = input_cdc[t] - cdc_center
        _corr_cdc = corr_cdc[t] - cdc_center
        if scale_cdc:
            pc_cdc = pc_cdc / (cdc_r + 1e-8) * 0.4
            _corr_cdc = _corr_cdc / (cdc_r + 1e-8) * 0.4
        viz_pc_cdc_list.append(pc_cdc)
        viz_corr_cdc_list.append(_corr_cdc)
    color = deepcopy(cdc_mesh.vertices)
    color = color - cdc_center
    color = color / (cdc_r + 1e-8) * 0.4
    color += 0.5
    viz_cdc_m.visual.vertex_colors = color

    inv_T = np.linalg.inv(object_T)
    for t in range(T):
        m = deepcopy(mesh_list[t])
        m.apply_transform(inv_T)
        m.visual.vertex_colors = color
        viz_m_list.append(m)
        pc = input_pc[t]  # N,3
        viz_pc_list.append((inv_T[:3, :3] @ pc.T + inv_T[:3, 3:4]).T)
        corr = corr_pc[t]
        viz_corr_list.append((inv_T[:3, :3] @ corr.T + inv_T[:3, 3:4]).T)

    # align cdc to the pose of posed frame first frame
    object_T = np.eye(4)
    if align_cdc:
        R = align(np.asarray(viz_m_list[int(T / 2)].vertices), cdc_vtx)
        object_T[:3, :3] = R

    query_viz_list = []
    if query is not None:
        assert query_occ is not None
        for _q, _o in zip(query, query_occ):
            _q = (inv_T[:3, :3] @ _q.T + inv_T[:3, 3:4]).T
            in_q = _q[_o > 0.5]
            out_q = _q[_o <= 0.5]
            rgb_query, _ = render(
                point_cloud=[in_q, out_q],
                point_cloud_r=0.008,
                point_cloud_material_color=[[1.0, 0.0, 0.6, 1.0], [0.0, 0.39, 1.0, 0.4]],
                light_intensity=4.0,
                cam_dst_default=cam_dst_default,
            )
            rgb_query2, _ = render(
                point_cloud=in_q,
                point_cloud_r=0.008,
                point_cloud_material_color=[1.0, 0.0, 0.6, 1.0],
                light_intensity=4.0,cam_dst_default=cam_dst_default,
            )
            query_viz_list.append(np.concatenate([rgb_query, rgb_query2], axis=0))
    # render
    fig_t_list = []
    # render cdc mesh
    rgb_cdc, _ = render(mesh=viz_cdc_m, object_pose=object_T)
    t_color = cm.summer(np.array([float(i) / float(T) for i in range(T)]))[:, :3]
    for t in range(T):
        if t % interval != 0:
            continue
        # render input pc
        rgb_input, _ = render(
            point_cloud=viz_pc_list[t],
            point_cloud_r=0.008,
            point_cloud_material_color=t_color[t],  # [0.0, 0.39, 1.0],
            light_intensity=4.0,cam_dst_default=cam_dst_default,
        )
        # render input cdc + cdc mesh
        rgb_input_cdc, _ = render(
            mesh=viz_cdc_m,
            mesh_material_color=[0.0, 1.0, 0.4, 0.2],
            use_mesh_material=True,
            point_cloud=viz_pc_cdc_list[t],
            point_cloud_r=0.008,
            point_cloud_material_color=t_color[t],  # [0.0, 0.39, 1.0],
            light_intensity=4.0,
            object_pose=object_T,cam_dst_default=cam_dst_default,
        )
        # render corr pc
        rgb_corr, _ = render(
            point_cloud=viz_corr_list[t],
            point_cloud_r=0.008,
            point_cloud_material_color=[1.0, 0.0, 0.6],
            light_intensity=4.0,cam_dst_default=cam_dst_default,
        )
        # render corr cdc + cdc mesh
        rgb_corr_cdc, _ = render(
            mesh=viz_cdc_m,
            mesh_material_color=[0.0, 1.0, 0.4, 0.2],
            use_mesh_material=True,
            point_cloud=viz_corr_cdc_list[t],
            point_cloud_r=0.008,
            point_cloud_material_color=[1.0, 0.0, 0.6],
            light_intensity=4.0,
            object_pose=object_T,cam_dst_default=cam_dst_default,
        )
        # render resulting mesh
        rgb_result, _ = render(mesh=viz_m_list[t])
        # render resulting mesh + input pc
        rgb_input_result, _ = render(
            mesh=viz_m_list[t],
            mesh_material_color=[0.0, 0.0, 1.0, 0.2],
            use_mesh_material=True,
            point_cloud=viz_pc_list[t],
            point_cloud_r=0.008,
            point_cloud_material_color=t_color[t],  # [0.0, 0.39, 1.0],
            light_intensity=3.0,cam_dst_default=cam_dst_default,
        )
        cdc_accumulate, _ = render(
            mesh=viz_cdc_m,
            mesh_material_color=[0.0, 1.0, 0.4, 0.2],
            use_mesh_material=True,
            point_cloud=viz_pc_cdc_list[:t],
            point_cloud_r=0.008,
            point_cloud_material_color=[t_color[_t].tolist() for _t in range(t + 1)],
            light_intensity=4.0,
            object_pose=object_T,cam_dst_default=cam_dst_default,
        )
        first_row = np.concatenate([rgb_input, rgb_corr, rgb_result, rgb_input_result], axis=1)
        second_row = np.concatenate([rgb_input_cdc, rgb_corr_cdc, rgb_cdc, cdc_accumulate], axis=1)
        fig_t_list.append(np.concatenate([first_row, second_row], axis=0))
    return fig_t_list, query_viz_list
