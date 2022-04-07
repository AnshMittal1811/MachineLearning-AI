# visualize occupancy flow
from copy import deepcopy
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from random import shuffle


def viz_flow(vtx_list, face, pc, mesh_viz_interval=4):
    T = vtx_list.shape[0]
    vtx_list = vtx_list.copy()[:, :, [0, 2, 1]]
    pc = pc.copy()[:, :, [0, 2, 1]]
    fig_list = []
    bbox_r = 0.5
    random_index = [i for i in range(vtx_list.shape[1])]
    shuffle(random_index)
    random_index = random_index[:256]

    for t in range(0, T, int(mesh_viz_interval)):
        # prepare fig
        fig = plt.figure(figsize=plt.figaspect(1.0 / 4.0))  # Square figure
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(141, projection="3d")
        ax2 = fig.add_subplot(142, projection="3d")
        ax3 = fig.add_subplot(143, projection="3d")
        ax4 = fig.add_subplot(144, projection="3d")

        # plot mesh
        vtx = vtx_list[t]
        for ax in [ax1, ax2, ax3, ax4]:
            ax.add_collection3d(
                art3d.Poly3DCollection(vtx[face], facecolors="tan", linewidths=0.08, alpha=0.3)
            )
            ax.add_collection3d(
                art3d.Line3DCollection(vtx[face], colors="k", linewidths=0.04, linestyles=":")
            )
            for axis in "xyz":
                getattr(ax, "set_%slim" % axis)((-bbox_r, bbox_r))

        # plot flow
        if t < T - 1:
            ax1.quiver(
                vtx_list[t][random_index, 0],
                vtx_list[t][random_index, 1],
                vtx_list[t][random_index, 2],
                vtx_list[t + 1][random_index, 0] - vtx_list[t][random_index, 0],
                vtx_list[t + 1][random_index, 1] - vtx_list[t][random_index, 1],
                vtx_list[t + 1][random_index, 2] - vtx_list[t][random_index, 2],
                color=[0.545, 0.270, 0.074, 0.7],
                length=10.0,
            )

        # plot input point cloud
        ax2.scatter(
            pc[t, :, 0],
            pc[t, :, 1],
            pc[t, :, 2],
            c="darkgoldenrod",
            marker=".",
            alpha=0.9,
            linewidths=0.03,
        )

        # adjust view point and title
        ax1.view_init(elev=20.0, azim=45)
        ax2.view_init(elev=20.0, azim=45)
        ax3.view_init(elev=20.0, azim=45)
        ax4.view_init(elev=20.0, azim=-45)
        ax1.set_title("Step {} Flow (x10)".format(t))
        ax2.set_title("Step {} Input PC".format(t))
        ax3.set_title("Step {} view-1".format(t))
        ax4.set_title("Step {} view-2".format(t))

        canvas.draw()
        fig.tight_layout()
        width, height = fig.get_size_inches() * fig.get_dpi()
        height, width = int(height), int(width)
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)

        # plt.savefig("./debug.png")
        plt.close("all")

        # fig_list.append(image[:, :, [2, 1, 0]])
        fig_list.append(image)
    return fig_list


def viz_flow_ssd_displacement(
    vtx_list,
    face,
    pc,
    rbf_center,
    rbf_v,
    rbf_alpha,
    rbf_scale,
    mesh_viz_interval=4,
    sphere_sample=20,
):
    T = vtx_list.shape[0]
    vtx_list = vtx_list.copy()[:, :, [0, 2, 1]]
    pc = pc.copy()[:, :, [0, 2, 1]]
    rbf_center = rbf_center.copy()[:, :, [0, 2, 1]]
    rbf_v = rbf_v.copy()[:, :, [0, 2, 1]] * rbf_alpha[..., np.newaxis]
    fig_list = []
    bbox_r = 0.5
    random_index = [i for i in range(vtx_list.shape[1])]
    shuffle(random_index)
    random_index = random_index[:256]

    for t in range(0, T, int(mesh_viz_interval)):
        # prepare fig
        fig = plt.figure(figsize=plt.figaspect(1.0 / 5.0))  # Square figure
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(151, projection="3d")
        ax2 = fig.add_subplot(152, projection="3d")
        ax3 = fig.add_subplot(153, projection="3d")
        ax4 = fig.add_subplot(154, projection="3d")
        ax5 = fig.add_subplot(155, projection="3d")

        # plot mesh
        vtx = vtx_list[t]
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.add_collection3d(
                art3d.Poly3DCollection(vtx[face], facecolors="tan", linewidths=0.08, alpha=0.3)
            )
            ax.add_collection3d(
                art3d.Line3DCollection(vtx[face], colors="k", linewidths=0.04, linestyles=":")
            )
            for axis in "xyz":
                getattr(ax, "set_%slim" % axis)((-bbox_r, bbox_r))

        # plot flow
        if t < T - 1:
            ax1.quiver(
                vtx_list[t][random_index, 0],
                vtx_list[t][random_index, 1],
                vtx_list[t][random_index, 2],
                vtx_list[t + 1][random_index, 0] - vtx_list[t][random_index, 0],
                vtx_list[t + 1][random_index, 1] - vtx_list[t][random_index, 1],
                vtx_list[t + 1][random_index, 2] - vtx_list[t][random_index, 2],
                color=[0.545, 0.270, 0.074, 0.7],
                length=10.0,
            )

        # plot input point cloud
        ax2.scatter(
            pc[t, :, 0],
            pc[t, :, 1],
            pc[t, :, 2],
            c="darkgoldenrod",
            marker=".",
            alpha=0.9,
            linewidths=0.03,
        )

        # plot rbfs
        u = np.linspace(0, 2 * np.pi, sphere_sample)
        v = np.linspace(0, np.pi, sphere_sample)
        unit_sphere_x = np.outer(np.cos(u), np.sin(v))
        unit_sphere_y = np.outer(np.sin(u), np.sin(v))
        unit_sphere_z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.concatenate(
            [
                unit_sphere_x[np.newaxis, ...],
                unit_sphere_y[np.newaxis, ...],
                unit_sphere_z[np.newaxis, ...],
            ],
            axis=0,
        )  # 3,100,100
        sphere = sphere.reshape(3, sphere_sample ** 2)
        n_basis = rbf_scale.shape[1]
        for rbf_idx in range(n_basis):
            level = 0.5
            r = np.sqrt(np.log(1 / level ** 2)) / rbf_scale[t, rbf_idx]
            p = deepcopy(sphere) * r + rbf_center[t, rbf_idx][..., np.newaxis]
            ax3.plot_surface(
                p[0].reshape(sphere_sample, sphere_sample),
                p[1].reshape(sphere_sample, sphere_sample),
                p[2].reshape(sphere_sample, sphere_sample),
                rstride=4,
                cstride=4,
                color="r",
                alpha=0.1,
            )  # 0,1,2

        ax3.scatter(
            rbf_center[t, :, 0],
            rbf_center[t, :, 1],
            rbf_center[t, :, 2],
            c="deeppink",
            marker=".",
            alpha=0.9,
            linewidths=0.1,
        )
        ax3.quiver(
            rbf_center[t, :, 0],
            rbf_center[t, :, 1],
            rbf_center[t, :, 2],
            rbf_v[t, :, 0],
            rbf_v[t, :, 1],
            rbf_v[t, :, 2],
            color=[0.6, 0.196, 0.8, 0.7],
            length=10.0,
        )

        # adjust view point and title
        ax1.view_init(elev=20.0, azim=45)
        ax2.view_init(elev=20.0, azim=45)
        ax3.view_init(elev=20.0, azim=45)
        ax4.view_init(elev=20.0, azim=45)
        ax4.view_init(elev=20.0, azim=-45)
        ax1.set_title("Step {} Flow (x10)".format(t))
        ax2.set_title("Step {} Input PC".format(t))
        ax3.set_title("Step {} RBF d (x10)".format(t))
        ax4.set_title("Step {} view-1".format(t))
        ax5.set_title("Step {} view-2".format(t))

        canvas.draw()
        fig.tight_layout()
        width, height = fig.get_size_inches() * fig.get_dpi()
        height, width = int(height), int(width)
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)

        # plt.savefig("./debug.png")
        plt.close("all")

        fig_list.append(image[:, :, [2, 1, 0]])
    return fig_list


def viz_flow_ssd_RT(
    vtx_list,
    face,
    pc,
    rbf_center,
    rbf_v,
    rbf_W,
    rbf_alpha,
    rbf_scale,
    mesh_viz_interval=4,
    sphere_sample=20,
):
    T = vtx_list.shape[0]
    vtx_list = vtx_list.copy()[:, :, [0, 2, 1]]
    pc = pc.copy()[:, :, [0, 2, 1]]
    rbf_center = rbf_center.copy()[:, :, [0, 2, 1]]
    rbf_v = rbf_v.copy()[:, :, [0, 2, 1]] * rbf_alpha[..., np.newaxis]
    fig_list = []
    bbox_r = 0.5
    random_index = [i for i in range(vtx_list.shape[1])]
    shuffle(random_index)
    random_index = random_index[:256]

    for t in range(0, T, int(mesh_viz_interval)):
        # prepare fig
        fig = plt.figure(figsize=plt.figaspect(1.0 / 5.0))  # Square figure
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(151, projection="3d")
        ax2 = fig.add_subplot(152, projection="3d")
        ax3 = fig.add_subplot(153, projection="3d")
        ax4 = fig.add_subplot(154, projection="3d")
        ax5 = fig.add_subplot(155, projection="3d")

        # plot mesh
        vtx = vtx_list[t]
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.add_collection3d(
                art3d.Poly3DCollection(vtx[face], facecolors="tan", linewidths=0.08, alpha=0.3)
            )
            ax.add_collection3d(
                art3d.Line3DCollection(vtx[face], colors="k", linewidths=0.04, linestyles=":")
            )
            for axis in "xyz":
                getattr(ax, "set_%slim" % axis)((-bbox_r, bbox_r))

        # plot flow
        if t < T - 1:
            ax1.quiver(
                vtx_list[t][random_index, 0],
                vtx_list[t][random_index, 1],
                vtx_list[t][random_index, 2],
                vtx_list[t + 1][random_index, 0] - vtx_list[t][random_index, 0],
                vtx_list[t + 1][random_index, 1] - vtx_list[t][random_index, 1],
                vtx_list[t + 1][random_index, 2] - vtx_list[t][random_index, 2],
                color=[0.545, 0.270, 0.074, 0.7],
                length=10.0,
            )

        # plot input point cloud
        ax2.scatter(
            pc[t, :, 0],
            pc[t, :, 1],
            pc[t, :, 2],
            c="darkgoldenrod",
            marker=".",
            alpha=0.9,
            linewidths=0.03,
        )

        # plot rbfs
        u = np.linspace(0, 2 * np.pi, sphere_sample)
        v = np.linspace(0, np.pi, sphere_sample)
        unit_sphere_x = np.outer(np.cos(u), np.sin(v))
        unit_sphere_y = np.outer(np.sin(u), np.sin(v))
        unit_sphere_z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.concatenate(
            [
                unit_sphere_x[np.newaxis, ...],
                unit_sphere_y[np.newaxis, ...],
                unit_sphere_z[np.newaxis, ...],
            ],
            axis=0,
        )  # 3,100,100
        sphere = sphere.reshape(3, sphere_sample ** 2)
        n_basis = rbf_scale.shape[1]
        for rbf_idx in range(n_basis):
            level = 0.5
            r = np.sqrt(np.log(1 / level ** 2)) / rbf_scale[t, rbf_idx]
            p = deepcopy(sphere) * r + rbf_center[t, rbf_idx][..., np.newaxis]
            ax3.plot_surface(
                p[0].reshape(sphere_sample, sphere_sample),
                p[1].reshape(sphere_sample, sphere_sample),
                p[2].reshape(sphere_sample, sphere_sample),
                rstride=4,
                cstride=4,
                color="pink",
                alpha=0.2,
            )  # 0,1,2

        ax3.scatter(
            rbf_center[t, :, 0],
            rbf_center[t, :, 1],
            rbf_center[t, :, 2],
            c="deeppink",
            marker=".",
            alpha=0.9,
            linewidths=0.1,
        )
        r = np.sqrt(np.log(1 / level ** 2)) / rbf_scale[t]
        for direction, color in zip(
            range(3), [(1.0, 0.0, 0.0, 0.3), (0.0, 1.0, 0.0, 0.3), (0.0, 0.0, 1.0, 0.3)]
        ):
            ax3.quiver(
                rbf_center[t, :, 0],
                rbf_center[t, :, 1],
                rbf_center[t, :, 2],
                rbf_W[t, :, 0, direction] * r,
                rbf_W[t, :, 1, direction] * r,
                rbf_W[t, :, 2, direction] * r,
                color=color,
                length=1.0,
            )
        ax3.quiver(
            rbf_center[t, :, 0],
            rbf_center[t, :, 1],
            rbf_center[t, :, 2],
            rbf_v[t, :, 0],
            rbf_v[t, :, 1],
            rbf_v[t, :, 2],
            color=[0.6, 0.196, 0.8, 0.7],
            length=10.0,
        )

        # adjust view point and title
        ax1.view_init(elev=20.0, azim=45)
        ax2.view_init(elev=20.0, azim=45)
        ax3.view_init(elev=20.0, azim=45)
        ax4.view_init(elev=20.0, azim=45)
        ax4.view_init(elev=20.0, azim=-45)
        ax1.set_title("Step {} Flow (x10)".format(t))
        ax2.set_title("Step {} Input PC".format(t))
        ax3.set_title("Step {} RBF d (x10)".format(t))
        ax4.set_title("Step {} view-1".format(t))
        ax5.set_title("Step {} view-2".format(t))

        canvas.draw()
        fig.tight_layout()
        width, height = fig.get_size_inches() * fig.get_dpi()
        height, width = int(height), int(width)
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)

        # plt.savefig("./debug.png")
        plt.close("all")

        fig_list.append(image[:, :, [2, 1, 0]])
    return fig_list


def viz_flow_mlp(vtx_list, face_list, flow_list, pc, key_frame_interval, mesh_viz_interval=4):
    T = len(vtx_list)
    vtx_list = [v[:, [0, 2, 1]].copy() for v in vtx_list]
    flow_list = [f[:, [0, 2, 1]].copy() for f in flow_list]
    pc = pc.copy()[:, :, [0, 2, 1]]
    fig_list = []
    bbox_r = 0.5
    random_index = [i for i in range(min([v.shape[0] for v in vtx_list]))]
    shuffle(random_index)
    random_index = random_index[:256]

    for t in range(0, T, int(mesh_viz_interval)):
        # prepare fig
        fig = plt.figure(figsize=plt.figaspect(1.0 / 4.0))  # Square figure
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(141, projection="3d")
        ax2 = fig.add_subplot(142, projection="3d")
        ax3 = fig.add_subplot(143, projection="3d")
        ax4 = fig.add_subplot(144, projection="3d")

        # plot mesh
        vtx = vtx_list[t]
        face = face_list[t]
        for ax in [ax1, ax2, ax3, ax4]:
            if t % key_frame_interval == 0:
                face_color = "greenyellow"
            else:
                face_color = "tan"
            ax.add_collection3d(
                art3d.Poly3DCollection(vtx[face], facecolors=face_color, linewidths=0.08, alpha=0.3)
            )
            ax.add_collection3d(
                art3d.Line3DCollection(vtx[face], colors="k", linewidths=0.04, linestyles=":")
            )
            for axis in "xyz":
                getattr(ax, "set_%slim" % axis)((-bbox_r, bbox_r))

        # plot flow
        if t < T - 1:
            ax1.quiver(
                vtx_list[t][random_index, 0],
                vtx_list[t][random_index, 1],
                vtx_list[t][random_index, 2],
                flow_list[t][random_index, 0],
                flow_list[t][random_index, 1],
                flow_list[t][random_index, 2],
                color=[0.545, 0.270, 0.074, 0.7],
                length=10.0,
            )

        # plot input point cloud
        ax2.scatter(
            pc[t, :, 0],
            pc[t, :, 1],
            pc[t, :, 2],
            c="darkgoldenrod",
            marker=".",
            alpha=0.9,
            linewidths=0.03,
        )

        # adjust view point and title
        ax1.view_init(elev=20.0, azim=45)
        ax2.view_init(elev=20.0, azim=45)
        ax3.view_init(elev=20.0, azim=45)
        ax4.view_init(elev=20.0, azim=-45)
        ax1.set_title("Step {} Flow (x10)".format(t))
        ax2.set_title("Step {} Input PC".format(t))
        ax3.set_title("Step {} view-1".format(t))
        ax4.set_title("Step {} view-2".format(t))

        canvas.draw()
        fig.tight_layout()
        width, height = fig.get_size_inches() * fig.get_dpi()
        height, width = int(height), int(width)
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)

        # plt.savefig("./debug.png")
        plt.close("all")

        fig_list.append(image[:, :, [2, 1, 0]])
    return fig_list
