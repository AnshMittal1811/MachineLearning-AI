# visualize occupancy flow
from copy import deepcopy
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from random import shuffle

# plt.style.use("dark_background")


def viz_cdc(vtx_list, cdc_vtx, face, pc, mesh_viz_interval=3, cdc_input=None, scale_cdc=True):
    T = vtx_list.shape[0]
    vtx_list = vtx_list.copy()[:, :, [0, 2, 1]]
    cdc_vtx = cdc_vtx.copy()[:, [0, 2, 1]]
    pc = pc.copy()[:, :, [0, 2, 1]]
    if cdc_input is not None:
        cdc_input = cdc_input.copy()[:, :, [0, 2, 1]]
        assert cdc_input.shape == pc.shape
    fig_list = []
    bbox_r = 0.5
    random_index = [i for i in range(vtx_list.shape[1])]
    shuffle(random_index)
    random_index = random_index[:256]
    random_index2 = random_index[:100]

    # normalize cdc_vtx
    _t0_vtx = vtx_list[0]
    r = np.linalg.norm(_t0_vtx, axis=1).max()
    cdc_vtx_cent = cdc_vtx.mean(axis=0)[None, :]
    cdc_vtx = cdc_vtx - cdc_vtx_cent
    r_cdc = np.linalg.norm(cdc_vtx, axis=1).max()
    cdc_vtx = cdc_vtx
    if scale_cdc:
        cdc_vtx = cdc_vtx / (r_cdc + 1e-12) * r
    if cdc_input is not None:
        cdc_input = cdc_input - cdc_vtx_cent
        if scale_cdc:
            cdc_input = cdc_input / (r_cdc + 1e-12) * r
        cdc_input_all = cdc_input.reshape(-1, 3).copy()

    for t in range(0, T, int(mesh_viz_interval)):
        Nv = 5
        vtx = vtx_list[t]
        # prepare fig
        fig = plt.figure(figsize=plt.figaspect(1.0 / (Nv + 0.5)))  # Square figure
        canvas = FigureCanvas(fig)
        gs = fig.add_gridspec(1, Nv + 1)
        ax_list = [fig.add_subplot(gs[0, i], projection="3d") for i in range(Nv - 1)] + [
            fig.add_subplot(gs[0, Nv - 1 : Nv + 1], projection="3d")
        ]
        ax1, ax2, ax3, ax4 = ax_list[:4]
        for ax in ax_list:
            ax.view_init(elev=20.0, azim=45)
        ax_list[-1].view_init(elev=20.0, azim=75)
        for ax in ax_list:
            ax.set_axis_off()
        ax1.set_title("Step {} Flow (x10)".format(t))
        ax2.set_title("Step {} Input PC".format(t))
        ax3.set_title("Step {} Mesh".format(t))
        ax4.set_title("Step {} CDC-Mesh+Inputs".format(t))
        ax_list[-1].set_title("Step {} Posed <--> CDC".format(t))

        # plot cdc
        ax4 = ax_list[3]
        ax4.add_collection3d(
            art3d.Poly3DCollection(
                cdc_vtx[face], facecolors="yellowgreen", linewidths=0.08, alpha=0.7
            )
        )
        ax4.add_collection3d(
            art3d.Line3DCollection(cdc_vtx[face], colors="k", linewidths=0.04, linestyles=":")
        )
        if cdc_input is not None:
            # ax4.scatter(
            #     cdc_input_all[:, 0],
            #     cdc_input_all[:, 1],
            #     cdc_input_all[:, 2],
            #     c="c",
            #     marker=".",
            #     alpha=0.05,
            #     linewidths=0.01,
            # )
            ax4.scatter(
                cdc_input[t, :, 0],
                cdc_input[t, :, 1],
                cdc_input[t, :, 2],
                c="r",
                marker=".",
                alpha=1.0,
                linewidths=0.02,
            )

        for axis in "xyz":
            getattr(ax4, "set_%slim" % axis)((-bbox_r, bbox_r))

        # plot cdc-mapping
        shift = np.array([[bbox_r, 0, 0]])
        cdc_shifted = cdc_vtx - shift
        vtx_shifted = vtx + shift
        ax_list[-1].add_collection3d(
            art3d.Poly3DCollection(
                cdc_shifted[face],
                facecolors="yellowgreen",
                linewidths=0.08,
                alpha=0.7,
            )
        )
        ax_list[-1].add_collection3d(
            art3d.Line3DCollection(cdc_shifted[face], colors="k", linewidths=0.02, linestyles=":")
        )
        ax_list[-1].add_collection3d(
            art3d.Poly3DCollection(
                vtx_shifted[face],
                facecolors="tan",
                linewidths=0.08,
                alpha=0.7,
            )
        )
        ax_list[-1].add_collection3d(
            art3d.Line3DCollection(vtx_shifted[face], colors="k", linewidths=0.02, linestyles=":")
        )
        ax_list[-1].quiver(
            cdc_shifted[random_index2, 0],
            cdc_shifted[random_index2, 1],
            cdc_shifted[random_index2, 2],
            vtx_shifted[random_index2, 0] - cdc_shifted[random_index2, 0],
            vtx_shifted[random_index2, 1] - cdc_shifted[random_index2, 1],
            vtx_shifted[random_index2, 2] - cdc_shifted[random_index2, 2],
            color=[0.2, 0.4, 0.4, 0.7],
            linewidth=0.2,
            arrow_length_ratio=0.05,
            alpha=0.9,
        )
        for axis in "yz":
            getattr(ax_list[-1], "set_%slim" % axis)((-bbox_r, bbox_r))
        for axis in "x":
            getattr(ax_list[-1], "set_%slim" % axis)((-2 * bbox_r, 2 * bbox_r))

        # plot mesh
        for ax in ax_list[:3]:
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
            alpha=1.0,
            linewidths=0.03,
        )

        canvas.draw()
        fig.tight_layout()
        width, height = fig.get_size_inches() * fig.get_dpi()
        height, width = int(height), int(width)
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)

        # # debug
        # plt.savefig("./debug/debug.png")
        plt.close("all")

        # fig_list.append(image[:, :, [2, 1, 0]])
        fig_list.append(image)
    return fig_list
