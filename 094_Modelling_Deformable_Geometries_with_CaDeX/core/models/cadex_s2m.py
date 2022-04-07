from .model_base import ModelBase
import torch
import copy
import trimesh
from torch import nn
import time
from core.net_bank.oflow_point import ResnetPointnet
from core.net_bank.oflow_decoder import DecoderCBatchNorm, Decoder
from core.net_bank.nvp_v2 import NVP_v2_5
from core.net_bank.cdc_v2_encoder import ATCSetEncoder, Query1D

import logging
from .utils.occnet_utils import get_generator
from torch import distributions as dist
import numpy as np
from copy import deepcopy

from core.models.utils.viz_cdc_render import viz_cdc
from core.models.utils.oflow_eval.evaluator import MeshEvaluator
from core.models.utils.oflow_common import eval_atc_all, eval_iou


class Model(ModelBase):
    def __init__(self, cfg):
        network = CaDeX_S2M(cfg)
        super().__init__(cfg, network)

        self.input_num = cfg["dataset"]["input_num"]
        self.num_atc = cfg["dataset"]["num_atc"]

        viz_mesh = []
        T = cfg["dataset"]["set_size"]
        for t in range(T):
            viz_mesh += ["mesh_t%d" % t]
        self.output_specs = {
            "metric": [
                "batch_loss",
                "loss_recon",
                "loss_corr",
                "loss_theta",
                "iou",
                "iou_gen",
                "iou_obs",
            ]
            + ["loss_reg_shift_len"],
            "image": ["mesh_viz_image", "query_viz_image"],
            "mesh": viz_mesh + ["cdc_mesh"],
            "video": ["flow_video"],
            "hist": ["loss_recon_i", "loss_corr_i", "cdc_shift", "theta_error"],
            "xls": ["running_metric_report", "results_observed", "results_generated"],
        }

        self.viz_one = cfg["logging"]["viz_one_per_batch"]
        self.iou_threshold = cfg["evaluation"]["iou_threshold"]
        self.corr_eval_project_to_final_mesh = cfg["evaluation"]["project_to_final_mesh"]
        if self.corr_eval_project_to_final_mesh:
            logging.warning("In config set Corr-Proj-To-Mesh true, ignore it, set to false")
            self.corr_eval_project_to_final_mesh = False
        self.mesh_extractor = get_generator(cfg)
        self.evaluator = MeshEvaluator(cfg["dataset"]["n_query_sample_eval"])

        self.viz_use_T = cfg["dataset"]["input_type"] != "pcl"

    def generate_mesh(self, c_t, c_g, use_uncomp_cdc=True):
        mesh_t_list = []
        net = self.network.module if self.__dataparallel_flag__ else self.network
        T = c_t.shape[0]
        # extract t0 mesh by query t0 space
        observation_c = {
            "c_t": c_t.unsqueeze(0).detach(),
            "c_g": c_g.unsqueeze(0).detach(),
            "query_t": torch.zeros((1, 1)).to(c_t.device),
        }
        mesh_t0 = self.mesh_extractor.generate_from_latent(c=observation_c, F=net.decode_by_current)
        # Safe operation, if no mesh is extracted, replace by a fake one
        if mesh_t0.vertices.shape[0] == 0:
            mesh_t0 = trimesh.primitives.Box(extents=(1.0, 1.0, 1.0))
            logging.warning("Mesh extraction fail, replace by a place holder")
        # get deformation code
        c_homeo = c_t.unsqueeze(0).transpose(2, 1)  # B,C,T
        # convert t0 mesh to cdc
        t0_mesh_vtx = np.array(mesh_t0.vertices).copy()
        t0_mesh_vtx = torch.Tensor(t0_mesh_vtx).cuda().unsqueeze(0)  # 1,Pts,3
        t0_mesh_vtx_cdc, t0_mesh_vtx_cdc_uncompressed = net.map2canonical(
            c_homeo[:, :, :1], t0_mesh_vtx.unsqueeze(1), return_uncompressed=True
        )  # code: B,C,T, query: B,T,N,3
        # get all frames vtx by mapping cdc to each frame
        # soruce_vtx_cdc = t0_mesh_vtx_cdc.expand(-1, T, -1, -1)
        soruce_vtx_cdc = t0_mesh_vtx_cdc_uncompressed.expand(-1, T, -1, -1)
        # surface_vtx = net.map2current(c_homeo, soruce_vtx_cdc).squeeze(0)
        surface_vtx = net.map2current(c_homeo, soruce_vtx_cdc, compressed=False).squeeze(0)
        # ! clamp all vtx to unit cube
        surface_vtx = torch.clamp(surface_vtx, -1.0, 1.0)
        surface_vtx = surface_vtx.detach().cpu().numpy()  # T,Pts,3
        # make meshes for each frame
        for t in range(0, T):
            mesh_t = deepcopy(mesh_t0)
            mesh_t.vertices = surface_vtx[t]
            mesh_t.update_vertices(mask=np.array([True] * surface_vtx.shape[0]))
            mesh_t_list.append(mesh_t)
        mesh_cdc = deepcopy(mesh_t0)
        mesh_cdc_vtx = t0_mesh_vtx_cdc_uncompressed if use_uncomp_cdc else t0_mesh_vtx_cdc
        mesh_cdc.vertices = mesh_cdc_vtx.squeeze(1).squeeze(0).detach().cpu().squeeze(0).numpy()
        mesh_cdc.update_vertices(mask=np.array([True] * surface_vtx.shape[0]))
        return mesh_t_list, surface_vtx, mesh_cdc

    def map_pc2cdc(self, batch, bid, key="seq_pc"):
        net = self.network.module if self.__dataparallel_flag__ else self.network
        _c_t = batch["c_t"][bid].unsqueeze(0)
        _obs_pc = batch[key][bid].unsqueeze(0)
        _, input_cdc_un = net.map2canonical(_c_t.transpose(2, 1), _obs_pc, return_uncompressed=True)
        input_cdc_un = input_cdc_un.detach().cpu().numpy().squeeze(0)
        return input_cdc_un

    def _postprocess_after_optim(self, batch):
        # eval iou
        if "occ_hat_iou" in batch.keys():
            report = {}

            occ_pred = batch["occ_hat_iou"].detach().cpu().numpy()
            occ_gt = batch["model_input"]["points.occ"].detach().cpu().numpy()
            iou = eval_iou(occ_gt, occ_pred, threshold=self.iou_threshold)  # B,T_all
            # make metric tensorboard
            iou_observed_theta_gt = iou[:, : self.input_num]
            iou_generated = iou[:, self.input_num :]
            batch["iou_obs"] = iou_observed_theta_gt.mean()
            batch["iou_gen"] = iou_generated.mean()
            batch["iou"] = iou.mean()
            # make report
            report["iou"] = iou.mean(axis=1).tolist()
            report["iou_obs"] = iou_observed_theta_gt.mean(axis=1).tolist()
            report["iou_gen"] = iou_generated.mean(axis=1).tolist()
            batch["running_metric_report"] = report

        if "c_t" in batch.keys():
            self.network.eval()
            phase = batch["model_input"]["phase"]
            viz_flag = batch["model_input"]["viz_flag"]
            TEST_RESULT_OBS = {}
            TEST_RESULT_GEN = {}
            B, T, _ = batch["c_t"].shape
            with torch.no_grad():
                # prepare viz mesh lists
                for t in range(T):
                    batch["mesh_t%d" % t] = []
                batch["cdc_mesh"] = []
                rendered_fig_list, rendered_fig_query_list, video_list = [], [], []
                for bid in range(B):
                    # generate mesh
                    # * With GT Theta
                    logging.info("Generating Mesh Observed/Unobserved with GT theta")
                    start_t = time.time()
                    mesh_t_list, _, mesh_cdc = self.generate_mesh(
                        batch["c_t"][bid], batch["c_g"][bid]
                    )
                    recon_time = time.time() - start_t
                    for t in range(0, T):  # if generate mesh, then save it
                        batch["mesh_t%d" % t].append(mesh_t_list[t])
                    batch["cdc_mesh"].append(mesh_cdc)
                    if phase.startswith("test"):
                        # evaluate the generated mesh list
                        logging.info("Generating Mesh Observed/Unobserved with GT theta")
                        # * With predict Theta
                        mesh_t_list_pred_theta, _, _ = self.generate_mesh(
                            batch["c_t_pred_theta"][bid], batch["c_g"][bid]
                        )
                        logging.warning("Start eval")
                        eval_dict_mean_gt_observed, _ = eval_atc_all(
                            pcl_corr=batch["model_input"]["points_mesh"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            pcl_chamfer=batch["model_input"]["points_chamfer"][bid][
                                : self.input_num
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            points_tgt=batch["model_input"]["points"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            occ_tgt=batch["model_input"]["points.occ"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            mesh_t_list=mesh_t_list[: self.input_num],
                            evaluator=self.evaluator,
                            corr_project_to_final_mesh=self.corr_eval_project_to_final_mesh,
                            eval_corr=self.input_num > 1,
                        )
                        eval_dict_mean_gt_generated, _ = eval_atc_all(
                            pcl_corr=batch["model_input"]["points_mesh"][bid][self.input_num :]
                            .detach()
                            .cpu()
                            .numpy(),
                            pcl_chamfer=batch["model_input"]["points_chamfer"][bid][
                                self.input_num :
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            points_tgt=batch["model_input"]["points"][bid][self.input_num :]
                            .detach()
                            .cpu()
                            .numpy(),
                            occ_tgt=batch["model_input"]["points.occ"][bid][self.input_num :]
                            .detach()
                            .cpu()
                            .numpy(),
                            mesh_t_list=mesh_t_list[self.input_num :],
                            evaluator=self.evaluator,
                            corr_project_to_final_mesh=self.corr_eval_project_to_final_mesh,
                        )
                        eval_dict_mean_pred_observed, _ = eval_atc_all(
                            pcl_corr=batch["model_input"]["points_mesh"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            pcl_chamfer=batch["model_input"]["points_chamfer"][bid][
                                : self.input_num
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            points_tgt=batch["model_input"]["points"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            occ_tgt=batch["model_input"]["points.occ"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            mesh_t_list=mesh_t_list_pred_theta,
                            evaluator=self.evaluator,
                            corr_project_to_final_mesh=self.corr_eval_project_to_final_mesh,
                            eval_corr=self.input_num > 1,
                        )
                        logging.warning("End eval")
                        # record the batch results
                        for k, v in eval_dict_mean_gt_observed.items():
                            _k = f"{k}(G)"
                            if _k not in TEST_RESULT_OBS.keys():
                                TEST_RESULT_OBS[_k] = [v]
                            else:
                                TEST_RESULT_OBS[_k].append(v)
                        for k, v in eval_dict_mean_pred_observed.items():
                            _k = f"{k}(P)"
                            if _k not in TEST_RESULT_OBS.keys():
                                TEST_RESULT_OBS[_k] = [v]
                            else:
                                TEST_RESULT_OBS[_k].append(v)
                        for k, v in eval_dict_mean_gt_generated.items():
                            _k = f"{k}(G)"
                            if _k not in TEST_RESULT_GEN.keys():
                                TEST_RESULT_GEN[_k] = [v]
                            else:
                                TEST_RESULT_GEN[_k].append(v)
                        theta_hat = batch["theta_hat"][bid].detach().cpu().numpy()
                        theta_gt = batch["theta_gt"][bid].detach().cpu().numpy()
                        for atc_i in range(self.num_atc):
                            error = abs(theta_hat[:, atc_i] - theta_gt[:, atc_i]).mean()
                            error = error / np.pi * 180.0
                            k = f"theta-{atc_i}-error(degree)"
                            if k not in TEST_RESULT_OBS.keys():
                                TEST_RESULT_OBS[k] = [error]
                            else:
                                TEST_RESULT_OBS[k].append(error)
                        if "time-all" not in TEST_RESULT_OBS.keys():
                            TEST_RESULT_OBS["time-all"] = [recon_time]
                        else:
                            TEST_RESULT_OBS["time-all"].append(recon_time)
                        logging.info("Test OBS: {}".format(TEST_RESULT_OBS))
                        logging.info("Test GEN: {}".format(TEST_RESULT_GEN))

                    # render an image of the mesh
                    if viz_flag:
                        scale_cdc = True
                        if "viz_cdc_scale" in self.cfg["logging"].keys():
                            scale_cdc = self.cfg["logging"]["viz_cdc_scale"]
                        viz_align_cdc = False
                        if "viz_align_cdc" in self.cfg["logging"].keys():
                            viz_align_cdc = self.cfg["logging"]["viz_align_cdc"]
                        fig_t_list, fig_query_list = viz_cdc(
                            mesh_t_list,
                            mesh_cdc,
                            input_pc=batch["seq_pc"][bid].detach().cpu().numpy(),
                            input_cdc=self.map_pc2cdc(batch, bid, key="seq_pc"),
                            corr_pc=batch["corr_pc"][bid].detach().cpu().numpy(),
                            corr_cdc=self.map_pc2cdc(batch, bid, key="corr_pc"),
                            object_T=batch["object_T"][bid].detach().cpu().numpy()
                            if self.viz_use_T
                            else None,
                            scale_cdc=scale_cdc,
                            interval=self.cfg["logging"]["mesh_viz_interval"],
                            query=batch["query"][bid].detach().cpu().numpy(),
                            query_occ=batch["query_occ"][bid].detach().cpu().numpy(),
                            align_cdc=viz_align_cdc,
                            cam_dst_default=1.7,
                        )
                        cat_fig = np.concatenate(fig_t_list, axis=0).transpose(2, 0, 1)
                        cat_fig = np.expand_dims(cat_fig, axis=0).astype(np.float) / 255.0
                        rendered_fig_list.append(cat_fig)
                        cat_fig2 = np.concatenate(fig_query_list, axis=1).transpose(2, 0, 1)
                        cat_fig2 = np.expand_dims(cat_fig2, axis=0).astype(np.float) / 255.0
                        rendered_fig_query_list.append(cat_fig2)
                        # pack a video
                        video = np.concatenate(
                            [i.transpose(2, 0, 1)[np.newaxis, ...] for i in fig_t_list], axis=0
                        )  # T,3,H,W
                        video = np.expand_dims(video, axis=0).astype(np.float) / 255.0
                        video_list.append(video)

                    # if not in test
                    if self.viz_one and not phase.startswith("test"):
                        break
                if viz_flag:
                    batch["mesh_viz_image"] = torch.Tensor(
                        np.concatenate(rendered_fig_list, axis=0)
                    )  # B,3,H,W
                    batch["query_viz_image"] = torch.Tensor(
                        np.concatenate(rendered_fig_query_list, axis=0)
                    )  # B,3,H,W
                    batch["flow_video"] = torch.Tensor(
                        np.concatenate(video_list, axis=0)
                    )  # B,T,3,H,W
            if phase.startswith("test"):
                batch["results_observed"] = TEST_RESULT_OBS
                batch["results_generated"] = TEST_RESULT_GEN
        del batch["model_input"]
        return batch


class CaDeX_S2M(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)

        self.num_atc = cfg["dataset"]["num_atc"]
        self.input_num = cfg["dataset"]["input_num"]

        cg_decoder = DecoderCBatchNorm
        if "cg_cbatchnorm" in cfg["model"].keys():
            if not cfg["model"]["cg_cbatchnorm"]:
                logging.info("Canonical Geometry Decoder not using CBatchNorm")
                cg_decoder = Decoder
        H = NVP_v2_5
        H_act = nn.LeakyReLU
        if "homeo_act" in cfg["model"].keys():
            act_type = cfg["model"]["homeo_act"]
            act_dict = {"relu": nn.ReLU, "elu": nn.ELU, "leakyrelu": nn.LeakyReLU}
            assert act_type in act_dict.keys(), "Homeo Activation not support"
            H_act = act_dict[act_type]

        self.network_dict = torch.nn.ModuleDict(
            {
                "homeomorphism_encoder": ATCSetEncoder(
                    **cfg["model"]["homeomorphism_encoder"], atc_num=self.num_atc
                ),
                "ci_decoder": Query1D(**cfg["model"]["ci_decoder"], t_dim=self.num_atc),
                "canonical_geometry_encoder": ResnetPointnet(
                    dim=3, **cfg["model"]["canonical_geometry_encoder"]
                ),
                "canonical_geometry_decoder": cg_decoder(
                    dim=3, z_dim=0, **cfg["model"]["canonical_geometry_decoder"]
                ),
                "homeomorphism_decoder": H(
                    **cfg["model"]["homeomorphism_decoder"], activation=H_act
                ),
            }
        )

        for k in self.network_dict:
            logging.info(
                "{} params in {}".format(
                    sum(param.numel() for param in self.network_dict[k].parameters()), k
                )
            )

        self.compress_cdc = cfg["model"]["compress_cdc"]

        self.corr_loss_weight = 1.0
        if "corr_weight" in cfg["model"].keys():
            self.corr_loss_weight = cfg["model"]["corr_weight"]
        self.corr_square = False
        if "corr_square" in cfg["model"].keys():
            self.corr_square = cfg["model"]["corr_square"]

        # regularization setting
        self.regularize_shift_len = -1.0
        if "regularize_shift_len" in cfg["model"].keys():
            if cfg["model"]["regularize_shift_len"] > 0.0:
                self.regularize_shift_len = float(cfg["model"]["regularize_shift_len"])
                logging.info(
                    "CDC regularize the deformation length by w={}".format(
                        self.regularize_shift_len
                    )
                )

        self.use_corr_loss = cfg["model"]["loss_corr"]

    @staticmethod
    def logit(x, safe=False):
        eps = 1e-16 if safe else 0.0
        return -torch.log((1 / (x + eps)) - 1)

    def map2canonical(self, code, query, return_uncompressed=False):
        # code: B,C,T, query: B,T,N,3
        # B1, M1, _ = F.shape # batch, templates, C
        # B2, _, M2, D = x.shape # batch, Npts, templates, 3
        coordinates = self.network_dict["homeomorphism_decoder"].forward(
            code.transpose(2, 1), query.transpose(2, 1)
        )
        if self.compress_cdc:
            out = torch.sigmoid(coordinates) - 0.5
        else:
            out = coordinates
        if return_uncompressed:
            return out.transpose(2, 1), coordinates.transpose(2, 1)  # B,T,N,3
        else:
            return out.transpose(2, 1)  # B,T,N,3

    def map2current(self, code, query, compressed=True):
        # code: B,C,T, query: B,T,N,3
        # B1, M1, _ = F.shape # batch, templates, C
        # B2, _, M2, D = x.shape # batch, Npts, templates, 3
        coordinates = self.logit(query + 0.5) if (self.compress_cdc and compressed) else query
        coordinates, _ = self.network_dict["homeomorphism_decoder"].inverse(
            code.transpose(2, 1), coordinates.transpose(2, 1)
        )
        return coordinates.transpose(2, 1)  # B,T,N,3

    def forward(self, input_pack, viz_flag):
        output = {}
        phase = input_pack["phase"]

        set_pc, theta_gt = input_pack["inputs"][:, : self.input_num], input_pack["theta"]
        B, T_all, _ = input_pack["theta"].shape
        T_in = self.input_num

        # encode Hoemo condition
        c_global, theta_hat = self.network_dict["homeomorphism_encoder"](set_pc)
        c_t = self.network_dict["ci_decoder"](c_global, theta_gt)  # B,T,C

        # tranform observation to CDC and encode canonical geometry
        inputs_cdc = self.map2canonical(c_t.transpose(2, 1), input_pack["inputs"])  # B,T,N,3
        c_g = self.network_dict["canonical_geometry_encoder"](inputs_cdc.reshape(B, -1, 3))

        # visualize
        if viz_flag:
            output["c_t"] = c_t.detach()
            output["c_g"] = c_g.detach()
            output["seq_pc"] = input_pack["inputs"]
            output["object_T"] = input_pack["object_T"].detach()
            output["corr_pc"] = input_pack["pointcloud"].detach()
            output["query"] = input_pack["points"].detach()
            output["query_occ"] = input_pack["points.occ"].detach()

        # if test, direct return
        if phase.startswith("test"):
            output["c_t"] = c_t.detach()
            output["c_g"] = c_g.detach()
            # also test with pred theta
            c_t_pred_theta = self.network_dict["ci_decoder"](c_global, theta_hat)
            output["c_t_pred_theta"] = c_t_pred_theta.detach()
            output["theta_hat"] = theta_hat.detach()
            output["theta_gt"] = input_pack["theta"][:, : self.input_num]
            output["theta_all"] = input_pack["theta"]

            return output

        # get deformation condition for traning steps
        c_homeomorphism = c_t.permute(0, 2, 1)  # B,C,T

        # transform to canonical frame
        cdc, uncompressed_cdc = self.map2canonical(
            c_homeomorphism, input_pack["points"], return_uncompressed=True
        )  # B,T,N,3
        shift = (uncompressed_cdc - input_pack["points"]).norm(dim=3)

        # reconstruct in canonical space
        pr = self.decode_by_cdc(observation_c=c_g, query=cdc)
        # occ_hat = pr.probs
        reconstruction_loss_i = torch.nn.functional.binary_cross_entropy_with_logits(
            pr.logits, input_pack["points.occ"], reduction="none"
        )
        reconstruction_loss = reconstruction_loss_i.mean()

        # theta regression loss
        theta_loss = torch.mean((theta_hat - theta_gt[:, :T_in]) ** 2)

        # compute corr loss
        if self.use_corr_loss:
            _, cdc_first_frame_un = self.map2canonical(
                c_t[:, 0].unsqueeze(2),
                input_pack["pointcloud"][:, 0].unsqueeze(1),
                return_uncompressed=True,
            )  # B,1,M,3
            cdc_forward_frames = self.map2current(
                c_t[:, 1:].transpose(2, 1),
                cdc_first_frame_un.expand(-1, T_all - 1, -1, -1),
                compressed=False,
            )
            if self.corr_square:
                corr_loss_i = (
                    torch.norm(
                        cdc_forward_frames - input_pack["pointcloud"][:, 1:].detach(), dim=-1
                    )
                    ** 2
                )
            else:
                corr_loss_i = torch.abs(
                    cdc_forward_frames - input_pack["pointcloud"][:, 1:].detach()
                ).sum(-1)
            corr_loss = corr_loss_i.mean()

        output["batch_loss"] = reconstruction_loss + theta_loss
        output["loss_recon"] = reconstruction_loss.detach()
        output["loss_theta"] = theta_loss.detach()
        output["loss_recon_i"] = reconstruction_loss_i.detach().reshape(-1)
        output["theta_error"] = torch.abs(theta_hat - theta_gt[:, :T_in]).view(-1).detach()
        if self.use_corr_loss:
            output["batch_loss"] = output["batch_loss"] + corr_loss * self.corr_loss_weight
            output["loss_corr"] = corr_loss.detach()
            output["loss_corr_i"] = corr_loss_i.detach().reshape(-1)
        if self.regularize_shift_len > 0.0:  # shift len loss
            regularize_shift_len_loss = shift.mean()
            output["batch_loss"] = (
                output["batch_loss"] + regularize_shift_len_loss * self.regularize_shift_len
            )
            output["loss_reg_shift_len"] = regularize_shift_len_loss.detach()
        output["cdc_shift"] = shift.detach().reshape(-1)

        if phase.startswith("val"):  # add eval
            output["occ_hat_iou"] = pr.probs

        return output

    def decode_by_cdc(self, observation_c, query):
        B, T, N, _ = query.shape
        query = query.reshape(B, -1, 3)
        logits = self.network_dict["canonical_geometry_decoder"](
            query, None, observation_c
        ).reshape(B, T, N)
        return dist.Bernoulli(logits=logits)

    def decode_by_current(self, query, z_none, c):
        # ! decoder by coordinate in current coordinate space, only used in viz
        c_t = c["c_t"]
        c_g = c["c_g"]
        query_t = c["query_t"]
        assert query.ndim == 3
        query = query.unsqueeze(1)
        idx = (query_t * (c_t.shape[1] - 1)).long()  # B,t
        c_homeomorphism = torch.gather(
            c_t, dim=1, index=idx.unsqueeze(2).expand(-1, -1, c_t.shape[-1])
        ).transpose(
            2, 1
        )  # B,C,T
        # transform to canonical frame
        cdc = self.map2canonical(c_homeomorphism, query)  # B,T,N,3
        logits = self.decode_by_cdc(observation_c=c_g, query=cdc).logits
        pr = dist.Bernoulli(logits=logits.squeeze(1))
        return pr
