from .model_base import ModelBase
import torch

import copy

from core.net_bank.lpdc_encoder import SpatioTemporalResnetPointnetCDC
from core.net_bank.oflow_point import ResnetPointnet
from core.net_bank.oflow_decoder import DecoderCBatchNorm
from core.net_bank.nvp_v2 import NVP_v2, NVP_v2_5
from core.net_bank.nice import NICE

import logging
from .utils.occnet_utils import get_generator
from torch import distributions as dist
import numpy as np
from copy import deepcopy

from core.models.utils.viz_cdc import viz_cdc
from core.models.utils.oflow_eval.evaluator import MeshEvaluator
from core.models.utils.oflow_common import eval_oflow_all, eval_iou


class Model(ModelBase):
    def __init__(self, cfg):
        network = CaDeX_DFAU(cfg)
        super().__init__(cfg, network)
        eval_metric, viz_mesh = [], []
        try:
            T = cfg["dataset"]["oflow_config"]["length_sequence"]
        except:
            T = cfg["dataset"]["seq_len"]
        for t in range(T):
            eval_metric += ["iou_t%d" % t]
            viz_mesh += ["mesh_t%d" % t]
        self.output_specs = {
            "metric": ["batch_loss", "loss_recon", "loss_corr", "iou", "rec_error"]
            + eval_metric
            + ["loss_reg_shift_len"],
            "image": ["mesh_viz_image"],
            "mesh": viz_mesh + ["cdc_mesh"],
            "video": ["flow_video"],
            "hist": ["loss_recon_i", "loss_corr_i", "cdc_shift"],
            "xls": ["running_metric_report", "results_m", "results_t"],
        }

        self.viz_one = cfg["logging"]["viz_one_per_batch"]
        self.iou_threshold = cfg["evaluation"]["iou_threshold"]
        self.corr_eval_project_to_final_mesh = cfg["evaluation"]["project_to_final_mesh"]
        if self.corr_eval_project_to_final_mesh:
            logging.warning("In config set Corr-Proj-To-Mesh true, ignore it, set to false")
            self.corr_eval_project_to_final_mesh = False
        self.mesh_extractor = get_generator(cfg)
        self.evaluator = MeshEvaluator(cfg["dataset"]["n_query_sample_eval"])

    def generate_mesh(self, c_t, c_g, eval_t, use_uncomp_cdc=True):
        mesh_t_list = []
        net = self.network.module if self.__dataparallel_flag__ else self.network
        T = len(eval_t)
        # extract t0 mesh by query t0 space
        observation_c = {
            "c_t": c_t.unsqueeze(0).detach(),
            "c_g": c_g.unsqueeze(0).detach(),
            "query_t": torch.zeros((1, 1)).to(c_t.device),
        }
        mesh_t0 = self.mesh_extractor.generate_from_latent(c=observation_c, F=net.decode_by_current)
        # get deformation code
        c_homeo = c_t.unsqueeze(0).transpose(2, 1)  # B,C,T
        # convert t0 mesh to cdc
        t0_mesh_vtx = np.array(mesh_t0.vertices).copy()
        t0_mesh_vtx = torch.Tensor(t0_mesh_vtx).cuda().unsqueeze(0)  # 1,Pts,3
        t0_mesh_vtx_cdc, t0_mesh_vtx_cdc_uncompressed = net.map2canonical(
            c_homeo[:, :, :1], t0_mesh_vtx.unsqueeze(1), return_uncompressed=True
        )  # code: B,C,T, query: B,T,N,3
        # get all frames vtx by mapping cdc to each frame
        soruce_vtx_cdc = t0_mesh_vtx_cdc.expand(-1, T, -1, -1)
        surface_vtx = net.map2current(c_homeo, soruce_vtx_cdc).squeeze(0)
        # ! clamp all vtx to unit cube
        surface_vtx = torch.clamp(surface_vtx, -1.0, 1.0)
        surface_vtx = surface_vtx.detach().cpu().squeeze(0).numpy()  # T,Pts,3
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

    def map_pc2cdc(self, batch, bid):
        net = self.network.module if self.__dataparallel_flag__ else self.network
        _c_t = batch["c_t"][bid].unsqueeze(0)
        _obs_pc = batch["seq_pc"][bid].unsqueeze(0)
        _, input_cdc_un = net.map2canonical(_c_t.transpose(2, 1), _obs_pc, return_uncompressed=True)
        input_cdc_un = input_cdc_un.detach().cpu().numpy().squeeze(0)
        return input_cdc_un

    def _postprocess_after_optim(self, batch):
        # eval iou
        if "occ_hat_iou" in batch.keys():
            report = {}

            occ_pred = batch["occ_hat_iou"].detach().cpu().numpy()
            occ_gt = batch["model_input"]["points.occ"].detach().cpu().numpy()
            iou = eval_iou(occ_gt, occ_pred, threshold=self.iou_threshold)
            # make metric tensorboard
            iou_meanbatch = iou.mean(axis=0)
            batch["iou"] = iou_meanbatch.sum() / len(iou_meanbatch)
            for i in range(len(iou_meanbatch)):
                batch["iou_t%d" % i] = iou_meanbatch[i]
            # make report
            iou_meantime = iou.mean(axis=1)
            report["iou"] = iou_meantime.tolist()
            for t in range(iou.shape[1]):
                report["iou_t%d" % t] = iou[:, t].tolist()
            batch["running_metric_report"] = report

        if "c_t" in batch.keys():
            self.network.eval()
            phase = batch["model_input"]["phase"]
            viz_flag = batch["model_input"]["viz_flag"]
            test_results_m, test_results_t = {}, {}
            B, T = batch["seq_t"].shape
            with torch.no_grad():
                # prepare viz mesh lists
                for t in range(T):
                    batch["mesh_t%d" % t] = []
                batch["cdc_mesh"] = []
                rendered_fig_list, video_list = [], []
                for bid in range(B):
                    # generate mesh
                    mesh_t_list, surface_vtx, mesh_cdc = self.generate_mesh(
                        batch["c_t"][bid], batch["c_g"][bid], batch["seq_t"][bid]
                    )
                    for t in range(0, T):  # if generate mesh, then save it
                        batch["mesh_t%d" % t].append(mesh_t_list[t])
                    batch["cdc_mesh"].append(mesh_cdc)

                    if phase.startswith("test"):
                        # evaluate the generated mesh list
                        eval_dict_mean, eval_dict_t = eval_oflow_all(
                            pcl_tgt=batch["model_input"]["points_mesh"][bid].detach().cpu().numpy(),
                            points_tgt=batch["model_input"]["points"][bid].detach().cpu().numpy(),
                            occ_tgt=batch["model_input"]["points.occ"][bid].detach().cpu().numpy(),
                            mesh_t_list=mesh_t_list,
                            evaluator=self.evaluator,
                            corr_project_to_final_mesh=self.corr_eval_project_to_final_mesh,
                        )
                        # record the batch results
                        for k, v in eval_dict_mean.items():
                            if k not in test_results_m.keys():
                                test_results_m[k] = [v]
                            else:
                                test_results_m[k].append(v)
                        for k, v in eval_dict_t.items():
                            if k not in test_results_t.keys():
                                test_results_t[k] = [v]
                            else:
                                test_results_t[k].append(v)
                        logging.info("Test results: {}".format(test_results_m))

                    # render an image of the mesh
                    if viz_flag:
                        scale_cdc = True
                        if "viz_cdc_scale" in self.cfg["logging"].keys():
                            scale_cdc = self.cfg["logging"]["viz_cdc_scale"]
                        fig_t_list = viz_cdc(
                            vtx_list=surface_vtx,
                            cdc_vtx=np.asarray(mesh_cdc.vertices),
                            face=np.array(mesh_t_list[0].faces).copy(),
                            pc=batch["seq_pc"][bid].detach().cpu().numpy(),
                            mesh_viz_interval=self.cfg["logging"]["mesh_viz_interval"],
                            cdc_input=self.map_pc2cdc(batch, bid),
                            scale_cdc=scale_cdc,
                        )
                        cat_fig = np.concatenate(fig_t_list, axis=0).transpose(2, 0, 1)
                        cat_fig = np.expand_dims(cat_fig, axis=0).astype(np.float) / 255.0
                        rendered_fig_list.append(cat_fig)
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
                    batch["flow_video"] = torch.Tensor(
                        np.concatenate(video_list, axis=0)
                    )  # B,T,3,H,W
            if phase.startswith("test"):
                batch["results_m"] = test_results_m
                batch["results_t"] = test_results_t
        del batch["model_input"]
        return batch


class CaDeX_DFAU(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)

        self.t_perm_inv = cfg["model"]["t_perm_inv"]
        if self.t_perm_inv:
            homeomorphism_encoder = ResnetPointnet(dim=3, **cfg["model"]["homeomorphism_encoder"])
        else:
            homeomorphism_encoder = SpatioTemporalResnetPointnetCDC(
                dim=3,
                **cfg["model"]["homeomorphism_encoder"],
                global_geometry=True,
                # * Note, here we still set global_geometry = True, but don't use this c_g
            )

        H = NVP_v2_5
        if "homeo_type" in cfg["model"].keys():
            assert cfg["model"]["homeo_type"] in ["NVP", "NICE"]
            if cfg["model"]["homeo_type"] == "NICE":
                logging.info("Use NICE!")
                H = NICE

        self.network_dict = torch.nn.ModuleDict(
            {
                "homeomorphism_encoder": homeomorphism_encoder,
                "canonical_geometry_encoder": ResnetPointnet(
                    dim=3, **cfg["model"]["canonical_geometry_encoder"]
                ),
                "canonical_geometry_decoder": DecoderCBatchNorm(
                    dim=3, z_dim=0, **cfg["model"]["canonical_geometry_decoder"]
                ),
                "homeomorphism_decoder": H(**cfg["model"]["homeomorphism_decoder"]),
            }
        )
        for k in self.network_dict:
            logging.info(
                "{} params in {}".format(
                    sum(param.numel() for param in self.network_dict[k].parameters()), k
                )
            )

        # ! Note here we bounded the cdc in a sigmoid cube. Is this necessary?
        self.compress_cdc = cfg["model"]["compress_cdc"]

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

        seq_t, seq_pc, = (
            input_pack["inputs.time"],
            input_pack["inputs"],
        )
        B, T = seq_t.shape

        # encode Hoemo condition
        if self.t_perm_inv:
            c_t = self.network_dict["homeomorphism_encoder"](seq_pc.reshape(B * T, -1, 3))
            c_t = c_t.reshape(B, T, -1)
        else:
            _, c_t = self.network_dict["homeomorphism_encoder"](seq_pc)  # B,C; B,T,C

        # tranform observation to CDC and encode canonical geometry
        inputs_cdc = self.map2canonical(c_t.transpose(2, 1), input_pack["inputs"])  # B,T,N,3
        c_g = self.network_dict["canonical_geometry_encoder"](inputs_cdc.reshape(B, -1, 3))

        # visualize
        if viz_flag:
            output["c_t"] = c_t.detach()
            output["c_g"] = c_g.detach()
            output["seq_t"] = seq_t.detach()
            output["seq_pc"] = seq_pc.detach()

        # if test, direct return
        if phase.startswith("test"):
            output["c_t"] = c_t.detach()
            output["c_g"] = c_g.detach()
            output["seq_t"] = seq_t.detach()
            return output

        # get deformation condition for traning steps
        idx = (input_pack["points.time"] * (seq_t.shape[1] - 1)).long()  # B,t
        c_homeomorphism = torch.gather(
            c_t, dim=1, index=idx.unsqueeze(2).expand(-1, -1, c_t.shape[-1])
        ).transpose(
            2, 1
        )  # B,C,T

        # transform to canonical frame
        cdc, uncompressed_cdc = self.map2canonical(
            c_homeomorphism, input_pack["points"], return_uncompressed=True
        )  # B,T,N,3
        shift = (uncompressed_cdc - input_pack["points"]).norm(dim=3)

        # reconstruct in canonical space
        pr = self.decode_by_cdc(observation_c=c_g, query=cdc)
        occ_hat = pr.probs
        reconstruction_loss_i = torch.nn.functional.binary_cross_entropy(
            occ_hat, input_pack["points.occ"], reduction="none"
        )
        reconstruction_loss = reconstruction_loss_i.mean()

        # compute corr loss
        if self.use_corr_loss:
            _, cdc_first_frame_un = self.map2canonical(
                c_t[:, 0].unsqueeze(2),
                input_pack["pointcloud"][:, 0].unsqueeze(1),
                return_uncompressed=True,
            )  # B,1,M,3
            cdc_forward_frames = self.map2current(
                c_t[:, 1:].transpose(2, 1),
                cdc_first_frame_un.expand(-1, T - 1, -1, -1),
                compressed=False,
            )
            corr_loss_i = torch.abs(
                cdc_forward_frames - input_pack["pointcloud"][:, 1:].detach()
            ).sum(-1)
            corr_loss = corr_loss_i.mean()

        output["batch_loss"] = reconstruction_loss
        output["loss_recon"] = reconstruction_loss.detach()
        output["loss_recon_i"] = reconstruction_loss_i.detach().reshape(-1)
        if self.use_corr_loss:
            output["batch_loss"] = output["batch_loss"] + corr_loss
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
