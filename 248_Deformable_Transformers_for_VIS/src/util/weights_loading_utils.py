import numpy as np
import torch

IDS_COCO_TO_YVIS = np.array(
    [1, -1, -1, -1, 41, -1, 3, 18, -1, -1, -1, -1, -1, 17, 21, -1, 7, 19, -1, 23, -1, 4, -1, -1, -1, -1, 42, 5, 8, 24,
     -1, 22, 36, 9, -1, 74, -1, -1, -1, -1, -1])


# Shifts class embed neurons to start with label=0 at neuron=0. Used to convert weights from deformable_detr official repo.
def shift_class_neurons(checkpoint: dict):
    checkpoint_state_dict = {}
    for k, v in checkpoint.items():
        if 'class_embed' in k:
            resume_value = v.clone()
            resume_value[:-1] = v[1:]
            resume_value[-1] = v[0]
            print(f"Load {k} {tuple(v.shape)} from resume model and "
                  "shift class embed neurons to start with label=0 at neuron=0.")

            checkpoint_state_dict[k] = resume_value
        else:
            checkpoint_state_dict[k] = v

    return checkpoint_state_dict


def adapt_weights_mask_head(checkpoint: dict, model_state_dict: dict):
    checkpoint_state_dict = {}
    for k, v in checkpoint.items():
        if not k.startswith("def_detr") and "bbox_attention" not in k and "mask_head" not in k:
            checkpoint_state_dict[f"def_detr.{k}"] = v
        else:
            checkpoint_state_dict[k] = v

    resume_state_dict = {}
    for k, v in model_state_dict.items():
        if k not in checkpoint_state_dict:
            resume_value = v
            print(f'Load {k} {tuple(v.shape)} from scratch.')

        elif k in checkpoint_state_dict and v.shape != checkpoint_state_dict[k].shape:
            raise NotImplementedError(f"{k}: resume state dict shape {checkpoint_state_dict[k].shape} incompatible with current model shape {v.shape}")

        else:
            resume_value = checkpoint_state_dict[k]

        resume_state_dict[k] = resume_value

    return resume_state_dict


def adapt_weights_devis(checkpoint: dict, model_state_dict: dict, lvl_res: int, focal_loss: bool, finetune_class_logits: bool, num_frames: int, finetune_query_embds: bool,
                        finetune_temporal_modules: bool, enc_connect_all_frames: bool, enc_temporal_window, enc_n_temporal_points: int,
                        dec_n_temporal_points: int):
    checkpoint_state_dict = {}

    for k, v in checkpoint.items():

        if "def_detr" not in k and ("transformer" in k or "class_embed" in k or "bbox_embed" in k or "input_proj" in k or "query_embed" in k or "backbone" in k):
            checkpoint_state_dict[f"def_detr.{k}"] = v

            # if finetune_temporal_modules and (("transformer.encoder" in k and "self_attn" in k) or ("transformer.decoder" in k and "cross_attn" in k)) \
            #         and 'value_proj' not in k and 'output_proj' not in k:
            #     name = "def_detr." + ".".join(k.split(".")[:5]) + ".temporal_" + ".".join(k.split(".")[5:])
            #     checkpoint_state_dict[name] = torch.clone(v)

            # Use /32 res when only 1 resolution is used
            if lvl_res == 1 and "input_proj.2" in k:
                name = "def_detr." + k.split(".")[0] + ".0." + ".".join(k.split(".")[2:])
                checkpoint_state_dict[name] = torch.clone(v)

        else:
            checkpoint_state_dict[k] = v

        if finetune_temporal_modules and (("transformer.encoder" in k and "self_attn" in k) or
                                          ("transformer.decoder" in k and "cross_attn" in k)) \
                and 'value_proj' not in k and 'output_proj' not in k:

            if "def_detr" in k:
                name = ".".join(k.split(".")[:6]) + ".temporal_" + ".".join(
                    k.split(".")[6:])
            else:
                name = "def_detr." + ".".join(k.split(".")[:5]) + ".temporal_" + ".".join(
                    k.split(".")[5:])

            checkpoint_state_dict[name] = torch.clone(v)

    resume_state_dict = {}
    for k, v in model_state_dict.items():
        if k not in checkpoint_state_dict or ('query_embed' in k and not finetune_query_embds) or ('class_embed' in k and not finetune_class_logits):
            resume_value = v
            print(f'Load {k} {tuple(v.shape)} from scratch.')

        elif 'query_embed' in k and finetune_query_embds:
            print(f"Adapting value {k}")
            checkpoint_value = checkpoint_state_dict[k]
            num_queries_to_gather = v.shape[0] // num_frames
            if num_queries_to_gather < checkpoint_value.shape[0]:
                assert not checkpoint_value.shape[0] % num_queries_to_gather
                picked_idx = []
                picked_percentage = checkpoint_value.shape[0] // num_queries_to_gather
                for i in range(checkpoint_value.shape[0]):
                    if i % picked_percentage == 0:
                        picked_idx.append(i)
                embed_idx = torch.tensor(picked_idx)[:v.shape[0]]
                resume_value = checkpoint_value[embed_idx].repeat(num_frames, 1)
            elif num_queries_to_gather == checkpoint_value.shape[0]:
                resume_value = checkpoint_value
            else:
                raise NotImplementedError


        elif k in checkpoint_state_dict and v.shape != checkpoint_state_dict[k].shape:
            print(f"Adapting value {k}")
            checkpoint_value = checkpoint_state_dict[k]

            if 'level_embed' in k:
                resume_value = checkpoint_value[:v.shape[0]]

            elif 'self_attn.attention_weights' in k or 'cross_attn.attention_weights' in k:
                if 'bias' not in k:
                    att_weights = checkpoint_value.view(8, 4, 4, 256)
                    resume_value = att_weights[:, :lvl_res].reshape(-1, 256)
                else:
                    att_weights = checkpoint_value.view(8, 4, 4)
                    resume_value = att_weights[:, :lvl_res].reshape(-1)

            elif 'self_attn.sampling_offsets' in k or 'cross_attn.sampling_offsets' in k:
                if 'bias' not in k:
                    sampling_offsets = checkpoint_value.view(8, 4, 4, 2, 256)
                    resume_value = sampling_offsets[:, :lvl_res].reshape(-1, 256)
                else:
                    sampling_offsets = checkpoint_value.view(8, 4, 4, 2)
                    resume_value = sampling_offsets[:, :lvl_res].reshape(-1)

            elif 'class_embed' in k:
                ids_coco_to_yt_vis = IDS_COCO_TO_YVIS[:-1] if focal_loss else IDS_COCO_TO_YVIS
                indices_to_gather = ids_coco_to_yt_vis != -1
                logits_to_gather = ids_coco_to_yt_vis[indices_to_gather] - 1
                tmp_v = v.clone().to('cpu')
                tmp_v[indices_to_gather] = checkpoint_value[logits_to_gather]
                resume_value = tmp_v

            elif 'temporal' in k and finetune_temporal_modules and ('self_attn' in k or 'cross_attn' in k):
                if 'transformer.encoder' in k:
                    num_temporal_frames = num_frames - 1 if enc_connect_all_frames else enc_temporal_window
                    num_temporal_points = enc_n_temporal_points
                elif 'transformer.decoder' in k:
                    num_temporal_frames = num_frames - 1
                    num_temporal_points = dec_n_temporal_points
                else:
                    raise NotImplementedError

                if 'sampling_offsets' in k:
                    if 'bias' not in k:
                        sampling_offsets = checkpoint_value.view(8, 1, 4, 4, 2, 256).repeat(1, num_temporal_frames, 1, 1, 1, 1)
                        resume_value = sampling_offsets[:, :, :lvl_res, :num_temporal_points].reshape(-1, 256)
                    else:
                        sampling_offsets = checkpoint_value.view(8, 1, 4, 4, 2).repeat(1, num_temporal_frames, 1, 1, 1)
                        resume_value = sampling_offsets[:, :, :lvl_res, :num_temporal_points].reshape(-1)

                else:
                    if 'bias' not in k:
                        att_weights = checkpoint_value.view(8, 1, 4, 4, 256).repeat(1, num_temporal_frames, 1, 1, 1)
                        resume_value = att_weights[:, :, :lvl_res, :num_temporal_points].reshape(-1, 256)

                    else:
                        att_weights = checkpoint_value.view(8, 1, 4, 4).repeat(1, num_temporal_frames, 1, 1)
                        resume_value = att_weights[:, :, :lvl_res, :num_temporal_points].reshape(-1)

            else:
                raise NotImplementedError(f"Shape mismatch for parameter {k}: Model shape: {v.shape} Checkpoint shape: {checkpoint_value.shape}")

        else:
            resume_value = checkpoint_state_dict[k]

        resume_state_dict[k] = resume_value

    for key in checkpoint_state_dict.keys():
        if key not in resume_state_dict.keys():
            print(f"Ignoring {key} from checkpoint")

    return resume_state_dict
