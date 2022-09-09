"""
Training script of DeVIS
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
import random
import warnings
from contextlib import redirect_stdout
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import src.util.misc as utils
from src.datasets import build_dataset
from src.engine import evaluate_coco, inference_vis, train_one_epoch
from src.models import build_model, build_tracker
from src.util.weights_loading_utils import shift_class_neurons, adapt_weights_devis, \
    adapt_weights_mask_head
from src.util.visdom_vis import build_visualizers, get_vis_win_names
from src.config import get_cfg_defaults


def get_args_parser():
    parser = argparse.ArgumentParser('DeVIS argument parser', add_help=False)

    parser.add_argument('--config-file', help="Configuration file path")
    parser.add_argument('--eval-only', action='store_true', help="Run test only")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed '
                                                             'training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def sanity_check(cfg):
    assert cfg.MODEL.LOSS.FOCAL_LOSS, "Softmax classification loss not implemented, " \
                                      "see src/models/critertion.py loss_labels() "
    assert cfg.MODEL.LOSS.FOCAL_LOSS

    if cfg.MODEL.LOSS.MASK_AUX_LOSS:
        assert min(cfg.MODEL.LOSS.MASK_AUX_LOSS) >= 0 and max(cfg.MODEL.LOSS.MASK_AUX_LOSS) <= 4, \
            f"Available MODEL.LOSS.MASK_AUX_LOSS levels : [0, 1, 2, 3, 4]"

    if cfg.MODEL.LOSS.AUX_LOSS_WEIGHTING:
        assert cfg.MODEL.TRANSFORMER.DECODER_LAYERS == 6, "MODEL.LOSS.AUX_LOSS_WEIGHTING  weights " \
                                                          "config available only for 6 layers "

    if cfg.TEST.USE_TOP_K:
        assert cfg.MODEL.LOSS.FOCAL_LOSS, "TopK can only be used with FOCAL_LOSS"
    else:
        if cfg.DATASETS.TYPE == 'vis':
            if cfg.TEST.NUM_OUT != (cfg.MODEL.NUM_QUERIES // cfg.MODEL.DEVIS.NUM_FRAMES):
                warnings.warn("TEST.NUM_OUT != to number of queries per frame for DeVIS, "
                              "automatically setting it")

        else:
            if cfg.TEST.NUM_OUT != cfg.MODEL.NUM_QUERIES:
                warnings.warn("TEST.NUM_OUT != to number of queries, automatically setting it")

    if cfg.DATASETS.TYPE == 'vis':
        assert cfg.MODEL.DEVIS.NUM_FRAMES > 1, "MODEL.DEVIS.NUM_FRAMES must be higher than 1"
        assert not (cfg.MODEL.NUM_QUERIES % cfg.MODEL.DEVIS.NUM_FRAMES), \
            "MODEL.NUM_QUERIES must be divisible by MODEL.DEVIS.NUM_FRAMES for VIS training"
        if cfg.SOLVER.DEVIS.FINETUNE_QUERY_EMBEDDINGS:
            assert not (300 % (cfg.MODEL.NUM_QUERIES // cfg.MODEL.DEVIS.NUM_FRAMES)), \
                "Number of queries per frame must be divisible by 300 for SOLVER.DEVIS.FINETUNE_QUERY_EMBEDDINGS"

        assert cfg.SOLVER.BATCH_SIZE == 1, "Batch size > 1 not implemented for VIS training"
        assert cfg.TEST.CLIP_TRACKING.STRIDE < cfg.MODEL.DEVIS.NUM_FRAMES, \
            "Clip tracking stride can not be higher than the clip size"

    if cfg.TEST.INPUT_FOLDER:
        assert len(cfg.TEST.EPOCHS_TO_EVAL) >= 1, \
            "TEST.EPOCHS_TO_EVAL must contain at least 1 epoch number"

    assert not (cfg.MODEL.WITH_BBX_REFINE and cfg.MODEL.WITH_REF_POINT_REFINE), \
        "MODEL.WITH_BBX_REFINE can not be activated together with cfg.MODEL.WITH_BBX_REFINE, select one of the two"


def main(args, cfg):
    sanity_check(cfg)

    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = cfg.SEED + utils.get_rank()

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset, num_classes = build_dataset(image_set="TRAIN", cfg=cfg)
    dataset_val, _ = build_dataset(image_set="VAL", cfg=cfg)
    model, criterion, postprocessors = build_model(num_classes, device, cfg)
    model.to(device)

    visualizers = {}
    if cfg.DATASETS.TYPE != 'vis' or not args.eval_only:
        visualizers = build_visualizers(cfg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    tracker = None
    if cfg.DATASETS.TYPE == 'vis':
        tracker = build_tracker(model, cfg)

    n_total_params = sum(p.numel() for p in model.parameters())
    print(f'Total num params: {n_total_params}')

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.SOLVER.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                   worker_init_fn=utils.seed_worker, generator=g)

    data_loader_val = DataLoader(dataset_val, cfg.SOLVER.BATCH_SIZE, sampler=sampler_val,
                                 collate_fn=utils.val_collate if cfg.DATASETS.TYPE == 'vis' else utils.collate_fn,
                                 num_workers=cfg.NUM_WORKERS
                                 )

    output_dir = Path(cfg.OUTPUT_DIR)

    if args.eval_only:
        if cfg.DATASETS.TYPE == 'vis':
            # Used for visualization purposes only
            selected_videos = ''
            if cfg.TEST.VIZ.VIDEO_NAMES:
                selected_videos = cfg.TEST.VIZ.VIDEO_NAMES.split(",")

            # Allow all checkpoints input_folder test
            if cfg.TEST.INPUT_FOLDER:
                for epoch_to_eval in cfg.TEST.EPOCHS_TO_EVAL:
                    print(f"************* Starting validation epoch {epoch_to_eval} *************")
                    checkpoint_path = os.path.join(cfg.TEST.INPUT_FOLDER,
                                                   f"checkpoint_epoch_{epoch_to_eval}.pth")
                    assert os.path.exists(
                        checkpoint_path), f"Checkpoint path {checkpoint_path} DOESN'T EXIST"
                    out_folder_name = f"val_epoch_{epoch_to_eval}"
                    resume_state_dict = torch.load(checkpoint_path, map_location=device)['model']
                    model_without_ddp.load_state_dict(resume_state_dict, strict=True)

                    _ = inference_vis(
                        tracker, data_loader_val, dataset_val, visualizers, device,
                        output_dir, out_folder_name, epoch_to_eval, selected_videos)

            else:
                out_folder_name = cfg.TEST.SAVE_PATH
                resume_state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location=device)['model']
                model_without_ddp.load_state_dict(resume_state_dict, strict=True)

                _ = inference_vis(
                    tracker, data_loader_val, dataset_val, visualizers, device, output_dir,
                    out_folder_name, 0, selected_videos)

        else:
            checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location=device)['model']
            if cfg.MODEL.SHIFT_CLASS_NEURON:
                checkpoint = shift_class_neurons(checkpoint)

            if cfg.MODEL.MASK_ON:
                checkpoint = adapt_weights_mask_head(checkpoint, model_without_ddp.state_dict())

            model_without_ddp.load_state_dict(checkpoint, strict=True)
            _, coco_evaluator = evaluate_coco(
                model, criterion, postprocessors, data_loader_val, device, output_dir,
                visualizers['val'], cfg.VISDOM_AND_LOG_INTERVAL, cfg.START_EPOCH
            )

            if cfg.OUTPUT_DIR:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    if cfg.SOLVER.FROZEN_PARAMS:
        for n, p in model_without_ddp.named_parameters():
            if utils.match_name_keywords(n, cfg.SOLVER.FROZEN_PARAMS):
                p.requires_grad_(False)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    utils.print_training_params(model_without_ddp, cfg)
    print(f'Number of training params: {n_train_params}')

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not utils.match_name_keywords(n, cfg.SOLVER.BACKBONE_NAMES +
                                                  cfg.SOLVER.LR_LINEAR_PROJ_NAMES +
                                                  cfg.SOLVER.LR_MASK_HEAD_NAMES +
                                                  cfg.SOLVER.DEVIS.LR_TEMPORAL_LINEAR_PROJ_NAMES)
                 and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR,
        },

        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       utils.match_name_keywords(n, cfg.SOLVER.BACKBONE_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.LR_BACKBONE,
        },

        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       utils.match_name_keywords(n,
                                                 cfg.SOLVER.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_LINEAR_PROJ_MULT,
        },

        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       utils.match_name_keywords(n,
                                                 cfg.SOLVER.LR_MASK_HEAD_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_MASK_HEAD_MULT,
        },

        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       utils.match_name_keywords(n,
                                                 cfg.SOLVER.DEVIS.LR_TEMPORAL_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.DEVIS.LR_TEMPORAL_LINEAR_PROJ_MULT,
        }

    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.BASE_LR,
                                  weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS)

    best_val_stats = None
    start_epoch = cfg.START_EPOCH
    if cfg.MODEL.WEIGHTS:
        if cfg.MODEL.WEIGHTS.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.MODEL.WEIGHTS, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')

        checkpoint_state_dict = checkpoint['model']
        model_state_dict = model_without_ddp.state_dict()

        # We assume that when resuming training no changes need to be made on the weights
        if not cfg.SOLVER.RESUME_OPTIMIZER:
            if cfg.DATASETS.TYPE == 'vis':
                checkpoint_state_dict = adapt_weights_devis(checkpoint_state_dict, model_state_dict,
                                                            cfg.MODEL.NUM_FEATURE_LEVELS,
                                                            cfg.MODEL.LOSS.FOCAL_LOSS,
                                                            cfg.SOLVER.DEVIS.FINETUNE_CLASS_LOGITS,
                                                            cfg.MODEL.DEVIS.NUM_FRAMES,
                                                            cfg.SOLVER.DEVIS.FINETUNE_QUERY_EMBEDDINGS,
                                                            cfg.SOLVER.DEVIS.FINETUNE_TEMPORAL_MODULES,
                                                            cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_CONNECT_ALL_FRAMES,
                                                            cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_TEMPORAL_WINDOW,
                                                            cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_N_POINTS_TEMPORAL_FRAME,
                                                            cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.DEC_N_POINTS_TEMPORAL_FRAME
                                                            )

            else:
                if cfg.MODEL.SHIFT_CLASS_NEURON:
                    checkpoint_state_dict = shift_class_neurons(checkpoint_state_dict)

                if cfg.MODEL.MASK_ON:
                    checkpoint_state_dict = adapt_weights_mask_head(checkpoint_state_dict,
                                                                    model_state_dict)

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint_state_dict,
                                                                          strict=False)

        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))

        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        # RESUME OPTIM
        if not args.eval_only and cfg.SOLVER.RESUME_OPTIMIZER:
            if 'optimizer' in checkpoint:
                for c_p, p in zip(checkpoint['optimizer']['param_groups'], param_dicts):
                    c_p['lr'] = p['lr']
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_val_stats' in checkpoint:
                best_val_stats = checkpoint['best_val_stats']

        if not args.eval_only and cfg.RESUME_VIS and 'vis_win_names' in checkpoint:
            for k, v in visualizers.items():
                for k_inner in v.keys():
                    visualizers[k][k_inner].win = checkpoint['vis_win_names'][k][k_inner]

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, visualizers['train'],
            cfg.VISDOM_AND_LOG_INTERVAL, cfg.SOLVER.GRAD_CLIP_MAX_NORM
        )

        lr_scheduler.step()

        checkpoint_paths = [output_dir / 'checkpoint.pth']

        if cfg.SOLVER.CHECKPOINT_INTERVAL and not epoch % int(cfg.SOLVER.CHECKPOINT_INTERVAL):
            checkpoint_paths.append(output_dir / f"checkpoint_epoch_{epoch}.pth")

        # # VAL
        if (epoch == 1 or not epoch % cfg.TEST.EVAL_PERIOD) and epoch >= cfg.TEST.START_EVAL_EPOCH:

            if cfg.DATASETS.TYPE == 'vis':
                out_folder_name = os.path.join(cfg.TEST.SAVE_PATH, f"epoch_{epoch}")
                _ = inference_vis(
                    tracker, data_loader_val, dataset_val, visualizers['val'], device, output_dir,
                    out_folder_name, epoch, '')
                # TODO: If val_dataset has_gt save additionally best epoch

            else:
                val_stats, _ = evaluate_coco(
                    model, criterion, postprocessors, data_loader_val, device,
                    output_dir, visualizers['val'], cfg.VISDOM_AND_LOG_INTERVAL, epoch)

                stat_names = ['BBOX_AP_IoU_0_50-0_95', ]
                if cfg.MODEL.MASK_ON:
                    stat_names.extend(['MASK_AP_IoU_0_50-0_95', ])

                if best_val_stats is None:
                    best_val_stats = val_stats
                best_val_stats = [best_stat if best_stat > stat else stat
                                  for best_stat, stat in zip(best_val_stats, val_stats)]

                for b_s, s, n in zip(best_val_stats, val_stats, stat_names):
                    if b_s == s:
                        checkpoint_paths.append(output_dir / f"checkpoint_best_{n}.pth")

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': cfg,
                'vis_win_names': get_vis_win_names(visualizers),
                'best_val_stats': best_val_stats
            }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeVIS training and evaluation script',
                                     parents=[get_args_parser()])
    args_ = parser.parse_args()

    cfg_ = get_cfg_defaults()
    cfg_.merge_from_file(args_.config_file)
    cfg_.merge_from_list(args_.opts)
    cfg_.freeze()
    if cfg_.OUTPUT_DIR:
        Path(cfg_.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg_.OUTPUT_DIR, 'config.yaml'), 'w') as yaml_file:
            with redirect_stdout(yaml_file):
                print(cfg_.dump())

    main(args_, cfg_)
