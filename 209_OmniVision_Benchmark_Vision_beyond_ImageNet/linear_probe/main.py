import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import os
import argparse
import pickle
import yaml
from PIL import Image
from easydict import EasyDict

import spring.linklink as link
from torch.utils.data import DataLoader
# from samplers import DistributedGivenIterationSampler
import torch.distributed as dist

import dataset
import models
from utils import create_logger, parameters_string, random_seed

import subprocess


def extract(model, loader):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            f = model(inputs).detach().cpu()
            # print("CUR Feature:",f.shape)
            feats.append(f)
            labels.append(targets)
            if (i + 1) % 100 == 0:
                logger.info('{} / {}'.format(i + 1, len(loader)))
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def test(net, feats, labels):
    net.eval()
    root = './scores'
    data_name = '_'.join(args.config.split('/')[-2:]).split('.')[0]
    output_dir = os.path.join(root, model_config.model.type, data_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_classes = config.data.num_classes
    cfmatrix = torch.zeros(num_classes, num_classes)

    model_output = net(feats)           ##
    softmax = nn.Softmax(dim=1)         ##
    logits = softmax(model_output)      ##
    _, preds = model_output.max(dim=1)  ##

    res_file = os.path.join(output_dir, f'scores.txt.rank{rank}')   ##
    writer = open(res_file, 'w')                            ##

    for sample_ix, feat in enumerate(feats):
        res = {
            'label': int(labels[sample_ix]),
            'score': str(logits[sample_ix].tolist())
        }
        writer.write(json.dumps(res, ensure_ascii=False) + '\n')
    writer.flush()

    # _, preds = net(feats).max(dim=1)

    for true_label, pred_label in zip(labels.view(-1), preds.view(-1)):
        cfmatrix[true_label.long(), pred_label.long()] += 1

    per_class_acc = (cfmatrix.diag() / cfmatrix.sum(dim=1)).mean()

    acc = preds.eq(labels).float().sum().item() / labels.numel()

    net.train()

    return acc, per_class_acc
    


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


if __name__ ==  '__main__':
    # rank, world_size = dist_init()
    _init_dist_slurm('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    parser = argparse.ArgumentParser(description='Linear Probe Evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--datetime', type=str, default='19700101-00:00:00')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)
    with open(args.model_config) as f2:
        model_config = yaml.load(f2, yaml.FullLoader)

    config = EasyDict(config)
    model_config = EasyDict(model_config)


    if hasattr(config, 'text'):
        assert world_size == 1, 'Zero-shot evaluation currently only requires one process.'
        sub_folder = 'zeroshot'
    else:
        config.lam = config.lam[rank]
        sub_folder = 'lam{}'.format(config.lam)
    
    config_name = args.config.split('/')[-1].rsplit('.', 1)[0]
    config.save_path_dated = os.path.join(config.save_path, config_name + '_' + args.datetime, sub_folder)
    os.makedirs(config.save_path_dated, exist_ok=True)
    
    logger = create_logger('global', os.path.join(config.save_path_dated, 'log.txt'), rank)
    logger.info(config)
    logger.info(model_config)

    # Data
    logger.info('==> Building dataloader..')
    input_size = model_config.data.get('input_size', 224)
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(
            model_config.data.get('mean', [0.485, 0.456, 0.406]),
            model_config.data.get('std', [0.229, 0.224, 0.225])
        )
    ])
    
    if not hasattr(config, 'text'):
        trainset = getattr(dataset, config.data.train.type)(
            transform=transform,
            **config.data.get('kwargs', {}),
            **config.data.train.get('kwargs', {})
        )
        logger.info('--- Train Set ---')
        logger.info(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.data.batch_size, shuffle=False, num_workers=1)

        if hasattr(config.data, 'val'):
            valset = getattr(dataset, config.data.val.type)(
                transform=transform,
                **config.data.get('kwargs', {}),
                **config.data.val.get('kwargs', {})
            )
            logger.info('--- Val Set ---')
            logger.info(valset)
            valloader = torch.utils.data.DataLoader(valset, batch_size=config.data.batch_size, shuffle=False, num_workers=1)

    testset = getattr(dataset, config.data.test.type)(
        transform=transform,
        **config.data.get('kwargs', {}),
        **config.data.test.get('kwargs', {})
    )
    logger.info('--- Test Set ---')
    logger.info(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.data.batch_size, shuffle=False, num_workers=1)

    # Set random seed
    random_seed(config.manual_seed)
            
    # Backbone
    logger.info('==> Building backbone (frozen)..')
    backbone = getattr(models, model_config.model.type)(**model_config.model.get('kwargs', {}))
    backbone.requires_grad_(False)
    backbone.cuda()
    
    logger.info(parameters_string(backbone))
    logger.info(backbone)
    
    # Extract features
    if not hasattr(config, 'text'):
        logger.info('==> Extracting train image features..')
        train_feats, train_labels = extract(backbone, trainloader)
        train_feats = train_feats.cuda()
        logger.info(train_feats.shape)

        if hasattr(config.data, 'val'):
            logger.info('==> Extracting val image features..')
            val_feats, val_labels = extract(backbone, valloader)
            val_feats = val_feats.cuda()
            logger.info(val_feats.shape)

    logger.info('==> Extracting test image features..')
    test_feats, test_labels = extract(backbone, testloader)
    test_feats = test_feats.cuda()
    logger.info(test_feats.shape)

    if hasattr(config, 'text'):
        # Read class names
        idx_to_class = pickle.load(open(config.text.idx_to_class, 'rb'))

        # Text model
        logger.info('==> Building text model..')
        text_model = getattr(models, config.text.model.type)(**config.text.model.get('kwargs', {}))
        text_model.requires_grad_(False)
        text_model.cuda()
        
        logger.info(parameters_string(text_model))
        logger.info(text_model)

        # Extract zero-shot weights
        logger.info('==> Extracting zero-shot weights from texts..')
        text_model.eval()
        zeroshot_weights = []
        for idx in range(config.data.num_classes):
            texts = [_.format(idx_to_class[idx]) for _ in config.text.templates]
            class_embeddings = text_model(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0, keepdim=True)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(torch.cat([class_embeddings, class_embedding], dim=0))
            if (idx + 1) % 100 == 0:
                logger.info('{} / {}'.format(idx + 1, config.data.num_classes))
        descriptions = ['template "{}"'.format(_) for _ in config.text.templates] + ['ensemble of templates']
        zeroshot_weights = torch.stack(zeroshot_weights, dim=2)
        logger.info(zeroshot_weights.shape)

        test_feats /= test_feats.norm(dim=-1, keepdim=True)
        for idx, description in enumerate(descriptions):
            logits = test_feats @ zeroshot_weights[idx]
            _, preds = logits.max(dim=1)
            acc = preds.eq(test_labels).float().sum().item() / test_labels.numel()
            logger.info('test acc with {}: {}'.format(description, acc))
    else:
        # Linear-probe classifier
        logger.info('==> Building classifier..')
        net = nn.Linear(model_config.model.output_dim, config.data.num_classes)
        net.cuda()
        
        logger.info(parameters_string(net))
        logger.info(net)
            
        criterion = nn.CrossEntropyLoss().cuda()
            
        optimizer = optim.LBFGS(net.parameters())
        
        best_acc = 0
        best_per = 0
        net.train()

        def train_feat_split(seg_num, train_feats):
            total_size = train_feats.size()[0]
            step = total_size // seg_num
            start = 0
            end = start + step
            feats = [train_feats[start:end, :]]

            for i in range(seg_num - 1):
                start = end
                end += step
                if end > total_size:
                    end = total_size
                if i == seg_num - 2:
                    end = total_size
                feat = train_feats[start:end, :]
                feats.append(feat)
            return feats

        def train_labels_split(seg_num, train_feats):
            total_size = train_feats.size()[0]
            step = total_size // seg_num
            start = 0
            end = start + step
            feats = [train_feats[start:end]]

            for i in range(seg_num - 1):
                start = end
                end += step
                if end > total_size:
                    end = total_size
                if i == seg_num - 2:
                    end = total_size
                feat = train_feats[start:end]
                feats.append(feat)
            return feats

        seg_num = 10
        print('labels size is ', train_labels.size())

        feats = train_feat_split(seg_num=seg_num, train_feats=train_feats)
        labels = train_labels_split(seg_num=seg_num, train_feats=train_labels)

        for it in range(config.total_iter):

            def closure():
                optimizer.zero_grad()
                for i in range(seg_num):
                    preds = net(feats[i])
                    # print(preds.shape)
                    loss = criterion(preds, labels[i])
                    if i == 9:
                        loss += config.lam * torch.norm(net.weight)
                    loss.backward()
                return loss

            optimizer.step(closure)
            
            if (it + 1) % config.val_freq == 0:
                if hasattr(config.data, 'val'):
                    acc, per = test(net, val_feats, val_labels)
                    logger.info('Iter [{}/{}] val acc: {} val per: {}'.format(it + 1, config.total_iter, acc, per))
                acc, per = test(net, test_feats, test_labels)
                logger.info('Iter [{}/{}] test acc: {}'.format(it + 1, config.total_iter, acc))
                if acc > best_acc:
                    best_acc = acc
                if per > best_per:
                    best_per = per
            
        logger.info("best_acc: " + str(best_acc))
        logger.info("best_per: " + str(best_per))
    
    # link.finalize()
