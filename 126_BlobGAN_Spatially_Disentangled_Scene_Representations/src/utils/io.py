import os
import re
from glob import glob
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from inputimeout import inputimeout, TimeoutOccurred
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torchvision.utils import make_grid

from .distributed import is_rank_zero

BED_CONF_COLORS = torch.tensor([[0.9803921580314636, 0.9450980424880981, 0.9176470637321472],
          [0.8980392217636108, 0.5254902243614197, 0.0235294122248888],
          [0.364705890417099, 0.4117647111415863, 0.6941176652908325],
          [0.32156863808631897, 0.7372549176216125, 0.6392157077789307],
          [0.6000000238418579, 0.7882353067398071, 0.2705882489681244],
          [0.18431372940540314, 0.5411764979362488, 0.7686274647712708],
          [0.6470588445663452, 0.6666666865348816, 0.6000000238418579],
          [0.8549019694328308, 0.6470588445663452, 0.10588235408067703],
          [0.4627451002597809, 0.30588236451148987, 0.6235294342041016],
          [0.800000011920929, 0.3803921639919281, 0.6901960968971252],
          [0.929411768913269, 0.3921568691730499, 0.3529411852359772],
          [0.1411764770746231, 0.4745098054409027, 0.42352941632270813],
          [0.4000000059604645, 0.772549033164978, 0.800000011920929],
          [0.9647058844566345, 0.8117647171020508, 0.4431372582912445],
          [0.9725490212440491, 0.6117647290229797, 0.45490196347236633],
          [0.8627451062202454, 0.6901960968971252, 0.9490196108818054],
          [0.529411792755127, 0.772549033164978, 0.37254902720451355],
          [0.6196078658103943, 0.7254902124404907, 0.9529411792755127],
          [0.9960784316062927, 0.5333333611488342, 0.6941176652908325],
          [0.7882353067398071, 0.8588235378265381, 0.45490196347236633],
          [0.545098066329956, 0.8784313797950745, 0.6431372761726379],
          [0.7058823704719543, 0.5921568870544434, 0.9058823585510254],
          [0.7019608020782471, 0.7019608020782471, 0.7019608020782471],
          [0.5215686559677124, 0.3607843220233917, 0.4588235318660736],
          [0.8509804010391235, 0.686274528503418, 0.41960784792900085],
          [0.686274528503418, 0.3921568691730499, 0.3450980484485626],
          [0.45098039507865906, 0.43529412150382996, 0.2980392277240753],
          [0.32156863808631897, 0.4156862795352936, 0.5137255191802979],
          [0.3843137323856354, 0.32549020648002625, 0.46666666865348816],
          [0.40784314274787903, 0.5215686559677124, 0.3607843220233917],
          [0.6117647290229797, 0.6117647290229797, 0.3686274588108063],
          [0.6274510025978088, 0.3803921639919281, 0.46666666865348816],
          [0.5490196347236633, 0.47058823704719543, 0.364705890417099],
          [0.48627451062202454, 0.48627451062202454, 0.48627451062202454]])

KLD_COLORS = torch.tensor([[0.9803921580314636, 0.9450980424880981, 0.9176470637321472],
              [0.5490196347236633, 0.47058823704719543, 0.364705890417099],
              [0.6274510025978088, 0.3803921639919281, 0.46666666865348816],
              [0.6117647290229797, 0.6117647290229797, 0.3686274588108063],
              [0.40784314274787903, 0.5215686559677124, 0.3607843220233917],
              [0.686274528503418, 0.3921568691730499, 0.3450980484485626],
              [0.7019608020782471, 0.7019608020782471, 0.7019608020782471],
              [0.45098039507865906, 0.43529412150382996, 0.2980392277240753],
              [0.8509804010391235, 0.686274528503418, 0.41960784792900085],
              [0.3843137323856354, 0.32549020648002625, 0.46666666865348816],
              [0.5215686559677124, 0.3607843220233917, 0.4588235318660736],
              [0.32156863808631897, 0.4156862795352936, 0.5137255191802979],
              [0.7058823704719543, 0.5921568870544434, 0.9058823585510254],
              [0.545098066329956, 0.8784313797950745, 0.6431372761726379],
              [0.7882353067398071, 0.8588235378265381, 0.45490196347236633],
              [0.9960784316062927, 0.5333333611488342, 0.6941176652908325],
              [0.6196078658103943, 0.7254902124404907, 0.9529411792755127],
              [0.529411792755127, 0.772549033164978, 0.37254902720451355],
              [0.8627451062202454, 0.6901960968971252, 0.9490196108818054],
              [0.9725490212440491, 0.6117647290229797, 0.45490196347236633],
              [0.9647058844566345, 0.8117647171020508, 0.4431372582912445],
              [0.4000000059604645, 0.772549033164978, 0.800000011920929],
              [0.6470588445663452, 0.6666666865348816, 0.6000000238418579],
              [0.929411768913269, 0.3921568691730499, 0.3529411852359772],
              [0.4627451002597809, 0.30588236451148987, 0.6235294342041016],
              [0.18431372940540314, 0.5411764979362488, 0.7686274647712708],
              [0.8549019694328308, 0.6470588445663452, 0.10588235408067703],
              [0.1411764770746231, 0.4745098054409027, 0.42352941632270813],
              [0.800000011920929, 0.3803921639919281, 0.6901960968971252],
              [0.6000000238418579, 0.7882353067398071, 0.2705882489681244],
              [0.32156863808631897, 0.7372549176216125, 0.6392157077789307],
              [0.364705890417099, 0.4117647111415863, 0.6941176652908325],
              [0.8980392217636108, 0.5254902243614197, 0.0235294122248888],
              [1.0, 1.0, 1.0],
              [0.48627451062202454, 0.48627451062202454, 0.48627451062202454],
              [0.5490196347236633, 0.47058823704719543, 0.364705890417099],
              [0.6274510025978088, 0.3803921639919281, 0.46666666865348816],
              [0.6117647290229797, 0.6117647290229797, 0.3686274588108063],
              [0.40784314274787903, 0.5215686559677124, 0.3607843220233917],
              [0.3843137323856354, 0.32549020648002625, 0.46666666865348816],
              [0.32156863808631897, 0.4156862795352936, 0.5137255191802979],
              [0.45098039507865906, 0.43529412150382996, 0.2980392277240753],
              [0.686274528503418, 0.3921568691730499, 0.3450980484485626],
              [0.8509804010391235, 0.686274528503418, 0.41960784792900085],
              [0.5215686559677124, 0.3607843220233917, 0.4588235318660736],
              [0.7019608020782471, 0.7019608020782471, 0.7019608020782471],
              [0.7058823704719543, 0.5921568870544434, 0.9058823585510254],
              [0.545098066329956, 0.8784313797950745, 0.6431372761726379],
              [0.7882353067398071, 0.8588235378265381, 0.45490196347236633],
              [0.9960784316062927, 0.5333333611488342, 0.6941176652908325],
              [0.6196078658103943, 0.7254902124404907, 0.9529411792755127],
              [0.529411792755127, 0.772549033164978, 0.37254902720451355],
              [0.8627451062202454, 0.6901960968971252, 0.9490196108818054],
              [0.9725490212440491, 0.6117647290229797, 0.45490196347236633],
              [0.9647058844566345, 0.8117647171020508, 0.4431372582912445],
              [0.4000000059604645, 0.772549033164978, 0.800000011920929],
              [0.6470588445663452, 0.6666666865348816, 0.6000000238418579],
              [0.929411768913269, 0.3921568691730499, 0.3529411852359772],
              [0.4627451002597809, 0.30588236451148987, 0.6235294342041016],
              [0.18431372940540314, 0.5411764979362488, 0.7686274647712708],
              [0.8549019694328308, 0.6470588445663452, 0.10588235408067703],
              [0.1411764770746231, 0.4745098054409027, 0.42352941632270813],
              [0.800000011920929, 0.3803921639919281, 0.6901960968971252],
              [0.6000000238418579, 0.7882353067398071, 0.2705882489681244],
              [0.32156863808631897, 0.7372549176216125, 0.6392157077789307],
              [0.364705890417099, 0.4117647111415863, 0.6941176652908325],
              [0.8980392217636108, 0.5254902243614197, 0.0235294122248888],
              [1.0, 1.0, 1.0]])


def load_pretrained_weights(name, pretrained, model):
    if pretrained:
        state_dict_prefix = None
        load_fn = 'load_state_dict'
        if type(pretrained) is DictConfig:
            key = pretrained.get('key', None)
            state_dict_prefix = pretrained.get('state_dict_prefix', None)
            load_fn = pretrained.get('load_fn', 'load_state_dict')
            if 'path' in pretrained:
                pretrained = pretrained['path']
            else:
                pretrained = get_checkpoint_path(**pretrained)
        else:
            key = pretrained
        state = torch.load(pretrained, map_location='cpu')
        if key is not None:
            state = state[key]
        if state_dict_prefix is not None:
            state_dict_prefix += '.'
            state = {k.replace(state_dict_prefix, ''): v for k, v in state.items() if k.startswith(state_dict_prefix)}
        getattr(model, load_fn)(state)
        if is_rank_zero():
            print(f'Loaded state from {pretrained} for {name}')
    return model


def get_checkpoint_path(log_dir: str, project: str, id: str, step: Optional[int] = None, epoch: Optional[int] = None,
                        last: bool = False, best: bool = False, **kwargs) -> str:
    base_path = Path(log_dir).absolute()
    # Used to store checkpoints separately under {project}/{id}, now merged under run folder with timestamp
    # Helps to keep logs and checkpoints in same place, and can abstract over this in loading logic
    old_format_checkpoints = glob(str(base_path / project / id / 'checkpoints' / '*'))
    new_format_checkpoints = glob(str(base_path / 'wandb' / f'*{id}*' / 'files' / 'checkpoints' / '*'))
    checkpoints = sorted(old_format_checkpoints + new_format_checkpoints, key=os.path.getmtime,
                         reverse=True)
    if last:
        try:
            fn = next(filter(lambda fn: fn.endswith('last.ckpt'), checkpoints))
        except StopIteration:
            if is_rank_zero():
                print('No checkpoint ending in `last.pt` found, perhaps since checkpointing was done every N steps, '
                      'without a monitoring metric. Using most recent available checkpoint instead.')
            fn = checkpoints[0]
    elif best:
        # Need to search across all runs to find best model with lowest score
        # Since stats are tracked separately for each run
        all_last_models = [fn for fn in checkpoints if fn.endswith('last.ckpt')]
        all_checkpoints = [torch.load(fn, map_location='cpu')['callbacks'][ModelCheckpoint] for fn in all_last_models]
        fn = sorted(all_checkpoints, key=lambda c: c['best_model_score'])[0]['best_model_path']
    else:
        filters = [lambda fn: 'compressed' not in fn]
        if epoch is not None:
            filters.append(lambda fn: f'epoch={epoch}' in fn)
        if step is not None:
            filters.append(lambda fn: f'step={step}' in fn)
        try:
            fn = next(filter(lambda fn: all(f(fn) for f in filters), checkpoints))
        except StopIteration:
            raise ValueError(f'No checkpoint found matching criteria. Please check parameters and try again.\n'
                             f'Valid checkpoints are: {", ".join(map(os.path.basename, checkpoints))}')
    epoch, = re.search('epoch=(\d+)', fn).groups()
    step, = re.search('step=(\d+)', fn).groups()
    if is_rank_zero():
        print(f'Resuming from epoch {epoch}, step {step}')
    return fn


def resolve_resume_id(log_dir: str, id: str, **kwargs) -> str:
    if id == "latest_run":
        base_path = Path(log_dir).absolute()
        latest_id = os.readlink(next((base_path / "wandb").glob('*latest-run*')))[-8:]
        if is_rank_zero():
            print(f'Resuming from latest run: {latest_id}')
        return latest_id
    return id


def show_imgs(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show_img_grid(imgs: Tensor, *args, **kwargs):
    show_imgs(make_grid(imgs.float().cpu(), *args, **kwargs))


def yes_or_no(q: str, default: Optional[bool] = None, timeout: Optional[int] = None) -> bool:
    assert not (timeout is not None and default is None), 'If using timeout, must set default value.'
    q = f"{q} [{'Y' if default is True else 'y'}/{'N' if default is False else 'n'}]: "
    if timeout is not None:
        def input_fn(prompt):
            try:
                return inputimeout(prompt=prompt, timeout=timeout)
            except TimeoutOccurred:
                print(f'Input timed out. Using default value of {default}.')
                return ''
    else:
        input_fn = input
    a = input_fn(q).lower().strip()
    print("")
    valid = ['y', 'n', 'yes', 'no']
    if default is not None:
        valid.append('')
    while a not in valid:
        print("Input yes or no")
        a = input_fn(q).lower().strip()
        print("")
    if a == "":
        return default
    elif a[0] == "y":
        return True
    else:
        return False
