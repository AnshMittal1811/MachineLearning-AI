import sys
from collections import OrderedDict


def define_G(opt):
    if opt.archG in ['stylegan2', 'stylegan3']:
        sys.path.append('./models/networks/stylegan3')
        import dnnlib
        import legacy
        network_pkl = opt.pretrained_G
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as fp:
            netG = legacy.load_network_pkl(fp)['G_ema']
        return netG
    else:
        raise KeyError(f"Unknown option archG: {opt.archG}")


def set_requires_grad(params, flag):
    """
    For more efficient optimization, turn on and off
    recording of gradients for |params|.
    """
    for p in params:
        p.requires_grad = flag


def get_modules(archG, netG, mode='conv') -> OrderedDict:
    if archG == 'stylegan3':
        return get_stylegan3_modules(netG, mode=mode)
    elif archG == 'stylegan2':
        return get_stylegan2_modules(netG, mode=mode)
    else:
        raise ValueError(f'Unknown G architecture: {archG}')


def get_stylegan3_modules(netG, mode='conv') -> OrderedDict:
    """Get list of module names of interest. Mode: [conv|affine|all]."""
    module_dict = OrderedDict()
    for name, mod in netG.named_modules():
        if '.L' in name:
            if mode == 'conv' and '.affine' not in name:
                module_dict[name] = mod
            elif mode == 'affine' and 'affine' in name:
                module_dict[name] = mod
            elif mode == 'all':
                module_dict[name] = mod
    return module_dict


def get_stylegan2_modules(netG, mode='conv') -> OrderedDict:
    """Get list of module names of interest. Mode: [conv|affine|all]."""
    module_dict = OrderedDict()
    for name, mod in netG.named_modules():
        if '.b' in name:
            if mode == 'all':
                module_dict[name] = mod
            elif '.conv' in name:
                if mode == 'conv' and 'affine' not in name:
                    module_dict[name] = mod
                elif mode == 'affine' and 'affine' in name:
                    module_dict[name] = mod
    return module_dict


def get_module_resolution(archG, module_dict: OrderedDict):
    if archG == 'stylegan3':
        return get_stylegan3_module_res(module_dict)
    elif archG == 'stylegan2':
        return get_stylegan2_module_res(module_dict)
    else:
        raise ValueError(f'Unknown G architecture: {archG}')


def get_stylegan3_module_res(module_dict):
    assert type(module_dict) == OrderedDict, f"Expect module_dict to be OrderedDict, but get type {type(module_dict)}"
    module_list = list(module_dict.values())
    in_sizes = [mod.in_size[0] if hasattr(mod, 'in_size') else None for mod in module_list]
    out_sizes = [mod.out_size[0] if hasattr(mod, 'out_size') else None for mod in module_list]
    return in_sizes, out_sizes


def get_stylegan2_module_res(module_dict):
    assert type(module_dict) == OrderedDict, f"Expect module_dict to be OrderedDict, but get type {type(module_dict)}"
    module_list = list(module_dict.values())
    in_sizes = [mod.resolution // mod.up if hasattr(mod, 'resolution') else None for mod in module_list]
    out_sizes = [mod.resolution if hasattr(mod, 'resolution') else None for mod in module_list]
    return in_sizes, out_sizes
