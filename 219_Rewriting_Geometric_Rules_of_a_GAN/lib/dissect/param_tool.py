from collections import OrderedDict


def get_params(net, update_layers, exclude_children=False):
    update_params = OrderedDict()
    for name, module in net.named_modules():
        if name in update_layers:
            params = get_params_from_module(module, prefix=name, exclude_children=exclude_children)
            update_params.update(params)
    return update_params


def get_params_from_module(module, prefix='', exclude_children=False):
    """
    Return all parameters and their names in the module.
    Includes options to add prefix to all the names (useful when the module is not root).
    If exclude_children is specified, only returns parameters that don't belong to the children modules.
    """
    if exclude_children:
        children_param = []
        for cname, child in module.named_children():
            pnames, _ = zip(*child.named_parameters())
            pnames = [f'{cname}.{pname}' for pname in pnames]
            children_param.extend(pnames)

    param_dict = OrderedDict()
    for name, param in module.named_parameters():
        if exclude_children and name in children_param:
            continue
        pname = f'{prefix}.{name}'
        param_dict[pname] = param

    return param_dict


def get_module_by_param(net, param, keyword):
    for module in net.modules():
        if getattr(module, keyword, None) is param:
            return module


def merge_state_dict(*state_dicts):
    merged = OrderedDict()
    for d in state_dicts:
        intersect = merged.keys() & d.keys()
        if len(intersect) > 0:
            raise ValueError(f"expect no keys in common in the state_dicts, but found {intersect} in common.")
        merged.update(d)
    return merged
