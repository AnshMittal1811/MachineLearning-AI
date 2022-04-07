import torch
import torch.nn as nn
import math
import copy
from torchmeta.modules import MetaModule
from torchmeta.modules.utils import get_subdict
from collections import OrderedDict
import numpy as np
from abc import ABC, abstractmethod


class AutoDiffNode(ABC, MetaModule):
    ''' THE ABSTRACT CLASS OF AN AUTODIFFNODE '''
    # This is what one of our nodes that is compatible with our "session"
    # mechanism, has to implement!

    def __init__(self):
        MetaModule.__init__(self)
        self.name = ' '
        self.copied_from = None
        self.order_idx = -1

        self.input_nodes = {}

    # Each node has to provide a name as it will appear in the tree
    # this is the child class name by default and can be overridden
    # (this name is used as a label, it is not used as a primary key
    # and hence does not need to be unique)
    def get_label(self):
        return self.__class__.__name__

    # Each node can provide a way it expects to share its parameters
    def share_params(self, obj_to_share_params_from):
        pass

    # Overrode lt='<' to heapify in the lexicographical topological ordering
    # /!\ Note that the __lt__ function below actually implements the function
    # greater than (gt). This is because, internally, in nx.sort_lexicographical_order,
    # the function heapq is used for sorting, and it uses by default "<" but we want
    # a descending sort, hence the override of "<" with ">".
    # TODO: could be cleaned y using appropriate keys
    def __lt__(self, other):
        return self.order_idx > other.order_idx

    # Each node will be deepcopied during the backward pass (so that we
    # can split the training in different modules), we override the
    # deepcopy operation here so that we can specify which parameters are
    # to be shared or not.
    def __deepcopy__(self, memodict={}):
        ''' we need to create a copy of the object not knowing its class (due to polymor.)'''
        # deep_copied_obj = type(self)()
        # would create a new instance of the same type as 'self'
        # but is "sensitive" to the interface of the constructor
        # hence, use the two following lines instead

        cls = self.__class__
        deep_copied_obj = cls.__new__(cls)

        # update the memo in place of the __init__
        memodict[id(self)] = deep_copied_obj
        for k, v in self.__dict__.items():
            setattr(deep_copied_obj, k, copy.deepcopy(v, memodict))

        # here we replace the params that need to be shared
        deep_copied_obj.share_params(self)  # we share the params from the deep copied obj

        # add some more meta_data which is useful for the compute_graph_fast
        deep_copied_obj.copied_from = self.name
        deep_copied_obj.input_nodes = {}

        # finally we return the deepcopied obj
        return deep_copied_obj

    # Each node has to provide a way to generate its derivative
    # We represent the backward graph of the node as an abstract
    # syntax tree formatted as a list in which:
    # a) Any input is encoded with an id: '0','1','2'
    # b) When prefixed with d, this corresponds to the derivative
    #    wrt this input: 'd0', 'd1'
    # c) other modules needed in the graph are instantiated
    @abstractmethod
    def gen_backward(self):
        pass

    # Each node has to provide how many input it uses/consumes
    @abstractmethod
    def get_num_inputs(self):
        pass

    # Each node reports its inputs
    def get_inputs(self):
        return self.input_nodes

    # Each node reports which "slot" in the input list an input occupies
    def get_input_pos(self, input):
        key_list = list(self.input_nodes.keys())
        val_list = list(self.input_nodes.values())
        return key_list[val_list.index(input)]
        # return self.input_nodes.index(input)

    # Each node can set an input at a given slot
    def set_input_at_pos(self, input, pos):
        self.input_nodes.update({pos: input})

    def add_input(self, input):
        pos = len(self.input_nodes)
        self.input_nodes.update({pos: input})

    # Each node has to provide how to compute its forward pass
    # inputs is the list of inputs: inputs[0] is the first input
    # inputs[1] is the second etc.
    @abstractmethod
    def _forward(self, *input_tensors):
        pass

    # Each node has to provide how it can be locally simplified
    # depending on its children
    # the method MUST return a string and an object.
    # - the string is either 'do_nothing' which means do not simplify
    #   or 'replace_by'.
    # - it is followed by an object like 'None' or one of the children
    #   or another instantiation, which is used in the case of 'replace_by'
    #   to further simplify the node. ('do_nothing' won't use the object at all)
    @abstractmethod
    def simplify(self, children):
        pass

    # We provide the default mechanism for the forward pass of a node.
    # Our mechanism wraps the forward pass of a nn.module
    # This is where the magic of building the forward graph happens.
    # To create a new AutoDiffModule, the user MUST NOT OVERRIDE
    # the forward() function BUT he should override the _forward() function.
    def forward(self, *in_values):
        session = in_values[0].session
        params = in_values[0].params

        out_data = self._forward([value.data for value in in_values],
                                 params)

        # self.input_nodes = [value.prod_node for value in in_values if value.prod_node is not None]
        self.input_nodes = {idx: value.prod_node for idx, value in enumerate(in_values)
                            if value.prod_node is not None}
        if (self.input_nodes) != self.get_num_inputs():
            "Invalid number of inputs"

        session.add_node_forward(self, self.input_nodes, self.get_label())
        return Value(data=out_data, session=session, prod_node=self, params=params)


class Value():
    ''' THE VALUE CLASS THAT IS FED TO NODES AND EXCHANGED IN BETWEEN NODES'''
    def __init__(self, data, session, prod_node=None, params=None):
        if isinstance(data, torch.Tensor):
            self.data = data
        else:
            print("Invalid data type")

        self.session = session
        self.params = params
        self.prod_node = prod_node


class Zero(AutoDiffNode):
    def __init__(self, shape_like):
        super().__init__()
        if isinstance(shape_like, torch.Tensor):
            self.value = torch.zeros_like(shape_like)
        else:
            self.value = torch.zeros_like(torch.tensor([shape_like]))

    def get_label(self):
        str = "\n"
        for d in range(self.value.dim()):
            str += f"{self.value.shape[d]}x"
        return super().get_label() + str[:-1]

    def gen_backward(self):
        zeros = Zero(self.value)
        return [zeros]

    def get_num_inputs(self):
        # this is a leaf
        return 0

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return self.value


class One(AutoDiffNode):
    def __init__(self, shape_like):
        super().__init__()
        if isinstance(shape_like, torch.Tensor):
            self.value = torch.ones_like(shape_like)
        else:
            self.value = torch.ones_like(torch.tensor([shape_like]))

    def get_label(self):
        str = "\n"
        for d in range(self.value.dim()):
            str += f"{self.value.shape[d]}x"
        return super().get_label() + str[:-1]

    def gen_backward(self):
        zeros = Zero(self.value)
        return [zeros]

    def get_num_inputs(self):
        # this is a leaf
        return 0

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return self.value


class Constant(AutoDiffNode):
    def __init__(self, value, name="Constant", id=None):
        super().__init__()
        if isinstance(value, torch.nn.Parameter):
            self.value = value
            self.value.requires_grad_(False)
        elif isinstance(value, torch.Tensor):
            if value.ndim == 0:
                value = value[None]
            self.value = nn.Parameter(value, requires_grad=False)
        else:
            self.value = nn.Parameter(torch.tensor([value]), requires_grad=False)

        self.name = name
        self.id = id

    def set_value(self, value, grad=False):
        dict_bak = self.value.__dict__.copy()
        if isinstance(value, torch.nn.Parameter):
            self.value = value
            self.value.requires_grad_(grad)
        else:
            self.value = nn.Parameter(value, requires_grad=grad)

        self.value.__dict__.update(dict_bak)

    def get_label(self):
        str = "\n"
        for d in range(self.value.dim()):
            str += f"{self.value.shape[d]}x"
        return super().get_label() + str[:-1]

    def gen_backward(self):
        zeros = Zero(self.value)
        return [zeros]

    def get_num_inputs(self):
        # this is a leaf
        return 0

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return self.value


class Input(AutoDiffNode):
    def __init__(self, input, name="Input", id=None):
        super().__init__()

        self.register_buffer('input', input)

        self.name = name
        self.id = id

    def set_value(self, input, grad=True):
        self.input = input

    def get_label(self):
        return super().get_label() + f"\n{self.input.shape[0]}x{self.input.shape[1]}"

    def gen_backward(self):
        ones = One(torch.ones_like(self.input))
        return [ones]

    def get_num_inputs(self):
        # this is a leaf
        return 0

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return self.input


def layer_factory(layer_type):
    layer_dict = \
        {
            'sine': (Sine(), sine_init),
            'fsine': (Sine(), first_layer_sine_init),
            'swish': (Swish(), swish_init),
            'requ': (ReQU(), requ_init)
        }
    return layer_dict[layer_type]


def sine_init(weight, bias, w0=30):
    num_input = weight.size(-1)
    weight.uniform_(-np.sqrt(6/num_input)/w0, np.sqrt(6/num_input)/w0)
    if bias is not None:
        num_input = weight.size(-1)
        std = 1/math.sqrt(num_input)
        bias.uniform_(-std, std)
    return weight, bias


def first_layer_sine_init(weight, bias):
    num_input = weight.size(-1)
    weight.uniform_(-1/num_input, 1/num_input)
    if bias is not None:
        num_input = weight.size(-1)
        std = 1/math.sqrt(num_input)
        bias.uniform_(-std, std)
    return weight, bias


def requ_init(weight, bias):
    nn.init.kaiming_normal_(weight, a=0.0, nonlinearity='relu', mode='fan_out')
    if bias is not None:
        nn.init.uniform_(bias, -.5, .5)
    return weight, bias


def swish_init(weight, bias):
    nn.init.kaiming_normal_(weight, a=0.0, nonlinearity='relu', mode='fan_in')
    if bias is not None:
        num_input = weight.size(-1)
        std = 1/math.sqrt(num_input)
        bias.uniform_(-std, std)
    return weight, bias


class Linear(AutoDiffNode):
    def __init__(self, in_features=None, out_features=None, batch_features=None, weight=None, bias=None,
                 nl='sine', w0=30, name='Linear'):
        super().__init__()

        self.name = name
        if in_features is not None and out_features is not None:

            if batch_features is None:
                self.w = torch.zeros(out_features, in_features)
                self.b = torch.zeros(out_features)
            else:
                self.w = torch.zeros(batch_features, 1, out_features, in_features)
                self.b = torch.zeros(out_features)

            self.w, self.b = layer_factory(nl)[1](self.w, self.b)  # To init the weights

            self.w = nn.Parameter(self.w)
            self.b = nn.Parameter(self.b)

        elif weight is not None:
            self.w = weight
            self.b = bias

    def get_label(self):
        return super().get_label() + f"\n{self.w.shape[0]}x{self.w.shape[1]}"

    def share_params(self, copied_from_obj):
        copied_from_obj.w.id = copied_from_obj.name
        self.w = copied_from_obj.w
        self.b = copied_from_obj.b

    def gen_backward(self):
        # ( Linear(u) )' = (W*u + b)' = W*u' = Linear(u') (with b=0)
        self.w.id = self.name
        linear = Linear(weight=self.w, bias=None)
        return [linear, ['d0']]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        if isinstance(children[0], Zero):
            return 'replace_by', Zero(torch.zeros((self.w.shape[0], 1)))
        else:
            return 'do_nothing', None

    def _forward(self, x, params=None):
        if params is None:
            p = OrderedDict(self.named_parameters())
        else:
            p = get_subdict(params, self.w.id)
            if len(p) == 0:
                p = OrderedDict(self.named_parameters())

        output = torch.matmul(x[0], p['w'].transpose(-1, -2))
        if self.b is not None:
            output += p['b'].unsqueeze(0).expand_as(output)

        return output


class HadamardAdd(AutoDiffNode):
    def __init__(self, name='HadamardAdd'):
        super().__init__()

    def gen_backward(self):
        # (u+v)' = u' + v'
        add = HadamardAdd()
        return [add, ['d0', 'd1']]

    def get_num_inputs(self):
        # addition takes two tensors as inputs
        return 2

    def simplify(self, children):
        if isinstance(children[0], Zero) and isinstance(children[1], Zero):  # 0 + 0 = 0
            return 'replace_by', children[0]
        else:
            return 'do_nothing', None

    def _forward(self, x, params=None):
        return x[0]+x[1]


class Concatenate(AutoDiffNode):
    def __init__(self, num_inputs=2, name='Concatenate'):
        super().__init__()
        self.num_inputs = num_inputs

    def gen_backward(self):
        # cat(u, v)' = cat(u', v')
        cat = Concatenate(num_inputs=self.num_inputs)
        derivatives = ['d' + str(x) for x in range(self.num_inputs)]
        return [cat, derivatives]

    def get_num_inputs(self):
        # product takes two tensors as inputs
        return self.num_inputs

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):

        # annoying, but we need to manually broadcast the inputs
        # before we concatenate them...
        shape = None
        for val in x:
            if val.shape[0] > 1 or val.shape[1] > 1:
                shape = val.shape[:-1]
                break

        for idx in range(len(x)):
            if shape is not None and x[idx].shape[0] == 1:
                x[idx] = x[idx].expand((*shape, -1))

        out = torch.cat(x, dim=-1)
        return out


class HadamardProd(AutoDiffNode):
    def __init__(self, name='HadamardProd'):
        super().__init__()

    def gen_backward(self):
        # (u*v)' = u'*v + u*v'
        add = HadamardAdd()
        prod1 = HadamardProd()
        prod2 = HadamardProd()
        return [add, [[prod1, ['d0', '1']], [prod2, ['0', 'd1']]]]

    def get_num_inputs(self):
        # product takes two tensors as inputs
        return 2

    def simplify(self, children):
        if isinstance(children[0], One) and isinstance(children[1], One):  # 1 * 1 = 1
            return 'replace_by', children[0]
        elif isinstance(children[0], Zero) and isinstance(children[1], Zero):  # 0 * 0 = 0
            return 'replace_by', children[0]
        else:
            return 'do_nothing', None

        return 'do_nothing', None

    def _forward(self, x, params=None):
        return x[0]*x[1]


class ReLU(AutoDiffNode):
    def __init__(self, name='ReLU'):
        super().__init__()

    def gen_backward(self):
        # TODO: Not impelemented
        return [0]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return torch.relu(x[0])


class ReQU(AutoDiffNode):
    def __init__(self, name='ReQU'):
        super().__init__()

    def gen_backward(self):
        # ( requ(u) )' = relu(u) * u'
        prod = HadamardProd()
        relu = ReLU()
        return [prod, [[relu, ['0']], 'd0']]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return .5*torch.relu(x[0])**2


class Sine(AutoDiffNode):
    def __init__(self, w0=30., name='Sine'):
        super().__init__()
        self.w0 = w0

    def gen_backward(self):
        prod = HadamardProd()
        dsine = DSine(w0=self.w0)
        return [prod, [[dsine, ['0']], 'd0']]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return torch.sin(self.w0*x[0])


class DSine(AutoDiffNode):
    def __init__(self, w0=1., name='DSine'):
        super().__init__()
        self.w0 = w0

    def gen_backward(self):
        # TODO: not implemented
        return [0]

    def get_num_inputs(self):
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return torch.cos(self.w0 * x[0]) * self.w0


class Cosine(AutoDiffNode):
    def __init__(self, w0=30., name='Cosine'):
        self.w0 = w0
        super().__init__()

    def gen_backward(self):
        # ( cos(u) )' = -sin(u) * u'
        prod = HadamardProd()
        dcosine = DCosine(w0=self.w0)
        return [prod, [[dcosine, ['0']], 'd0']]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return torch.cos(self.w0 * x[0])


class DCosine(AutoDiffNode):
    def __init__(self, w0=1., name='DCosine'):
        super().__init__()
        self.w0 = w0

    def gen_backward(self):
        # TODO: Not implemented
        return [0]

    def get_num_inputs(self):
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return -torch.sin(self.w0 * x[0]) * self.w0


class Exp(AutoDiffNode):
    def __init__(self, name='Exp'):
        super().__init__()

    def gen_backward(self):
        prod = HadamardProd()
        exp = Exp()
        return [prod, [[exp, ['0']], 'd0']]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return torch.exp(x[0])


class Sigmoid(AutoDiffNode):
    def __init__(self, name='Sigmoid'):
        super().__init__()

    def gen_backward(self):
        prod0 = HadamardProd()
        prod1 = HadamardProd()
        prod2 = HadamardProd()

        add0 = HadamardAdd()

        sigmoid0 = Sigmoid()
        sigmoid1 = Sigmoid()

        mSigmoid = [prod0, [[sigmoid0, ['0']], Constant(-1.)]]
        one_minus_sigmoid = [add0, [mSigmoid, Constant(1.)]]
        sigmoid = [sigmoid1, ['0']]

        return [prod2, [sigmoid, [prod1, [one_minus_sigmoid, 'd0']]]]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return torch.sigmoid(x[0])


class DSwish(AutoDiffNode):
    def __init__(self, name='DSwish'):
        super().__init__()

    def gen_backward(self):
        # TODO: not implemented
        return [0]

    def get_num_inputs(self):
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        s = torch.sigmoid(x[0])
        return s + x[0]*(s - s**2)


class Swish(AutoDiffNode):
    def __init__(self, name='Swish'):
        super().__init__()

    def gen_backward(self):
        prod = HadamardProd()
        dswish = DSwish()
        return [prod, [[dswish, ['0']], 'd0']]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        return x[0]*torch.sigmoid(x[0])


class PositionalEncoding(AutoDiffNode):
    def __init__(self, num_encoding_functions=4, include_input=True, normalize=True,
                 log_sampling=True, name='PositionalEncoding'):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.normalize = normalize
        self.include_input = include_input
        self.log_sampling = log_sampling

        self.frequency_bands = None
        if self.log_sampling:
            self.frequency_bands = 2.0 ** torch.linspace(
                0.0,
                self.num_encoding_functions - 1,
                self.num_encoding_functions)
        else:
            self.frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (self.num_encoding_functions - 1),
                self.num_encoding_functions)

    def gen_backward(self):
        dpositionalencoding = DPositionalEncoding(self.frequency_bands, self.include_input, self.normalize)
        repeat = Repeat(times=len(self.frequency_bands) * 2)
        if self.include_input:
            return [Concatenate(num_inputs=2), ['d0', [HadamardProd(), [[dpositionalencoding, ['0']], [repeat, ['d0']]]]]]
        else:
            return [HadamardProd(), [[dpositionalencoding, ['0']], [repeat, ['d0']]]]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        encoding = [x[0]] if self.include_input else []
        for freq in self.frequency_bands:
            for func in [torch.sin, torch.cos]:
                if self.normalize:
                    encoding.append(1/freq*func(x[0] * freq))
                else:
                    encoding.append(func(x[0] * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


class Repeat(AutoDiffNode):
    def __init__(self, times, dim=-1, name='Swish'):
        super().__init__()
        self.times = times
        self.dim = dim

    def gen_backward(self):
        # TODO: not needed for now
        return [0]

    def get_num_inputs(self):
        # this takes a single tensor as input
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        rep = x[0].ndim * [1, ]
        rep[self.dim] = self.times
        return x[0].repeat(rep)


class DPositionalEncoding(AutoDiffNode):
    def __init__(self, frequency_bands, include_input, normalize, name='DPositionalEncoding'):
        super().__init__()
        self.frequency_bands = frequency_bands
        self.include_input = include_input
        self.normalize = normalize

    def gen_backward(self):
        # TODO: not important for now
        return [0]

    def get_num_inputs(self):
        return 1

    def simplify(self, children):
        return 'do_nothing', None

    def _forward(self, x, params=None):
        back = []
        def msin(x): return -torch.sin(x)
        for freq in self.frequency_bands:
            for func in [torch.cos, msin]:
                if self.normalize:
                    back.append(func(freq * x[0]))
                else:
                    back.append(func(freq * x[0]) * freq)
        return torch.cat(back, dim=-1)



