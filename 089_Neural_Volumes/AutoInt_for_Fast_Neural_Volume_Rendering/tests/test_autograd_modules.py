import unittest
import os
import sys
from torchmeta.modules import MetaModule
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autograd.session import Session
from autograd.autograd_modules import *
from collections import OrderedDict
from abc import ABC
from diff_operators import jacobian
import torch


class TestSession(MetaModule):
    def __init__(self, fn):
        super().__init__()
        self.input_name = ['values']
        self.session = Session()
        self.fn = fn

        # run forward pass to bootstrap session
        self.trace_graph()
        self.session = self.session.cuda()
        self.backward_session = self.session.get_backward_graph().cuda()

    def trace_graph(self):
        # we need to explicitly step the session through the graph
        # to trace out each node
        x = Value(torch.ones(1, 1), self.session)

        out = Input(torch.Tensor(1, 1), id=self.input_name[0])(x)
        self.fn()(out)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.session.named_parameters())
        return self.session({'values': x, 'params': params})

    def backward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.backward_session.named_parameters())
        return self.backward_session({'values': x, 'params': params})


class TestConcatSession(MetaModule):
    def __init__(self):
        super().__init__()
        self.input_name = ['values0', 'values1', 'values2', 'values3']
        self.session = Session()

        # run forward pass to bootstrap session
        self.trace_graph()
        self.session = self.session.cuda()
        self.backward_session = self.session.get_backward_graph().cuda()

    def trace_graph(self):
        # we need to explicitly step the session through the graph
        # to trace out each node
        x1 = Value(torch.ones(1, 1), self.session)
        x2 = Value(torch.ones(1, 1), self.session)
        x3 = Value(torch.ones(1, 1), self.session)

        in1 = Input(torch.Tensor(1, 1), id=self.input_name[0])(x1)
        in2 = Constant(torch.Tensor(1, 1), id=self.input_name[1])(x2)

        out = HadamardAdd()(in1, in2)

        in3 = Constant(torch.Tensor(1, 1), id=self.input_name[2])(x3)
        out = HadamardProd()(in3, out)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.session.named_parameters())
        return self.session({'values0': x[0], 'values1': x[1], 'values2': x[2], 'values3': x[3], 'params': params})

    def backward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.backward_session.named_parameters())
        return self.backward_session({'values0': x[0], 'values1': x[1], 'values2': x[2], 'values3': x[3], 'params': params})


class TestFunction(ABC):

    def test_backward(self):
        # run our backward graph
        x = torch.randn(1, 1).cuda()
        our_grad = self.test_sess.backward(x).squeeze()

        # run forward graph and calc grad using pytorch
        x = torch.nn.Parameter(x)
        out = self.test_sess(x)
        pytorch_grad = jacobian(out[None, None, :], x)[0].squeeze()
        print(our_grad)
        print(pytorch_grad)
        self.assertTrue(torch.allclose(our_grad, pytorch_grad))


class TestSigmoid(TestFunction, unittest.TestCase):

    def setUp(self):
        self.test_sess = TestSession(Sigmoid)


class TestExp(TestFunction, unittest.TestCase):

    def setUp(self):
        self.test_sess = TestSession(Exp)


class TestSwish(TestFunction, unittest.TestCase):

    def setUp(self):
        self.test_sess = TestSession(Swish)


class TestPositionalEncoding(TestFunction, unittest.TestCase):

    def setUp(self):
        self.test_sess = TestSession(PositionalEncoding)


class TestSine(TestFunction, unittest.TestCase):

    def setUp(self):
        self.test_sess = TestSession(Sine)


class TestCosine(TestFunction, unittest.TestCase):

    def setUp(self):
        self.test_sess = TestSession(Cosine)


class TestConcat(TestFunction, unittest.TestCase):

    def setUp(self):
        self.test_sess = TestConcatSession()

    def test_backward(self):
        # run our backward graph
        x = [torch.randn(1, 1).cuda() for i in range(4)]
        our_grad = self.test_sess.backward(x).squeeze()

        # run forward graph and calc grad using pytorch
        x[0] = torch.nn.Parameter(x[0])
        out = self.test_sess(x)
        pytorch_grad = jacobian(out[None, None, :], x[0])[0].squeeze()
        self.assertTrue(torch.allclose(our_grad, pytorch_grad))


if __name__ == '__main__':
    unittest.main()
