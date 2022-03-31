import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autograd.session import Session
import autograd.autograd_modules as modules
import matplotlib.pyplot as plt
from torchmeta.modules import MetaModule


class SIREN(MetaModule):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.trace_graph()

    def trace_graph(self):
        x1 = modules.Value(torch.ones(1, 1), self.session)
        x2 = modules.Value(torch.ones(1, 1), self.session)

        in1 = modules.Input(torch.Tensor(1, 1), id='x_coords')(x1)
        in2 = modules.Constant(torch.Tensor(1, 1), id='y_coords')(x2)
        in3 = modules.Concatenate()(in1, in2)

        self.net = []
        self.net.append(modules.Linear(2, 128))
        self.net.append(modules.Sine())
        self.net.append(modules.Linear(128, 128))
        self.net.append(modules.Sine())
        self.net.append(modules.Linear(128, 128))
        self.net.append(modules.Sine())
        self.net.append(modules.Linear(128, 128))
        self.net.append(modules.Sine())
        self.net.append(modules.Linear(128, 1))
        self.net = nn.Sequential(*self.net)
        self.net(in3)

    def forward(self, model_input):
        input_dict = {key: input.clone().detach().requires_grad_(True)
                      for key, input in model_input.items()}

        out = self.session.compute_graph_fast(input_dict)
        return out, input_dict


error_fn = torch.nn.MSELoss()

GRAD_GRAD_EVAL = True 
EVAL_PYTORCH = True

# this creates a small siren
print("\t. Instantiate a SIREN")
torch.manual_seed(0)
session = Session()
net = SIREN(session)

# create simple input with manual seed
print("\t. Creating input")
x = torch.ones(1, 1)
y = torch.ones(1, 1)
x.requires_grad_(True)

sess_input = {'x_coords': x,
              'y_coords': y}

print("\t. (1) Forward evaluation SIREN")
forward_siren_evaluation, model_input = net(sess_input)


print("\t. (2) Forward evaluation of a gradientSIREN")
back_session = session.get_backward_graph()
output = back_session(sess_input)
forward_gradsiren_evaluation = output

print("\t. (3) Calculate PyTorch gradient")
backward_siren_evaluation = torch.autograd.grad(forward_siren_evaluation, model_input['x_coords'],
                                                torch.ones_like(forward_siren_evaluation),
                                                create_graph=False)[0]

print("\t. Discrepancy between (2) and (3)")
print(f"\t\t. Error={error_fn(forward_gradsiren_evaluation.squeeze(),backward_siren_evaluation.squeeze())}")

print("\t. Making graph visualizations of session_siren & session_gradsiren")
print("\t\t. Plotting forward graph in session_siren")
session.draw('SIREN')

print("\t\t. Plotting backward graph in session_siren")
back_session.draw('grad_SIREN')

if GRAD_GRAD_EVAL:
    print("\t. Computing & making graph visualizations of a grad_grad_siren")
    back_back_session = back_session.get_backward_graph()
    back_back_session.simplify()
    back_back_session.draw('grad_grad_siren')

plt.show()
