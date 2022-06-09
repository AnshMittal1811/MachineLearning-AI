import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.distributions import Normal, Categorical
import math

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data.view(1, -1), gain=gain)
    return module

init_r_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    nn.init.orthogonal_,
    0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActorCriticNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], num_contact=0):
        super(ActorCriticNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        self.hidden_layer_v = [1024, 1024]
        if (len(hidden_layer) > 0):
            p_fc = init_r_(nn.Linear(num_inputs, self.hidden_layer[0]))
            v_fc = init_r_(nn.Linear(num_inputs, self.hidden_layer_v[0]))
            self.p_fcs.append(p_fc)
            self.v_fcs.append(v_fc)
            for i in range(len(self.hidden_layer)-1):
                p_fc = init_r_(nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1]))
                v_fc = init_r_(nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1]))
                self.p_fcs.append(p_fc)
                self.v_fcs.append(v_fc)
            self.mu = init_r_(nn.Linear(self.hidden_layer[-1], num_outputs))
        else:
            #p_fc = init_r_(nn.Linear(num_inputs, num_outputs))
            #self.p_fcs.append(p_fc)
            self.mu = init_r_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.v = init_r_(nn.Linear(self.hidden_layer_v[-1],1))
        self.noise = 0
        #self.train()


    def forward(self, inputs):
        # actor
        if len(self.hidden_layer) > 0:
            x = F.relu(self.p_fcs[0](inputs))
            for i in range(len(self.hidden_layer)-1):
                x = F.relu(self.p_fcs[i+1](x))
            mu = torch.tanh(self.mu(x))
        else:
            mu = torch.tanh(self.mu(inputs))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

    def get_log_stds(self, actions):
        return Variable(torch.Tensor(self.noise)*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(actions).to(device)
        #return self.log_std.unsqueeze(0).expand_as(actions)

    def sample_best_actions(self, inputs):
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        return mu

    def sample_actions(self, inputs):
        mu = self.sample_best_actions(inputs)
        log_std = self.get_log_stds(mu)
        #std = torch.exp(log_std)
        eps = torch.randn(mu.size(), device=device)
        actions = torch.clamp(mu + log_std.exp()*(eps), -1, 1)
        return actions, mu

    def set_noise(self, noise):
        self.noise = noise

    def get_action(self, inputs):
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)
        return mu, log_std

    def get_value(self, inputs, device="cpu"):
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        return v
    def calculate_prob_gpu(self, inputs, actions):
        log_stds = self.get_log_stds(actions).to(device)
        #print(log_stds.shape)
        mean_actions = self.sample_best_actions(inputs)
        #print(mean_actions.shape)
        #w = self.get_w(inputs).to(device)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        #probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        #print(probs)
        return log_probs, mean_actions

    def calculate_prob(self, inputs, actions, mean_actions):
        log_stds = self.get_log_stds(actions)
        #print(log_stds.shape)
        # mean_actions = self.sample_best_actions(inputs)
        #print(mean_actions.shape)
        #w = self.get_w(inputs).to(device)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        #print(log_probs)
        #probs = (log_probs.exp() * w.t()).sum(dim=0).log()
        #print(probs)
        return log_probs


class ActorCriticNetMann(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64], num_contact=0):
        super(ActorCriticNetMann, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer

        self.policy_experts = [nn.ModuleList(), nn.ModuleList()]
        self.value_experts = [nn.ModuleList(), nn.ModuleList()]
        self.num_experts = 8
        self.num_gating_input = num_inputs #8
        self.num_additional_expert_input = num_inputs
        self.policy_layers = [
            (
                nn.Parameter(torch.empty(self.num_experts, num_inputs-self.num_gating_input+self.num_additional_expert_input, hidden_layer[0], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[0], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[0], hidden_layer[1], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[1], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[1], num_outputs, device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, num_outputs, device=device)),
                F.tanh,
            ),
        ]

        self.value_layers = [
            (
                nn.Parameter(torch.empty(self.num_experts, num_inputs-self.num_gating_input+self.num_additional_expert_input, hidden_layer[0], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[0], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[0], hidden_layer[1], device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, hidden_layer[1], device=device)),
                F.relu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, hidden_layer[1], 1, device=device)),
                nn.Parameter(torch.empty(self.num_experts, 1, 1, device=device)),
                lambda x: x,
            ),
        ]

        self.actor_params = []
        for index, (weight, bias, activation) in enumerate(self.policy_layers):
            if "relu" in activation.__name__:
                gain = 0.1
            elif "tanh" in activation.__name__:
                gain = 0.1
            else:
                gain = 1.0
            index = str(index)
            for w in weight:
                nn.init.orthogonal_(w, gain=gain)
            for b in bias:
                nn.init.orthogonal_(b, gain=gain)
            self.register_parameter("policy_w" + index, weight)
            self.register_parameter("policy_b" + index, bias)

            self.actor_params.append(weight)
            self.actor_params.append(bias)

        self.critic_params = []
        for index, (weight, bias, activation) in enumerate(self.value_layers):
            if activation is None:
                gain = 1.0
            elif "relu" in activation.__name__:
                gain = 0.1
            elif "tanh" in activation.__name__:
                gain = 0.1
            else:
                gain = 1.0
            index = str(index)
            for w in weight:
                nn.init.orthogonal_(w, gain=gain)
            for b in bias:
                nn.init.orthogonal_(b, gain=gain)
            self.register_parameter("value_w" + index, weight)
            self.register_parameter("value_b" + index, bias)

            self.critic_params.append(weight)
            self.critic_params.append(bias)

        self.policy_gate = nn.Sequential(
            nn.Linear(self.num_gating_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_experts),
        )

        self.value_gate = nn.Sequential(
            nn.Linear(self.num_gating_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_experts),
        )
        
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        # self.v = nn.Linear(self.hidden_layer_v[-1],1)
        self.noise = 0
        #self.train()

    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = torch.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

    def get_log_stds(self, actions):
        # print("\n\n\n\n", actions.shape, self.noise, self.num_outputs)
        return self.noise_tensor.unsqueeze(0).expand_as(actions)
        #return self.log_std.unsqueeze(0).expand_as(actions)

    def evaluate_policy_gate_l2(self, inputs):
        gating_weights = F.softmax(self.policy_gate(inputs[:, -self.num_gating_input:]), dim=1)
        return (gating_weights**2).mean()

    def evaluate_value_gate_l2(self, inputs):
        gating_weights = F.softmax(self.value_gate(inputs[:, -self.num_gating_input:]), dim=1)
        return (gating_weights**2).mean()

    def sample_best_actions(self, inputs):
        gating_weights = F.softmax(self.policy_gate(inputs[:, -self.num_gating_input:]), dim=1).t().unsqueeze(-1)
        out = inputs[:, :].clone()

        for (weight, bias, activation) in self.policy_layers[:-1]:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .mul(gating_weights)  # (B, H)
                .sum(dim=0)
            )
        weight, bias, activation = self.policy_layers[-1]
        out = activation(
            out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
            .add(bias)  # (N, B, H)
            .mul(gating_weights)  # (B, H)
            .sum(dim=0)
        )
        return out

    def sample_best_action_and_coefficients(self, inputs):
        gating_weights = F.softmax(self.policy_gate(inputs[:, -self.num_gating_input:]), dim=1).t().unsqueeze(-1)
        # out = inputs[:, :-self.num_gating_input+self.num_additional_expert_input].clone()
        out = inputs[:, :].clone()

        for (weight, bias, activation) in self.policy_layers[:-1]:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .mul(gating_weights)  # (B, H)
                .sum(dim=0)
            )
        weight, bias, activation = self.policy_layers[-1]

        out = activation(
            out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
            .add(bias)  # (N, B, H)
            .mul(gating_weights)  # (B, H)
            .sum(dim=0)
        )
        return out, gating_weights

    def sample_actions(self, inputs):
        mu = self.sample_best_actions(inputs)
        log_std = self.get_log_stds(mu)
        #std = torch.exp(log_std)
        eps = torch.randn(mu.size(), device=device)
        actions = torch.clamp(mu + log_std.exp()*(eps), -1, 1)
        return actions, mu

    def set_noise(self, noise):
        self.noise = noise
        self.noise_tensor = torch.Tensor(self.noise).to(device)

    def get_value(self, inputs, device="cpu"):
        gating_weights = F.softmax(self.value_gate(inputs[:, -self.num_gating_input:]), dim=1).t().unsqueeze(-1)
        # out = inputs[:, :-self.num_gating_input+self.num_additional_expert_input].clone()
        out = inputs[:, :].clone()

        for (weight, bias, activation) in self.value_layers[:-1]:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .mul(gating_weights)  # (B, H)
                .sum(dim=0)
            )

        weight, bias, activation = self.value_layers[-1]
        out = activation(
            out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
            .add(bias)  # (N, B, H)
            .mul(gating_weights)  # (B, H)
            .sum(dim=0)
        )            
        return out

    def calculate_prob_gpu(self, inputs, actions):
        log_stds = self.get_log_stds(actions)
        mean_actions = self.sample_best_actions(inputs)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        return log_probs, mean_actions


    def calculate_prob(self, inputs, actions, mean_actions):
        log_stds = self.get_log_stds(actions)
        numer = ((actions - mean_actions) / (log_stds.exp())).pow(2)
        log_probs = (-0.5 * numer).sum(dim=-1) - log_stds.sum(dim=-1)
        return log_probs