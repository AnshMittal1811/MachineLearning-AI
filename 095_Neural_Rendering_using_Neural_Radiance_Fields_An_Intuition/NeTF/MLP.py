import torch
import time

class Network_S_Relu(torch.nn.Module):
    def __init__(self, D=8, H=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], no_rho=False):
        super(Network_S_Relu, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.no_rho = no_rho
        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, H)] + [torch.nn.Linear(H, H) if i not in self.skips else torch.nn.Linear(H + input_ch, H) for i in range(D-1)])
        self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + H, H//2)])
        if self.no_rho:
            self.output_linear = torch.nn.Linear(H, output_ch)
        else:
            self.feature_linear = torch.nn.Linear(H, H)
            self.alpha_linear = torch.nn.Linear(H, 1)
            self.rho_linear = torch.nn.Linear(H//2, 1)

    def forward(self, x):
        # y_pred = self.linear(x)
        if self.no_rho:
            input_pts = x
            h = x
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.no_rho:
            outputs = self.output_linear(h)
        else:
            alpha = self.alpha_linear(h)
            alpha = torch.abs(alpha)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = torch.nn.functional.relu(h)
            rho = self.rho_linear(h)
            rho = torch.abs(rho)
            outputs = torch.cat([rho, alpha], -1)
        return outputs
