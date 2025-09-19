# ==============================================================================
# This is the complete code for mamba.py
# Copy this entire block.
# ==============================================================================

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Mamba(nn.Module):
    def _init_(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super()._init_()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.inv_dt = nn.Parameter(inv_dt)
        self.dt_proj.bias.data = self.inv_dt

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(z)
        return self.out_proj(y)

    def ssm(self, x):
        (d_inner, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bld,bln->bldn", delta, u, B)
        x = torch.zeros(b, d_in, n, device=u.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum("bdn,bln->bd", x, C[:, i].unsqueeze(1))
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y