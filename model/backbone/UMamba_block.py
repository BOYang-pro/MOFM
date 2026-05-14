import torch
import torch.nn as nn
import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from model.backbone.selective_scan_interface_UMamba import selective_scan_fn_v1
import numpy as np
from hilbert import decode
from pyzorder import ZOrderIndexer

import math
from functools import partial
from typing import Callable

# This class encodes and decodes images using different scanning patterns.
'''
sweep:    Linear sweep (default). Indices are visited in sequential order.
scan:     Standard row-by-row scan; odd rows are traversed in reverse order.
zorder:   Scanning based on the Z-order (Morton) curve.
zigzag:   Zigzag scanning pattern.
hilbert:  Scanning based on the Hilbert space-filling curve.
'''
class HSCANS(nn.Module):
	def __init__(self, size=16, dim=2, scan_type='hilbert', ):
		super().__init__()
		size = int(size)
		max_num = size ** dim
		indexes = np.arange(max_num)
		if 'sweep' == scan_type:  # ['sweep', 'scan', 'zorder', 'zigzag', 'hilbert']
			locs_flat = indexes
		elif 'scan' == scan_type:
			indexes = indexes.reshape(size, size)
			for i in np.arange(1, size, step=2):
				indexes[i, :] = indexes[i, :][::-1]
			locs_flat = indexes.reshape(-1)
		elif 'zorder' == scan_type:
			zi = ZOrderIndexer((0, size - 1), (0, size - 1))
			locs_flat = []
			for z in indexes:
				r, c = zi.rc(int(z))
				locs_flat.append(c * size + r)
			locs_flat = np.array(locs_flat)
		elif 'zigzag' == scan_type:
			indexes = indexes.reshape(size, size)
			locs_flat = []
			for i in range(2 * size - 1):
				if i % 2 == 0:
					start_col = max(0, i - size + 1)
					end_col = min(i, size - 1)
					for j in range(start_col, end_col + 1):
						locs_flat.append(indexes[i - j, j])
				else:
					start_row = max(0, i - size + 1)
					end_row = min(i, size - 1)
					for j in range(start_row, end_row + 1):
						locs_flat.append(indexes[j, i - j])
			locs_flat = np.array(locs_flat)
		elif 'hilbert' == scan_type:
			bit = int(math.log2(size))
			locs = decode(indexes, dim, bit)
			locs_flat = self.flat_locs_hilbert(locs, dim, bit)
		else:
			raise Exception('invalid encoder mode')
		locs_flat_inv = np.argsort(locs_flat)
		index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
		index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
		self.index_flat = nn.Parameter(index_flat, requires_grad=False)
		self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

	def flat_locs_hilbert(self, locs, num_dim, num_bit):
		ret = []
		l = 2 ** num_bit
		for i in range(len(locs)):
			loc = locs[i]
			loc_flat = 0
			for j in range(num_dim):
				loc_flat += loc[j] * (l ** j)
			ret.append(loc_flat)
		return np.array(ret).astype(np.uint64)

	def __call__(self, img):
		img_encode = self.encode(img)
		return img_encode

	def encode(self, img):
		img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat_inv.expand(img.shape), img)
		return img_encode

	def decode(self, img):
		img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat.expand(img.shape), img)
		return img_decode

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            ssm_ratio=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            size=8,
            scan_type='hilbert',
            num_direction=8,
            prev_state_chan=None,
            skip_state_chan=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(self.ssm_ratio * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.num_direction = num_direction

        x_proj_weight = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight for _ in range(self.num_direction)]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))
        dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(self.num_direction)]

        ####################################################################3
        # previous ssm_state cache gating =======================
        ####################################################################3
        k_group = 4
        if prev_state_chan is not None:
            self.prev_sta_proj_w = nn.Parameter(torch.randn((k_group*self.d_inner, prev_state_chan, 1)))
            if skip_state_chan is not None:
                self.skip_sta_proj_w = nn.Parameter(torch.randn((k_group*self.d_inner, self.d_inner, 1)))
            else:
                self.skip_sta_proj_w = None
            self.xs_gate_weight = nn.Parameter(torch.randn((k_group*d_state, self.d_inner, 1)))
            self.ssm_gate_ratio = nn.Parameter(torch.randn(1, k_group, self.d_inner, 1))
        else:
            self.prev_sta_proj_w = None
            self.skip_sta_proj_w = None
            self.xs_gate_weight = None
            self.ssm_gate_ratio = None
        self.Ds = self.D_init(self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)
        self.dt_projs_weight = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs], dim=0))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.scans = HSCANS(size=size, scan_type=scan_type)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor,
                        prev_state: torch.Tensor=None,
                        skip_state: torch.Tensor=None,
                        no_einsum=True):
        self.selective_scan = selective_scan_fn_v1
        B, C, H, W = x.shape
        L = H * W
        _, N = self.A_logs.shape
        _, D, R = self.dt_projs_weight.shape
        K = self.num_direction

        xs = []
        if K >= 2:

            xs.append(self.scans.encode(x.view(B, -1, L)))
        if K >= 4:
            xs.append(self.scans.encode(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)))
        if K >= 8:
            xs.append(self.scans.encode(torch.rot90(x, k=1, dims=(2, 3)).contiguous().view(B, -1, L)))
            xs.append(self.scans.encode(torch.transpose(torch.rot90(x, k=1, dims=(2, 3)), dim0=2, dim1=3).contiguous().view(B, -1, L)))
        xs = torch.stack(xs,dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=None, groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L)
            dts = F.conv1d(dts, self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            # previous states cache
            if skip_state is not None and self.skip_sta_proj_w is not None:
                skip_state = F.conv1d(skip_state,   # [b, K*D, d_state]
                                  self.skip_sta_proj_w,   # [K*D, D, 1]
                                  bias=None,
                                  groups=K)
    
            if prev_state is not None and self.prev_sta_proj_w is not None:
                # [B, D, d_state]
                prev_state = F.conv1d(prev_state, self.prev_sta_proj_w, bias=None, groups=K)
            
                if skip_state is not None:
                    prev_state = prev_state + skip_state
                gating = F.conv1d(xs.view(B, -1, L), self.xs_gate_weight,bias=None, groups=K)
                upd = torch.einsum('bkdn,bknl->bkdl', prev_state.view(B, K, -1, N), gating.view(B, K, -1, L))
                xs = xs + upd * self.ssm_gate_ratio
            else:    
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
                dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)


        out_y, ssm_state = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        )
        out_y=out_y.view(B, K, -1, L)
        assert out_y.dtype == torch.float


        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []
        if K >= 2:
            ys.append(self.scans.decode(out_y[:, 0]))
            ys.append(self.scans.decode(inv_y[:, 0]))
        if K >= 4:
            ys.append(torch.transpose(self.scans.decode(out_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L))
            ys.append(torch.transpose(self.scans.decode(inv_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L))
        if K >= 8:
            ys.append(torch.rot90(self.scans.decode(out_y[:, 2]).view(B, -1, W, H), k=3, dims=(2,3)).contiguous().view(B, -1, L))
            ys.append(torch.rot90(self.scans.decode(inv_y[:, 2]).view(B, -1, W, H), k=3, dims=(2,3)).contiguous().view(B, -1, L))
            ys.append(torch.rot90(torch.transpose(self.scans.decode(out_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3), k=3, dims=(2,3)).contiguous().view(B, -1, L))
            ys.append(torch.rot90(torch.transpose(self.scans.decode(inv_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3), k=3, dims=(2,3)).contiguous().view(B, -1, L))
        y = sum(ys)
        return y, ssm_state

    def forward(self, x: torch.Tensor, prev_states: torch.Tensor=None,skip_states:torch.Tensor=None):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y, ssm_state = self.forward_core(x, prev_state=prev_states, skip_state=skip_states)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out, ssm_state

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        mlp_drop_rate: float = 0.0,
        d_state: int = 16,
        d_conv: int = 3,
        ssm_ratio: float = 2.0,
		    size: int = 8,    
		    scan_type='hilbert',  
		    num_direction=4,
        prev_state_chan: int = None,
        skip_state_chan: int = None,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.prev_state_gate = prev_state_chan is not None
        self.op = SS2D(d_model=hidden_dim,
                                   dropout=mlp_drop_rate, 
                                   ssm_ratio=ssm_ratio,
                                   d_state=d_state, 
                                   d_conv=d_conv,
                                   size=size, 
                                   scan_type=scan_type, 
                                   num_direction=num_direction,
                                   prev_state_chan=prev_state_chan,
                                   skip_state_chan=skip_state_chan,
                                   )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor, *ssm_state: tuple[torch.Tensor]):
        x, ssm_state = self.op(self.ln_1(input), *ssm_state)
        x = input + self.drop_path(x)
        return x,ssm_state


# 
class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class MambaBlock(nn.Module):
    def __init__(self,in_chan, 
                  inner_chan, 
                  ssm_conv=[3, 11], 
                  window_size=8,  
                  global_size=256,
                  d_states=[16, 32],  
                  ssm_ratio=2.0,
                  drop_path=0.0,
                  prev_state_chan=None,
                  skip_state_chan=None,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  num_direction = 4,
                  scan_type='hilbert',
    ):
        super().__init__()
        self.inner_chan = inner_chan
        self.window_size = window_size

        ssm_local_conv, ssm_global_conv = ssm_conv[0], ssm_conv[1]
        local_d_state, global_d_state = d_states[0], d_states[1]

        self.mamba = VSSBlock(
            hidden_dim=inner_chan,
            drop_path=drop_path,
            norm_layer=norm_layer,
            d_state=global_d_state,
            d_conv=ssm_global_conv,
            ssm_ratio=ssm_ratio,
            size=global_size,
            scan_type=scan_type,
            num_direction=num_direction,
            prev_state_chan=prev_state_chan,
            skip_state_chan=skip_state_chan,
        )

    def forward(self, feat: torch.Tensor,
                prev_local_state: torch.Tensor = None,
                prev_global_state: torch.Tensor = None,
                skip_local_state: torch.Tensor = None,
                skip_global_state: torch.Tensor = None):

        b,h,w,c=feat.shape
        
        x = feat 

        # check cache for mamba blocks
        if not self.mamba.prev_state_gate:
            prev_global_state = None

        
        local_ssm_state = None
        
        
        x, global_ssm_state = self.mamba(x, prev_global_state, skip_global_state)
        return x, local_ssm_state, global_ssm_state
            
class Permute(nn.Module):
    def __init__(self, mode="c_first"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "c_first":
            # b h w c -> b c h w
            return x.permute(0, 3, 1, 2)
        elif self.mode == "c_last":
            # b c h w -> b h w c
            return x.permute(0, 2, 3, 1)
        else:
            raise NotImplementedError
        
    def __repr__(self):
        return f"Permute(mode={self.mode})"      

def down(chan, down_type='patch_merge', permute=False, r=2, chan_r=2):
    if down_type == 'conv':
        return nn.Sequential(
            # Rearrange('b h w c -> b c h w', h=h, w=w),
            Permute("c_first") if permute else nn.Identity(),
            nn.Conv2d(chan, chan * chan_r, r, r),
            # Rearrange('b c h w -> b h w c'),
            Permute("c_last") if permute else nn.Identity(),
        )
    elif down_type == 'patch_merge':
        return PatchMerging2D(chan, chan*2)
    else:
        raise NotImplementedError(f'down type {down_type} not implemented')
class UniSequential(nn.Module):
    def __init__(self, *args: tuple[nn.Module]):
        super().__init__()
        self.mods = nn.ModuleList(args)

    def __getitem__(self, idx):
        return self.mods[idx]

    
    # @get_local('outp', 'global_state')
    def LEMM_enc_forward(self, feat, cond,states=[None, None]):
        outp = feat
        local_state, global_state = states[0], states[1]
        for i, mod in enumerate(self.mods):
            outp, local_state, global_state = mod(outp, cond, local_state, global_state) # in_block states share
        return outp, (local_state, global_state)

    def LEMM_dec_forward(self, feat, cond, prev_states=[None, None],skip_states=[None, None]):

        feat = self.mods[0](feat)
        outp = feat
        for i, mod in enumerate(self.mods[1:]):
            outp, local_state, global_state = mod(outp, cond, *(prev_states + skip_states))  # in_block states share

        return outp, (local_state, global_state)
    
class Feat_Encoder(nn.Module):
    def __init__(
        self,
        img_channel=1,
        width=32,
        img_size=128,
        # LEMMBlock settings
        ssm_enc_blk_nums=[2, 1, 1],
        middle_blk_nums=1,
        ssm_chan_upscale=[2,2,2],
        ssm_enc_convs=[[7, 11], [7, 11], [None, 11]],
        ssm_ratios=[3,2,2],
        ssm_enc_d_states=[[16, 32], [16, 32], [None, 32]],
        window_sizes=[8,8,None],
        # model settings
        upscale=1,
        drop_path_rate=0.2,
    ):
        super().__init__()
        self.upscale = upscale

        #
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, stride=1, padding=1, bias=True)
        self.down_intro =nn.Conv2d(width, width, 2, 2)
        ## main body
        self.lemm_encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.lemm_downs = nn.ModuleList()
        
        depth =  middle_blk_nums + sum(ssm_enc_blk_nums) 
        inter_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inner_chan = width
        n_prev_blks = 0
        
        ## encoder
        # LEMM layer
       
        for enc_i, num in enumerate(ssm_enc_blk_nums):
            def prev_state_chan_fn(i, mod='enc'):
                if mod == 'enc':
                    if enc_i == 0:
                        if i == 0: prev_state_chan = None
                        else: prev_state_chan = inner_chan * ssm_ratios[enc_i]
                    else:
                        if i == 0: prev_state_chan = None
                        else: prev_state_chan = inner_chan * ssm_ratios[enc_i]
                elif mod == 'mid':
                    if i == 0: prev_state_chan = None
                    else: prev_state_chan = inner_chan * ssm_ratios[-1]
                return prev_state_chan
            global_sizes=[img_size/2,img_size/4,img_size/8]
            self.lemm_encoders.append(UniSequential(
                    *[MambaBlock(
                            img_channel,
                            inner_chan,
                            ssm_conv=ssm_enc_convs[enc_i],
                            window_size=window_sizes[enc_i],
                            global_size=global_sizes[enc_i],
                            d_states=ssm_enc_d_states[enc_i],
                            ssm_ratio=ssm_ratios[enc_i],
                            drop_path=inter_dpr[n_prev_blks + i],
                            prev_state_chan=prev_state_chan_fn(i, mod='enc'),
                            )  for i in range(num)]))
            self.lemm_downs.append(down(inner_chan, down_type='conv', permute=True, r=2, chan_r=ssm_chan_upscale[enc_i]))
            inner_chan = inner_chan * ssm_chan_upscale[enc_i]
            n_prev_blks += num

        ## middel laye
        self.middle_blks = UniSequential(
            *[MambaBlock(
                    img_channel,
                    inner_chan,
                    ssm_conv=ssm_enc_convs[-1],
                    window_size=window_sizes[-1],
                    global_size=img_size/16,
                    d_states=ssm_enc_d_states[-1],
                    ssm_ratio=ssm_ratios[-1],
                    drop_path=inter_dpr[n_prev_blks + i],
                    prev_state_chan=prev_state_chan_fn(i, mod='mid'))  for i in range(middle_blk_nums)]
        )
        n_prev_blks += middle_blk_nums
        # init
        self.apply(self._init_weights)


    def _init_weights(self, m: nn.Module):
        # print(type(m))
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.intro(x)
        x=self.down_intro(x)
        x = x.permute(0, 2, 3, 1)
        lemm_encs = []
        states = [None, None]
        for i, (encoder, down) in enumerate(zip(self.lemm_encoders, self.lemm_downs)):
            # TODO: input previous state
            x, states = encoder.LEMM_enc_forward(x, states)
            x_private= rearrange(x, 'b h w c -> b c h w')
            lemm_encs.append(x_private)
            x = down(x)
        x, states = self.middle_blks.LEMM_enc_forward(x, states)
        x = rearrange(x, 'b h w c -> b c h w')
        lemm_encs.append(x)
        
        return lemm_encs[0],lemm_encs[1],lemm_encs[2],lemm_encs[3]


