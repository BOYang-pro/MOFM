import math
from functools import partial
from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_,to_2tuple
import numpy as np
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from model.backbone.selective_scan_interface import selective_scan_fn_v1
#from selective_scan_interface import selective_scan_fn_v1

# cross selective scan ===============================
if True:
    import selective_scan_cuda_oflex as selective_scan_cuda
    
    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True
            #The last item is to determine whether to output as float. For flase, output the same type as u
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows,False)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)
    
    class CrossScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, HW = x_rgb.shape
            ctx.shape = (B, C, HW)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * HW))
            xs_fuse[:, 0] = torch.concat([x_rgb, x_e], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, HW = ctx.shape
            L = 2 * HW
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:HW].view(B, -1, HW), ys[:, :, HW:2*HW].view(B, -1, HW) 
         
    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W, broadcast
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            return ys[:, :, 0:L//2], ys[:, :, L//2:L]
        
        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            # H, W = ctx.shape
            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2*L))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2*L)
            return xs, None, None

   
    def cross_selective_scan_multimodal_k2(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,    
        x_proj_bias: torch.Tensor=None,      
        dt_projs_weight: torch.Tensor=None,  
        dt_projs_bias: torch.Tensor=None,    
        A_logs: torch.Tensor=None,           
        Ds: torch.Tensor=None,               
        out_norm1: torch.nn.Module=None,     
        out_norm2: torch.nn.Module=None,     
        softmax_version=False,
        nrows = -1,                         
        delta_softplus = True,               
    ):
        
        B, D, HW = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * HW
        
        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1
        nrows = 1
    
        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, HW -> B, 2, C, 2 * HW
        
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1) #(B, K, C, L)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight) 

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # # Use the apply method of the SelectiveScan class for scanning operations
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*HW)

        # Used to separate the scan result ys into two outputs: y_rgb and y_e
        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, HW, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, HW, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)
        
        return y_rgb, y_e

# ===================# Normal scanning module==========================   
# =====================================================
       
class Self_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # Handle different modal data separately
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False, **factory_kwargs)
        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        
        
        #
        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        

        # A, D =======================================
        
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        # 
        self.out_norm_1 = nn.LayerNorm(self.d_model)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,**factory_kwargs):
        #Create the projection layer
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),"n -> d n",d=d_inner).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
       
        
    def forward(self, x_rgb: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        x_rgb = self.in_proj(x_rgb)
        B, L, d = x_rgb.shape
        #(B, L, d)==>(B, d, L)
        x_rgb = x_rgb.permute(0, 2, 1)
        # (B, d, L)==>(B * L, d)
        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)

        #x_dbl ：dt、B and C
        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        #
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()

        #dt_rgb and dt_e --> (B, d, L)
        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        # 
        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        
        #cross scan
        y_rgb = selective_scan(
            x_rgb, dt_rgb,
            A_rgb, B_rgb, C_rgb, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        
        # 
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        y_rgb = self.out_norm_1(y_rgb)
        
        return y_rgb


# =====================================================
class Cross_Mamba_Attention_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        
       
        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
     
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)

        # 
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,**factory_kwargs):
        #
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        #
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # 
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),"n -> d n",d=d_inner).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        #
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        #
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
       
    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        #(B, L, d)==>(B, d, L)
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        # (B, d, L)==>(B * L, d)
        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)

     
        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()

        
        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        
        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        
        y_rgb = selective_scan(
            x_rgb, dt_rgb,
            A_rgb, B_e, C_e, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        y_e = selective_scan(
            x_e, dt_e,
            A_e, B_rgb, C_rgb, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )
        
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        return y_rgb, y_e

# =====================================================
class CrossMambaFusion_SS2D_SSM(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        #
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1)//2,
            **factory_kwargs,
        )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self.CMA_ssm = Cross_Mamba_Attention_SSM(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            **kwargs,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        #
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        B, HW, D = x_rgb.shape

        if self.d_conv > 1:
            #[B, HW, D]===>[B, D, HW]
            x_rgb_trans = x_rgb.permute(0, 2, 1).contiguous()
            x_e_trans = x_e.permute(0, 2, 1).contiguous()
            #
            x_rgb_conv = self.act(self.conv1d(x_rgb_trans)) # (b, d, h, w)
            x_e_conv = self.act(self.conv1d(x_e_trans)) # (b, d, h, w)
            #[B, D, HW]===>[B, HW, D]
            x_rgb_conv = rearrange(x_rgb_conv, "b d hw -> b hw d")
            x_e_conv = rearrange(x_e_conv, "b d hw -> b hw d")
            #
            y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv) 
            # to b, d, h, w
            y_rgb = y_rgb.view(B, HW, -1)
            y_e = y_e.view(B, HW, -1)
        #
        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        out_e = self.dropout_e(self.out_proj_e(y_e))
        return out_rgb, out_e


class TSGBlock(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''
    def __init__(
        self,
        embed_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(embed_dim)
        self.op = CrossMambaFusion_SS2D_SSM(
            d_model=embed_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        


    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb_cross, x_e_cross = self.op(x_rgb, x_e)
        x_rgb = x_rgb + self.drop_path1(x_rgb_cross)
        x_e = x_e + self.drop_path2(x_e_cross)
        return x_rgb, x_e
        
DEV = False

class ConMB_SS2D(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_act = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx_act = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1)//2,
            **factory_kwargs)

            self.conv1d_modalx = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1)//2,
            **factory_kwargs)
           
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner*2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    # dt 
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))+ math.log(dt_min)).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        
        return dt_proj
    # A 
    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),"n -> d n",d=d_inner,).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log
    # D 
    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_k2(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None), self.softmax_version, 
            nrows=nrows,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        y_rgb_act = self.act(self.in_proj_act(x_rgb))
        y_e_act = self.act(self.in_proj_modalx_act(x_e))
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)

        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 2, 1).contiguous()
            x_e_trans = x_e.permute(0, 2, 1).contiguous()
            x_rgb_conv = self.act(self.conv1d(x_rgb_trans)) # (b, d, hw)
            x_e_conv = self.act(self.conv1d_modalx(x_e_trans)) # (b, d, hw)

            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv) # b, d, hw -> b, hw, d
            y_rgb = y_rgb*y_rgb_act
            y_e = y_e*y_e_act
            y = torch.concat([y_rgb, y_e], dim=-1)
        out = self.dropout(self.out_proj(y))
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class MambaFusionBlock(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''
    def __init__(
        self,
        embed_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        softmax_version=False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
    
        self.op = ConMB_SS2D(
            d_model=embed_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(embed_dim)
            mlp_embed_dim = int(embed_dim * mlp_ratio)
            self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_embed_dim, act_layer=act_layer, drop=drop, channels_first=False)


    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        fuse_x1x2=self.drop_path(self.op(x_rgb, x_e))
        x = x_rgb + x_e + fuse_x1x2
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4,  embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops

#===========
class Cross_scan(nn.Module):
    def __init__(self,img_size=256, 
                    in_channel=48,
                    embed_dim=48, 
                    patch_size=4,
                    depth=4, 
                    d_state=16,
                    dt_rank="auto",  
                    ssm_ratio=2.0,     
                    mlp_ratio=4.,
                    drop_path=0.,
                    norm_layer=nn.LayerNorm,
                    ape=False 
                    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ape=ape
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, embed_dim= self.embed_dim, norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = DropPath(drop_path)
        
        self.blocks = nn.ModuleList([
            TSGBlock(
                    embed_dim=embed_dim,  
                    mlp_ratio=mlp_ratio,   
                    d_state=d_state,      
                    drop_path=drop_path,   
                    ssm_ratio=ssm_ratio,   
                    dt_rank=dt_rank,  
                )
            for _ in range(depth)])
        self.norm_A = norm_layer(embed_dim)
        self.norm_B = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        
    def forward(self, x, y):
        image_size = (x.shape[2], x.shape[3])
        #
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        
        x=self.pos_drop(x)
        y=self.pos_drop(y)

        for blk in self.blocks:
            x,y = blk(x_rgb=x,x_e=y)

        x = self.norm_A(x)
        y = self.norm_B(y)
        
        x=self.patch_unembed(x,image_size)
        y=self.patch_unembed(y,image_size)
        return x,y
        
# handle multiple input 
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
                



##==================================
class ConvBnLeakyRelu2d(nn.Module):
    '''Conv2d + BN + LeakyReLU'''
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)
class Fusion_Encoder(nn.Module):
    def __init__(self,img_size=256, 
                    in_channel=1,
                    embed_dim=96, 
                    patch_size=4,
                    depth=4, 
                    d_state=16,
                    dt_rank="auto",  
                    ssm_ratio=2.0,     
                    mlp_ratio=4.,
                    drop_path=0.,
                    drop_path_rate=0., 
                    norm_layer=nn.LayerNorm,
                    ape=False 
                    ):
        super().__init__()
        self.embed_dim = embed_dim
        embed_dim_temp= int(embed_dim / 2)
        self.ape=ape
        ####shallow feature extraction 
        self.conv_first1_A = nn.Conv2d(in_channel, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.feat_A = nn.ModuleList([Self_SSM(
            d_model=self.embed_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank
        )  for _ in range(1)])
        self.feat_B = nn.ModuleList([Self_SSM(
            d_model=self.embed_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank
        )  for _ in range(1)])
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, embed_dim= self.embed_dim, norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = DropPath(drop_path)
        self.ups_A = nn.ModuleList([
            ConvBnLeakyRelu2d(self.embed_dim * (2 ** (depth - 2 - i)), (self.embed_dim * (2 ** (depth - 2 - i))) // 2)
            for i in range(depth-1)
        ])  #4 layer
        self.ups_B = nn.ModuleList([
            ConvBnLeakyRelu2d(self.embed_dim * (2 ** (depth - 2 - i)), (self.embed_dim * (2 ** (depth - 2 - i))) // 2)
            for i in range(depth-1)
        ])  #4 layer
        self.blocks = nn.ModuleList([
            TSGBlock(
                    embed_dim=embed_dim,  
                    mlp_ratio=mlp_ratio,   
                    d_state=d_state,      
                    drop_path=drop_path,   
                    ssm_ratio=ssm_ratio,   
                    dt_rank=dt_rank,  
                )
            for _ in range(2)])
        self.norm_A = norm_layer(embed_dim)
        self.norm_B = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        
    def forward(self, x, y):
        ####shallow feature extraction
        x = self.lrelu(self.conv_first1_A(x))       
        image_size = (x.shape[2], x.shape[3])
        y = self.lrelu(self.conv_first1_A(y))
                  
        
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        
        x=self.pos_drop(x)
        y=self.pos_drop(y)
        ####deep feature extraction
        for layer in self.feat_A:
            x = layer(x)
            
        for layer in self.feat_B:
            y = layer(y)  
        
        y = self.patch_unembed(y, image_size) 
        x = self.patch_unembed(x, image_size)   
        y = self.patch_embed(y)
        x = self.patch_embed(x)
        #
        for blk in self.blocks:
            x,y = blk(x_rgb=x,x_e=y)

        x_rgb = self.norm_A(x)
        y_e = self.norm_B(y)
        
        return x_rgb,y_e,image_size
        



class Fusion_Decoder(nn.Module):
    def __init__(self, img_size=256, 
                       out_channel=1,
                       embed_dim=96,
                       patch_size=4,
                       depth=4,
                       d_state=4,
                       ssm_ratio=2.0,
                       norm_layer=nn.LayerNorm, 
                       drop_path=0.,
                       attn_drop_rate=0.,
                       mlp_ratio=4.0,
                       drop=0.1,
                       dt_rank="auto",
                       ape=False,
                       ):
        super().__init__()
        self.ape=ape
        self.embed_dim = embed_dim
        self.embed_dim_temp = int(self.embed_dim / 2)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=out_channel, embed_dim=embed_dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, embed_dim= self.embed_dim, norm_layer=norm_layer)
        self.Fusion_blocks = MambaFusionBlock(
                        embed_dim=self.embed_dim,
                        drop_path=drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop_rate,
                        d_state=d_state,
                        ssm_ratio=ssm_ratio,
                        mlp_ratio=mlp_ratio,
                        drop=drop
                      )
        num_patches = self.patch_embed.num_patches
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = DropPath(drop_path)
        
        self.Re_norm = norm_layer(embed_dim)
        self.Re_blocks = nn.ModuleList([Self_SSM(
            d_model=self.embed_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank
        )  for _ in range(depth)])
       
        self.Re_drop_path = DropPath(drop_path)
        self.conv_last1 = nn.Conv2d(self.embed_dim, int(self.embed_dim_temp/2), 3, 1, 1)
        self.conv_last3 = nn.Conv2d(int(self.embed_dim_temp/2), out_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, y,image_size):
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        
        x=self.pos_drop(x)
        y=self.pos_drop(y)
        #
        f=self.Fusion_blocks(x,y)
        for layer in self.Re_blocks:
            f = layer(f)
        f = self.Re_norm(f)  # B L C
        f = self.patch_unembed(f, image_size)
        # Convolution 
        f = self.lrelu(self.conv_last1(f))
        f = self.conv_last3(f) 
        return f

class Fusion_frarework(nn.Module):
    def __init__(self, img_size=256, 
                       in_channel=1,
                       embed_dim=32,
                       patch_size=4,
                       depth=4,
                       d_state=4,
                       ssm_ratio=2.0,
                       norm_layer=nn.LayerNorm, 
                       drop_path=0.,
                       attn_drop_rate=0.,
                       mlp_ratio=4.0,
                       drop=0.1,
                       dt_rank="auto",
                       ape=False,
                       ):
        super().__init__()
        self.Encoder=Fusion_Encoder(img_size=img_size, 
                    in_channel=in_channel,
                    embed_dim=embed_dim, 
                    patch_size=patch_size,
                    depth=depth, 
                    d_state=d_state,
                    dt_rank=dt_rank,  
                    ssm_ratio=ssm_ratio, 
                    norm_layer=norm_layer,     
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    ape=False,
                    )
                
                    
        self.Decoder=Fusion_Decoder(img_size=img_size, 
                       out_channel=in_channel,
                       embed_dim=embed_dim,
                       patch_size=patch_size,
                       depth=2,
                       d_state=d_state,
                       dt_rank=dt_rank,
                       ssm_ratio=ssm_ratio,
                       norm_layer=norm_layer, 
                       drop_path=drop_path,
                       attn_drop_rate=attn_drop_rate,
                       mlp_ratio=mlp_ratio,
                       drop=drop,
                       ape=ape,
                       )

   

    def forward(self, x, y):
    
        output1,output2,image_size= self.Encoder(x,y)
        fusion_result= self.Decoder(output1,output2,image_size)  
            
        return fusion_result
    

if __name__ == '__main__':
    from model.backbone.Reg_mamba_cross import Flow_Net
    from fvcore.nn import flop_count,FlopCountAnalysis, parameter_count_table
    model = Flow_Net(init_embed=32,in_channels=1).cuda(0)
    model.eval()
    a = torch.randn(1, 1, 256, 256).cuda(0)
    b = torch.randn(1, 1, 256, 256).cuda(0)
    flow,mov_encs,fix_aligneds,fusion_fix_encs,fusion_warp_encs = model(a,b)
    model_f=Fusion_frarework().cuda(0)  
    fuse_result=model_f(a, b,flow,fusion_fix_encs,fusion_warp_encs)     
    print('fuse_result:',fuse_result.shape) 
    tensor = (a, b,flow,fusion_fix_encs,fusion_warp_encs,)
    flops1 = FlopCountAnalysis(model_f, tensor)
    flops1=flops1.total()/1e9
    print("FLOPs:",flops1) 
    # Model Size
    total1 = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total1 / 1e6))
    
    # Model Size
    total2 = sum([param.nelement() for param in model_f.parameters()])
    print("Number of parameter: %.3fM" % (total2 / 1e6))
    
    print("All parameter: %.3fM" % ((total2+total1) / 1e6))
  