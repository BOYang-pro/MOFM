import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.backbone.Fusion_mamba import  Cross_scan
from model.backbone.UMamba_block import  Feat_Encoder
shape = [256, 256]
# shape = [256, 320]
# shape = [240, 320]

class ConvBnLeakyRelu2d(nn.Module):
    '''Conv2d + BN + LeakyReLU'''
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)

class ConvBlock(nn.Module):
    '''Conv2d + LeakyReLU'''
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.1)
        
class ConvResBlock(nn.Module):
    '''Conv2d + LeakyReLU'''
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out=self.conv(x)
        out=self.shortcut(x)+out
        return F.leaky_relu(out, negative_slope=0.1)
        
        
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


   
class TransConvBnLeakyRelu2d(nn.Module):
    '''ConvTranspose2d + BN + LeakyReLU'''
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, dilation=1, groups=1):
        super(TransConvBnLeakyRelu2d, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.transconv(x)), negative_slope=0.1)

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        gpu_use = True
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='zeros', align_corners=True), new_locs

class ConvSigmoid(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvSigmoid, self).__init__()
        self.conv     = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.sigmoid  = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv(x))
        
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """
    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)[0]
        return vec

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.avg_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias)
        )
        # global max pooling: feature --> point
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.max_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_fc(self.avg_pool(x))
        max_out = self.max_fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
      
        return out
        
class Bottleneck_flow(nn.Module):
    def __init__(self, in_channels, dim=2):
        super(Bottleneck_flow, self).__init__()
        #
        self.bottle_1 = ConvBnLeakyRelu2d(2*in_channels, in_channels, 3)
        self.bottle_2 = ConvBnLeakyRelu2d(in_channels, in_channels, 3)
        self.upsample = TransConvBnLeakyRelu2d(in_channels, in_channels//2, 4, 2, 1)
        #
        self.Conv1 = ConvBlock(in_channels, in_channels)
        self.Conv2 = ConvResBlock(in_channels, in_channels)
        self.Conv3 = ConvResBlock(in_channels, in_channels)
        self.Conv31 = ConvResBlock(in_channels, in_channels)
        self.Conv4 = nn.Conv2d(in_channels, in_channels//2, 3,padding=1, bias=False, stride = 1)
        self.Conv5 = nn.Conv2d(in_channels//2, dim, 3,padding=1, bias=False, stride = 1)
    def forward(self, x):
        bottle0 = self.bottle_2(self.bottle_1(x))
        cost_vol = self.Conv1(bottle0)
        cost_vol = self.Conv2(cost_vol)
        cost_vol = self.Conv3(cost_vol)
        cost_vol = self.Conv31(cost_vol)
        flow = self.Conv5(self.Conv4(cost_vol))
        bottle0_up=self.upsample(bottle0)
        return flow,bottle0_up 
                     
class FlowNet(nn.Module):
    def __init__(self, in_channels,dim=2,task_classes=2,emb_dim=32):
        super(FlowNet, self).__init__()
        self.channels = in_channels
        
        self.downconv_1 = nn.Sequential(conv(3 * self.channels, self.channels, kernel_size=1),
                                        nn.ReLU(inplace=True))
        
        self.plainconv  = nn.Sequential(conv(self.channels, self.channels, kernel_size=3),
                                        nn.ReLU(inplace=True))

        self.downconv_2 = nn.Sequential(conv(self.channels, 3, kernel_size=3),
                                        nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=-1)

        self.ca_layer = ChannelAttention(3 * self.channels)
        self.upsample = TransConvBnLeakyRelu2d(3 * self.channels, self.channels//2, 4, 2, 1)
        #
        self.Conv1 = ConvBlock(3 *in_channels, 3 *in_channels)
        self.Conv2 = ConvResBlock(3 *in_channels, 3 *in_channels)
        self.Conv3 = ConvResBlock(3 *in_channels, 3 *in_channels)
        self.Conv31 = ConvResBlock(3 *in_channels, 3 *in_channels)
        self.Conv4 = nn.Conv2d(3 *in_channels, 3 *in_channels//2, 3,padding=1, bias=False, stride = 1) 
        self.Conv5 = nn.Conv2d(3 *in_channels//2, dim, 3,padding=1, bias=False, stride = 1) 
        
    def forward(self, dec, mov_warp, fix_enc):
        
        dec_feat_in = torch.cat([dec, mov_warp, fix_enc], dim=1)
        dec_feat  = self.plainconv(self.downconv_1(dec_feat_in)) # [B, C, H, W] torch.Size([2, 32, 64, 64])
        down_feat_in = self.downconv_2(dec_feat) # [B, 3, H, W] torch.Size([2, 3, 64, 64])
        down_feat    = down_feat_in.view(down_feat_in.shape[0], down_feat_in.shape[1], -1) # [B, 3, H * W] torch.Size([2, 3, 4096])

        weight_maps = self.softmax(down_feat) # [B, 3, H * W] torch.Size([2, 3, 4096])
        weight_maps = weight_maps.view(down_feat_in.shape[0], down_feat_in.shape[1], down_feat_in.shape[2], down_feat_in.shape[3]) # [B, 3, H, W] torch.Size([2, 3, 64, 64])
        spatial_dec      = dec * weight_maps[:, :1, :, :]
        spatial_mov_warp = mov_warp * weight_maps[:, 1:2, :, :]
        spatial_fix_enc  = fix_enc * weight_maps[:, 2:, :, :]
        spatial_fms = torch.cat([spatial_dec, spatial_mov_warp, spatial_fix_enc], dim=1) # torch.Size([2, 96, 64, 64])

        channel_wise = self.ca_layer(spatial_fms) # torch.Size([2, 96, 1, 1])
        output = channel_wise * spatial_fms # torch.Size([2, 96, 64, 64])
        
        cost_vol = self.Conv1(output)
        cost_vol = self.Conv2(cost_vol)
        cost_vol = self.Conv3(cost_vol)
        cost_vol = self.Conv31(cost_vol)
        flow = self.Conv5(self.Conv4(cost_vol))
        
        feat_up=self.upsample(output)
        

        return flow,feat_up
class PromptCostAttention(nn.Module):
    def __init__(self, dim, prompt_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(prompt_dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, cost_vol, prompt):
        """
        cost_vol: [B, C, H, W]
        prompt:   [B, prompt_dim]
        """
        B, C, H, W = cost_vol.shape
        N = H * W

        cost = cost_vol.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        Q = self.q(prompt).unsqueeze(1)                 # [B, 1, C]
        K = self.k(cost)                                # [B, N, C]
        V = self.v(cost)

        attn = (Q @ K.transpose(-2, -1)) * self.scale   # [B, 1, N]
        attn = attn.softmax(dim=-1)

        out = attn @ V                                  # [B, 1, C]
        out = self.proj(out).repeat(1, N, 1)            # broadcast

        out = out.permute(0, 2, 1).view(B, C, H, W)

        return cost_vol + out

class FlowNet_last(nn.Module):
    def __init__(self, in_channels,dim=2,prompt=True,task_classes=2,emb_dim=32):
        super(FlowNet_last, self).__init__()
        self.channels = in_channels
        self.prompt=prompt
        if self.prompt:
            self.prompt_task = nn.Parameter(torch.randn(1, task_classes, 1, emb_dim))
            self.prompt_attn = PromptCostAttention(dim=3 * in_channels,prompt_dim=emb_dim)  
        self.downconv_1 = nn.Sequential(conv(3 * self.channels, self.channels, kernel_size=1),
                                        nn.ReLU(inplace=True))
        
        self.plainconv  = nn.Sequential(conv(self.channels, self.channels, kernel_size=3),
                                        nn.ReLU(inplace=True))

        self.downconv_2 = nn.Sequential(conv(self.channels, 3, kernel_size=3),
                                        nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=-1)

        self.ca_layer = ChannelAttention(3 * self.channels)
        #
        self.Conv1 = ConvBlock(3 *in_channels, 3 *in_channels)
        self.Conv2 = ConvResBlock(3 *in_channels, 3 *in_channels)
        self.Conv3 = ConvResBlock(3 *in_channels, 3 *in_channels)
        self.Conv31 = ConvResBlock(3 *in_channels, 3 *in_channels)
        self.Conv4 = nn.Conv2d(3 *in_channels, 3 *in_channels//2, 3,padding=1, bias=False, stride = 1) 
        self.Conv5 = nn.Conv2d(3 *in_channels//2, dim, 3,padding=1, bias=False, stride = 1) 
        
    def forward(self, dec, mov_warp, fix_enc,prompt_token):
        dec_feat_in = torch.cat([dec, mov_warp, fix_enc], dim=1)
        dec_feat  = self.plainconv(self.downconv_1(dec_feat_in)) # [B, C, H, W] torch.Size([2, 32, 64, 64])
        down_feat_in = self.downconv_2(dec_feat) # [B, 3, H, W] torch.Size([2, 3, 64, 64])
        down_feat    = down_feat_in.view(down_feat_in.shape[0], down_feat_in.shape[1], -1) # [B, 3, H * W] torch.Size([2, 3, 4096])

        weight_maps = self.softmax(down_feat) # [B, 3, H * W] torch.Size([2, 3, 4096])
        weight_maps = weight_maps.view(down_feat_in.shape[0], down_feat_in.shape[1], down_feat_in.shape[2], down_feat_in.shape[3]) # [B, 3, H, W] torch.Size([2, 3, 64, 64])
        spatial_dec      = dec * weight_maps[:, :1, :, :]
        spatial_mov_warp = mov_warp * weight_maps[:, 1:2, :, :]
        spatial_fix_enc  = fix_enc * weight_maps[:, 2:, :, :]
        spatial_fms = torch.cat([spatial_dec, spatial_mov_warp, spatial_fix_enc], dim=1) # torch.Size([2, 96, 64, 64])

        channel_wise = self.ca_layer(spatial_fms) # torch.Size([2, 96, 1, 1])
        output = channel_wise * spatial_fms # torch.Size([2, 96, 64, 64])
        
        cost_vol = self.Conv1(output)
        cost_vol = self.Conv2(cost_vol)
        cost_vol = self.Conv3(cost_vol)
        if self.prompt:
            prompt = prompt_token @ self.prompt_task.squeeze(2).squeeze(0)  # [B, emb_dim]
            cost_vol = self.prompt_attn(cost_vol, prompt)
        cost_vol = self.Conv31(cost_vol)
        flow = self.Conv5(self.Conv4(cost_vol))

        return flow

class DFF(nn.Module):
    def __init__(self, in_channels=2, list_num=4):
        super(DFF, self).__init__()
        self.channels = in_channels
        self.num = list_num
        self.exp = 16
        self.step = 7
        self.conv_1 = nn.Sequential(conv(self.num * self.channels, self.exp * self.channels, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True)) # num*2 -> 16*2
        self.conv_2 = nn.Sequential(conv(self.exp * self.channels, self.exp * self.channels, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True))# 16*2 -> 16*2

        self.convsig = [ConvSigmoid(self.exp * self.channels, 2, kernel_size=3, stride=1) for _ in range(self.num)]
        self.convsig = nn.Sequential(*self.convsig)

    def forward(self, predict_flows): # torch.Size([2, 2, 32, 32])

        pred_cache = []
        for i, flow in enumerate(predict_flows):
            pred_cache.append(F.interpolate(flow, scale_factor=(2**(self.num-i), 2**(self.num-i)), mode='bilinear', align_corners=True))

        pred_cat = torch.cat(pred_cache, dim=1) # torch.Size([2, 2, 64, 64])
        weights_cat = self.conv_2(self.conv_1(pred_cat)) # torch.Size([2, 32, 64, 64])

        for i, flow in enumerate(pred_cache):
            weight_map = self.convsig[i](weights_cat) # torch.Size([2, 2, 64, 64])
            pred_cache[i] = flow * weight_map
            if i==0:
                progress_field = pred_cache[i] # torch.Size([2, 2, 64, 64])
            else:
                progress_field = progress_field + pred_cache[i] #
        #self.integrate = VecInt([progress_field.shape[2], progress_field.shape[3]], self.step)
        #progress_field = self.integrate(progress_field) # torch.Size([2, 2, 64, 64])

        return progress_field


################  LMA################
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=64):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)

def calc_mean_std(x, eps=1e-5):
    N, C = x.size()[:2]
    x_var = x.view(N, C, -1).var(dim=2) + eps
    x_std = x_var.sqrt().view(N, C, 1, 1)
    x_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return x_mean, x_std

class calc_mean_std_learnable(nn.Module):
    def __init__(self, in_feat, out_feat, eps=1e-5):
        super(calc_mean_std_learnable, self).__init__()
        self.eps = eps
        self.std_mlp = MLP(in_feat, out_feat)
        self.mean_mlp = MLP(in_feat, out_feat)
        self.out_feat = out_feat

    def forward(self, x):
        N, C = x.size()[:2]
        x_var = x.view(N, C, -1).var(dim=2) + self.eps
        x_std = x_var.sqrt()
        x_mean = x.view(N, C, -1).mean(dim=2)

        x_std = self.std_mlp(x_std).view(N, self.out_feat, 1, 1)
        x_mean = self.mean_mlp(x_mean).view(N, self.out_feat, 1, 1)

        return x_mean, x_std

class LearnableDomainAlignment(nn.Module):
    def __init__(self, in_feat, out_feat=1):
        super(LearnableDomainAlignment, self).__init__()
        self.calc_msl = calc_mean_std_learnable(in_feat, out_feat)
        

        self.conv_after_LDA=nn.Conv2d(in_feat, in_feat, kernel_size=1,stride=1, padding=0, bias=True)
    def forward(self, x, y):
        size = x.size()
        y_mean, y_std = self.calc_msl(y)
        x_mean, x_std = calc_mean_std(x)

        x_normalized = (x - x_mean.expand(size)) / x_std.expand(size)
        x_LDA = x_normalized * y_std.expand(size) + y_mean.expand(size)
        x_LDA=x+self.conv_after_LDA(x_LDA)
        return x_LDA

class PointNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(2, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc_fea = nn.Linear(16*16, num_classes)

    def forward(self, fea):
        B, C, H, W = fea.shape
        fea = fea.view(B, C, -1)  # [B, C, L]
        #fea = nn.MaxPool1d(fea.size(-1))(fea).squeeze(-1)
        #print(fea.shape)
        fea = F.relu(self.fc_fea(fea))
        
        
        '''learn the superpoints' features'''
        x = F.relu(self.conv1(fea))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = x.view(-1, 512)
        
        '''classification'''
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x             
class Flow_Net(nn.Module):
    def __init__(self, init_embed=16, in_channels=1, task='VI-IR',num_classes=2):
        super(Flow_Net, self).__init__()
        self.in_channels = in_channels
        self.embed_in = init_embed

        #
        self.spatial_transform_f1 = SpatialTransformer(volsize=[int(d / 8) for d in shape])
        self.spatial_transform_f2 = SpatialTransformer(volsize=[int(d / 4) for d in shape])
        self.spatial_transform_f3 = SpatialTransformer(volsize=[int(d / 2) for d in shape])
        # 
        self.fix_encoder =Feat_Encoder(img_size=shape[0], img_channel=self.in_channels,width=self.embed_in)
        self.mov_encoder =Feat_Encoder(img_size=shape[0], img_channel=self.in_channels,width=self.embed_in)
        
        # 
        self.cross_scan0=Cross_scan(img_size=shape[0]/8,in_channel=self.embed_in*8,embed_dim=self.embed_in*8, depth=1, d_state=16)
        self.cross_scan1=Cross_scan(img_size=shape[0]/4,in_channel=self.embed_in*4,embed_dim=self.embed_in*4, depth=1, d_state=16)
        self.cross_scan2=Cross_scan(img_size=shape[0]/2,in_channel=self.embed_in*2,embed_dim=self.embed_in*2, depth=1, d_state=16)
        self.cross_scan3=Cross_scan(img_size=shape[0],in_channel=self.embed_in,embed_dim=self.embed_in, depth=1, d_state=16)
        # TODO: Bottle neck
        self.task_prompt=PointNet(num_classes=num_classes)
        self.flownet0 = Bottleneck_flow(8 * self.embed_in, 2)
        self.flownet1 = FlowNet(in_channels=4 * self.embed_in, dim=2,task_classes=num_classes)
        self.flownet2 = FlowNet(in_channels=2 * self.embed_in, dim=2,task_classes=num_classes)
        
        if task=='VI-IR(Elastic)':
            self.flownet3 = FlowNet_last(in_channels=self.embed_in, dim=2,prompt=False,task_classes=num_classes)
        else:
            self.flownet3 = FlowNet_last(in_channels=self.embed_in, dim=2,prompt=True,task_classes=num_classes)
        
       
        
        
        self.feature_align0 = LearnableDomainAlignment(8 *self.embed_in,1)
        self.feature_align1 = LearnableDomainAlignment(4 *self.embed_in, 1)
        self.feature_align2 = LearnableDomainAlignment(2 *self.embed_in, 1)
        self.feature_align3 = LearnableDomainAlignment(self.embed_in, 1)
        
        #DFF
        self.DFF_1 = DFF(list_num=1)
        self.DFF_2 = DFF(list_num=2)
        self.DFF_3 = DFF(list_num=3)
        self.DFF_4 = DFF(list_num=4)

   
        
    def forward(self, fix_img, mov_img, shape=shape):
       
        #feature 
        [mov_enc3, mov_enc2, mov_enc1, mov_enc0] = self.mov_encoder(mov_img)
        [fix_enc3, fix_enc2, fix_enc1, fix_enc0] = self.fix_encoder(fix_img)
        #print(mov_enc3.shape, mov_enc2.shape, mov_enc1.shape, mov_enc0.shape)
        #torch.Size([1, 32, 128, 128]) 
        #torch.Size([1, 64, 64, 64]) 
        #torch.Size([1, 128, 32, 32]) 
        #torch.Size([1, 256, 16, 16])

        predict_flows = []
        #===================stage-1
        fix_aligned0 = self.feature_align0(fix_enc0, mov_enc0)  
        fix_aligned0, mov_enc0=self.cross_scan0(fix_aligned0, mov_enc0)
        # predict flow-0
        dec_feat0 = torch.cat([mov_enc0, fix_aligned0], dim=1)
        flow0,up_dec0 =self.flownet0(dec_feat0)
        prompt_token=self.task_prompt(flow0)
        prompt = F.softmax(prompt_token, dim=1)
        
        predict_flows.append(flow0)
        
        phi_1   = self.DFF_1(predict_flows)
        warped_1, _ = self.spatial_transform_f1(mov_enc1, phi_1)
        
        #===================stage-2
        fix_aligned1 = self.feature_align1(fix_enc1, warped_1) 
        fix_aligned1, warped_1=self.cross_scan1(fix_aligned1, warped_1)
        
        # predict flow-1
        flow1,up_dec1 = self.flownet1(up_dec0, warped_1, fix_aligned1)
        predict_flows.append(flow1)
        
        

        phi_2   = self.DFF_2(predict_flows)
        warped_2, _ = self.spatial_transform_f2(mov_enc2, phi_2)
        
        
        #===================stage-3
        fix_aligned2 = self.feature_align2(fix_enc2, warped_2) 
        fix_aligned2, warped_2=self.cross_scan2(fix_aligned2, warped_2)
        # predict flow-2
        flow2,up_dec2 = self.flownet2(up_dec1, warped_2, fix_aligned2)
        predict_flows.append(flow2)

        phi_3   = self.DFF_3(predict_flows)
        warped_3, _ = self.spatial_transform_f3(mov_enc3, phi_3)
        #===================stage-4
        fix_aligned3 = self.feature_align3(fix_enc3, warped_3) 
        fix_aligned3, warped_3=self.cross_scan3(fix_aligned3, warped_3)
        # predict flow-3
        flow3 = self.flownet3(up_dec2, warped_3, fix_aligned3,prompt)
        
        
        predict_flows.append(flow3)
        flow4      = self.DFF_4(predict_flows)
        mov_encs=[mov_enc0, mov_enc1,mov_enc2,mov_enc3]
        fix_aligneds=[fix_aligned0,fix_aligned1,fix_aligned2,fix_aligned3]
        

        return flow4,mov_encs,fix_aligneds,prompt_token



if __name__ == '__main__':
    from fvcore.nn import flop_count,FlopCountAnalysis, parameter_count_table
    model = Flow_Net(init_embed=24,in_channels=1).cuda(8)
    model.eval()
    a = torch.randn(1, 1, 256, 256).cuda(8)
    b = torch.randn(1, 1, 256, 256).cuda(8)
    '''
    tensor = (a,b,)
    flops1 = FlopCountAnalysis(model, tensor)
    flops1=flops1.total()/1e9
    print("FLOPs:",flops1) 
    '''
    flow4,mov_encs,fix_aligneds = model(a,b)
    print(flow4.shape)
    
    
    total_params = sum(p.numel() for p in model.parameters())/ 1e6
    print(f'Total parameters: {total_params}')