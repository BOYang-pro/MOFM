# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from torch import nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from model.head.loss import Fusion_loss
from model.head.reg_losses import LossFunction_Dense
from model.backbone.Reg_mamba import Flow_Net
from model.backbone.Fusion_mamba  import Fusion_frarework
from Data_generate.utils_flow.pixel_wise_mapping import warp 

class Fusionmamba_Framework(BaseModule):
    def __init__(
            self,
            opt,
            img_size=256,
            in_channel=1,
            embed_dim=48,
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
        #Reg model
        self.reg_model=Flow_Net(init_embed=embed_dim,in_channels=1,task=opt['model']['task'])
        
        #Fusion model
        
        self.Fusion_model=Fusion_frarework(img_size=img_size, 
                               in_channel=in_channel,
                               embed_dim=embed_dim,
                               patch_size=patch_size,
                               depth=depth,
                               d_state=d_state,
                               ssm_ratio=ssm_ratio,
                               norm_layer=norm_layer, 
                               drop_path=drop_path,
                               attn_drop_rate=attn_drop_rate,
                               mlp_ratio=mlp_ratio,
                               drop=drop,
                               dt_rank=dt_rank,
                               ape=ape,
                               )
                         
        
    def set_loss(self,device):
        #
        self.fusion_loss = Fusion_loss().to(device)
        self.mse_loss =nn.MSELoss().to(device)
        #
        self.Reg_loss = LossFunction_Dense().to(device)
        self.task_classify = nn.CrossEntropyLoss()
    
        
    def Val_Reg(self,x_vis,x_ir,ir_ori,device):
        with torch.no_grad():
            flow,mov_encs,fix_aligneds,_= self.reg_model(x_vis, x_ir)
            flow = torch.clamp(flow, min=-1., max=1.)
            _,_, H, W = flow.shape
            flow[:,0:1,:,:] *= W
            flow[:,1:2,:,:] *= H
            if H != 512 or W != 512:
                   
                flow = F.interpolate(flow, (512, 512), mode='bilinear', align_corners=True)
                flow[:,0,:,:] *= float(512) / float(W)
                flow[:,1,:,:] *= float(512) / float(H)
            warped_mov, _= warp((ir_ori+1)/2*255, flow, padding_mode='zeros', return_mask=True)
            warped_mov=warped_mov/127.5-1.0     
        return warped_mov,mov_encs,fix_aligneds
        
    def Val_Fusion(self,x_vis,x_ir,vis_ori,ir_ori,device):
        with torch.no_grad():
            flow,_,_,_= self.reg_model(x_vis, x_ir)
            flow = torch.clamp(flow, min=-1., max=1.)
            _,_, H, W = flow.shape
            flow[:,0:1,:,:] *= W
            flow[:,1:2,:,:] *= H
            
            if H != 512 or W != 512:
                flow = F.interpolate(flow, (512, 512), mode='bilinear', align_corners=True)
                flow[:,0,:,:] *= float(512) / float(W)
                flow[:,1,:,:] *= float(512) / float(H)
            
            warped_mov, _= warp((ir_ori+1)/2*255, flow, padding_mode='zeros', return_mask=True)
            warped_mov=warped_mov/127.5-1.0     
           
            fusion_result= self.Fusion_model(vis_ori,warped_mov)  
           
        return warped_mov,fusion_result
            
    def Val_Fusion_med(self,x_vis,x_ir,device):
        with torch.no_grad():
            flow,_,_,_ = self.reg_model(x_vis, x_ir)
            flow = torch.clamp(flow, min=-1., max=1.)
            _,_, H, W = flow.shape
            flow[:,0:1,:,:] *= W
            flow[:,1:2,:,:] *= H
            warped_mov, _= warp((x_ir+1)/2*255, flow, padding_mode='zeros', return_mask=True)
            warped_mov=warped_mov/127.5-1.0     

            fusion_result= self.Fusion_model(x_vis,warped_mov)  
            
        return warped_mov,fusion_result
    
    def test_stage1(self,x_in,device):  
        x_vis = x_in[:, :1]
        x_ir = x_in[:, 1:2]
    
        with torch.no_grad():
            flow,_,_,_ = self.reg_model(x_vis, x_ir)
            flow = torch.clamp(flow, min=-1., max=1.)
       
            _,_, H, W = flow.shape
            flow[:,0:1,:,:] *= W
            flow[:,1:2,:,:] *= H
            warped_mov, _= warp((x_ir+1)/2*255, flow, padding_mode='zeros', return_mask=True)
            warped_fix, _= warp((x_vis+1)/2*255, -flow, padding_mode='zeros', return_mask=True)
            warped_mov=warped_mov/127.5-1.0
            warped_fix=warped_fix/127.5-1.0
        return warped_mov,warped_fix
        
    def test_stage2(self,x_in,device):  
        x_vis = x_in[:, :1]
        x_ir = x_in[:, 1:]
        with torch.no_grad():
            flow,_,_,_= self.reg_model(x_vis, x_ir)
            flow = torch.clamp(flow, min=-1., max=1.)
       
            _,_, H, W = flow.shape
            flow[:,0:1,:,:] *= W
            flow[:,1:2,:,:] *= H
            warped_mov, _= warp((x_ir+1)/2*255, flow, padding_mode='zeros', return_mask=True)
            
            warped_mov=warped_mov/127.5-1.0
            
            fusion_result= self.Fusion_model(x_vis,warped_mov) 
        return fusion_result,warped_mov
      
    def forward_stage1(self, x_vis, x_ir,ir, vis_warp,prompt_gt,flag_iter):
        batch_size = x_vis.shape[0]
        device = x_vis.device
        dtype=x_vis.dtype
        
        flow,mov_encs,fix_aligneds,prompt= self.reg_model(x_vis, x_ir)
        flow = torch.clamp(flow, min=-1., max=1.)
       
        _,_, H, W = flow.shape

        flow[:,0:1,:,:] *= W
        flow[:,1:2,:,:] *= H
        
        warped_mov, _= warp((x_ir+1)/2*255, flow, padding_mode='zeros', return_mask=True)
        warped_fix, _= warp((x_vis+1)/2*255, -flow, padding_mode='zeros', return_mask=True)
        warped_mov=warped_mov/127.5-1.0
        warped_fix=warped_fix/127.5-1.0
        loss_reg_all, loss_ssim, loss_L1,loss_grad,loss_ncc=self.Reg_loss(warped_mov, warped_fix, ir, vis_warp,mov_encs,fix_aligneds,flow)
        loss_classify=0.1*self.task_classify(prompt,prompt_gt)
        if flag_iter==1:
            with torch.no_grad(): 
                fusion_result_predict= self.Fusion_model(x_vis,warped_mov) 
                fusion_result_gt= self.Fusion_model(x_vis,ir) 
            loss_gt=self.mse_loss(fusion_result_predict,fusion_result_gt)     
            total_loss = loss_reg_all+loss_gt+loss_classify
            output = {'warped_mov': warped_mov,
                      'warped_fix': warped_fix,
                      'loss_ssim':loss_ssim,
                      'loss_grad':loss_grad, 
                      'loss_classify':loss_classify,
                      'loss_ncc':loss_ncc,
                      'loss_gt':loss_gt,
                      'loss_L1':loss_L1, 
                      'loss':total_loss
                      }
        else:
            total_loss = loss_reg_all+loss_classify
            
            output = {'warped_mov': warped_mov,
                      'warped_fix': warped_fix,
                      'loss_ssim':loss_ssim,
                      'loss_grad':loss_grad,
                      'loss_classify':loss_classify, 
                      'loss_ncc':loss_ncc,
                      'loss_L1':loss_L1, 
                      'loss':total_loss
                      }
        return output
   
              
    def forward(self, x_vis, x_ir):
        batch_size = x_vis.shape[0]
        device = x_vis.device
        dtype=x_vis.dtype
        with torch.no_grad():
            flow,_,_,_ = self.reg_model(x_vis, x_ir)
            flow = torch.clamp(flow, min=-1., max=1.)
            _,_, H, W = flow.shape
            flow[:,0:1,:,:] *= W
            flow[:,1:2,:,:] *= H

        warped_mov, _= warp((x_ir+1)/2*255, flow, padding_mode='zeros', return_mask=True)
        warped_mov=warped_mov/127.5-1.0
            
        fusion_result= self.Fusion_model(x_vis,warped_mov) 
        fusion_loss, loss_gradient, loss_l1, loss_SSIM=self.fusion_loss(x_vis,warped_mov,fusion_result)
        total_loss = fusion_loss 
        output = {
                  'fusion_result': fusion_result,
                  'warped_mov':warped_mov,
                  'fusion_loss':fusion_loss,
                  'loss_gradient':loss_gradient, 
                  'loss_l1':loss_l1, 
                  'loss_SSIM':loss_SSIM, 
                  'loss':total_loss
                  }
        return output
        




if __name__ == '__main__':


    test_sample1 = torch.randn(1, 1, 256, 256)
    test_sample2 = torch.randn(1, 1, 256, 256)
    

    print(test_sample1.shape)
