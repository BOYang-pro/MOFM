# Copyright (c) Phigent Robotics. All rights reserved.

import os
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import model.networks as networks
from model.base_model import BaseModel
from collections import OrderedDict
from util.util import RGB2YCrCb,YCrCb2RGB
import core.logger as Logger
from image_fuse_show import combine_show
class Multistage_Reg_Fus_Model(BaseModel):
    def __init__(self,opt,local_rank):
        super(Multistage_Reg_Fus_Model, self).__init__(opt)
        # define network
        self.Reg_Fus_net = networks.define_Network(opt,local_rank)
        self.local_rank=local_rank
        self.schedule_phase = None
        self.centered = opt['datasets']['centered']

        # set loss and load resume state
        self.set_loss()

        if self.opt['phase'] == 'train':
            train_opt = self.opt['train']
            if isinstance(self.Reg_Fus_net, nn.parallel.DistributedDataParallel):
                self.Reg_Fus_net.module.reg_model.train()
                self.Reg_Fus_net.module.Fusion_model.train()
                
                fusion_optim_params = list(self.Reg_Fus_net.module.Fusion_model.parameters())
                Reg_optim_params = list(self.Reg_Fus_net.module.reg_model.parameters())
            # set optim
            self.fusion_optG = torch.optim.AdamW(fusion_optim_params, lr=train_opt["optimizer"]["lr"], betas=(0.9, 0.999),weight_decay=train_opt["optimizer"]["weight_decay"])
            self.Reg_optG = torch.optim.AdamW(Reg_optim_params, lr=train_opt["optimizer"]["lr"], betas=(0.9, 0.999),weight_decay=train_opt["optimizer"]["weight_decay"])
            self.optimizers.append(self.fusion_optG)
            self.optimizers.append(self.Reg_optG)
        
            
            # set learn schedulers
            self.setup_schedulers()
            self.log_dict_stage1 = OrderedDict()
            self.log_dict_stage2 = OrderedDict()
        self.load_network()
        
        
    def feed_data(self, data):
        for key in data:
            data[key] = data[key].cuda(self.local_rank)
        self.data =data
        #self.data = self.set_device(data)
        
    def optimize_parameters_stage1(self,flag_iter):
    
        self.fusion_optG.zero_grad()  
        self.Reg_optG.zero_grad() 
           
        if isinstance(self.Reg_Fus_net, nn.parallel.DistributedDataParallel):
            self.Reg_Fus_net.module.Fusion_model.eval()
            for param in self.Reg_Fus_net.module.Fusion_model.parameters():
                param.requires_grad = False
            self.Reg_Fus_net.module.reg_model.train()
            for param in self.Reg_Fus_net.module.reg_model.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
        
            output = self.Reg_Fus_net.module.forward_stage1(self.data['vis'], 
                                                            self.data['ir_warp'],
                                                            self.data['ir'], 
                                                            self.data['vis_warp'], 
                                                            self.data['task_id'], 
                                                            flag_iter)
        
        # reg result
        self.warped_mov = output['warped_mov']
        self.warped_fix = output['warped_fix']
        
        # reg loss
        loss_ssim = output['loss_ssim']
        loss_grad = output['loss_grad']
        loss_ncc  = output['loss_ncc']
        loss_L1   = output['loss_L1']   
        loss_classify= output['loss_classify']  
        if flag_iter==1:
            loss_gt  = output['loss_gt']
        loss      = output['loss']
        
        reduce_loss_ssim=self.reduce_tensor(loss_ssim.data)
        reduce_loss_grad=self.reduce_tensor(loss_grad.data)
        reduce_loss_ncc =self.reduce_tensor(loss_ncc.data)
        reduce_loss_L1  =self.reduce_tensor(loss_L1.data)
        loss_classify_L1  =self.reduce_tensor(loss_classify.data)
        if flag_iter==1:
            reduce_loss_gt  =self.reduce_tensor(loss_gt.data)
        #All loss
        reduce_loss    =self.reduce_tensor(loss.data)
        # 
        loss.backward()
        self.Reg_optG.step()
        
        # Set log 
        self.log_dict_stage1['l_ssim'] = reduce_loss_ssim.item()
        self.log_dict_stage1['l_grad'] = reduce_loss_grad.item()
        self.log_dict_stage1['lncc'] = reduce_loss_ncc.item()
        self.log_dict_stage1['l_1'] = reduce_loss_L1.item()
        self.log_dict_stage1['l_cls'] = loss_classify_L1.item()
        if flag_iter==1:
            self.log_dict_stage1['l_gt'] = reduce_loss_gt.item()
        #All loss
        self.log_dict_stage1['l_tot'] = reduce_loss.item()
        
        
    def optimize_parameters_stage2(self):
    
        self.fusion_optG.zero_grad()  
        self.Reg_optG.zero_grad()   
        if isinstance(self.Reg_Fus_net, nn.parallel.DistributedDataParallel):
            self.Reg_Fus_net.module.reg_model.eval()
            for param in self.Reg_Fus_net.module.reg_model.parameters():
                param.requires_grad = False
            self.Reg_Fus_net.module.Fusion_model.train()
            for param in self.Reg_Fus_net.module.Fusion_model.parameters():
                param.requires_grad = True
            output = self.Reg_Fus_net.module(self.data['vis'], self.data['ir_warp'])
        #fused result
        self.fusion_result = output['fusion_result']
        self.warped_mov = output['warped_mov']
        
        #fusion loss
        fusion_loss = output['fusion_loss']
        loss_gradient = output['loss_gradient']
        loss_l1 = output['loss_l1']
        loss_SSIM = output['loss_SSIM']
        loss = output['loss']

        reduce_fusion_loss=self.reduce_tensor(fusion_loss.data)
        reduce_loss_gradient=self.reduce_tensor(loss_gradient.data)
        reduce_loss_l1=self.reduce_tensor(loss_l1.data)
        reduce_loss_SSIM=self.reduce_tensor(loss_SSIM.data)
        
        
        #all loss
        reduce_loss=self.reduce_tensor(loss.data)
        
        loss.backward()   
        
        self.fusion_optG.step()
        
        self.log_dict_stage2['l_fu'] = reduce_fusion_loss.item()
        self.log_dict_stage2['l_g'] = reduce_loss_gradient.item()
        self.log_dict_stage2['l_1'] = reduce_loss_l1.item()     
        self.log_dict_stage2['l_ssim'] = reduce_loss_SSIM.item()
        self.log_dict_stage2['l_tot'] = reduce_loss.item()

    def reduce_tensor(self, tensor: torch.Tensor):
        
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
        return rt
        
    
    #     test reg result
    def test_stage1(self):
        self.Reg_Fus_net.eval()
        input = torch.cat([self.data['vis'], self.data['ir_warp']], dim=1)
        if isinstance(self.Reg_Fus_net, nn.parallel.DistributedDataParallel):
            self.warped_mov,self.warped_fix= self.Reg_Fus_net.module.test_stage1(input, self.device)
        else:
            self.warped_mov,self.fusion_result= self.Reg_Fus_net.test_stage1(input, self.device)
            self.warped_fix = YCrCb2RGB(torch.cat((self.warped_fix, self.data['fuse_VU'][:, 0:1, :, :], self.data['fuse_VU'][:, 1:2, :, :]), dim=1))
        self.Reg_Fus_net.train()
    # test fusion result
    def test_stage2(self):
        self.Reg_Fus_net.eval()
        input = torch.cat([self.data['vis'], self.data['ir_warp']], dim=1)
        if isinstance(self.Reg_Fus_net, nn.parallel.DistributedDataParallel):
            self.fusion_result,self.warped_mov= self.Reg_Fus_net.module.test_stage2(input, self.device)
            visible_image_YUV = RGB2YCrCb(self.data['vis_rgb'])
            self.fusion_result = YCrCb2RGB(torch.cat((self.fusion_result, visible_image_YUV[:, 1:2, :, :], visible_image_YUV[:, 2:3, :, :]), dim=1))
        else:
            self.fusion_result,self.warped_mov= self.Reg_Fus_net.test_stage2(input, self.device)
        self.Reg_Fus_net.train()   
    

    #
    def set_loss(self):
        if isinstance(self.Reg_Fus_net, nn.parallel.DistributedDataParallel):
            self.Reg_Fus_net.module.set_loss(self.device)
        else:
            self.Reg_Fus_net.set_loss(self.device)

    def get_current_log_stage1(self):
        return self.log_dict_stage1
        
        
    def get_current_log_stage2(self):
        return self.log_dict_stage2
        
        
        
    def get_current_visuals_stage1(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['vis'] = self.tensor2im(self.data['vis_rgb'], min_max=(0, 1))
        out_dict['ir'] = self.tensor2im(self.data['ir'], min_max=(0, 1))
        out_dict['ir_warp'] = self.tensor2im(self.data['ir_warp'], min_max=(0, 1))
        warped_mov = self.tensor2fu(self.warped_mov, min_max=(0, 1))
        out_dict['warped_fix'] = self.tensor2fu(self.warped_fix, min_max=(0, 1))
        out_dict['warped_mov'] = warped_mov
        out_dict['img_show'] = combine_show(self.tensor2fu(self.data['ir'], min_max=(0, 1)), warped_mov[:,:,0:1])
        return out_dict

    def get_current_test_stage1(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)

        out_dict['vis'] = self.tensor2im(self.data['vis_rgb'], min_max=(0, 1))
        out_dict['ir'] = self.tensor2im(self.data['ir'], min_max=(0, 1))
        out_dict['ir_warp'] = self.tensor2im(self.data['ir_warp'], min_max=(0, 1))
        warped_mov = self.tensor2fu(self.warped_mov, min_max=(0, 1))
        out_dict['warped_fix'] = self.tensor2fu(self.warped_fix, min_max=(0, 1))
        out_dict['warped_mov'] = warped_mov
        out_dict['img_show'] = combine_show(self.tensor2fu(self.data['ir'], min_max=(0, 1)), warped_mov[:,:,0:1])
        return out_dict
        
    def get_current_visuals_stage2(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['vis'] = self.tensor2im(self.data['vis_rgb'], min_max=(0, 1))
        out_dict['ir_warp'] = self.tensor2im(self.data['ir_warp'], min_max=(0, 1))
        out_dict['ir'] = self.tensor2im(self.data['ir'], min_max=(0, 1))
        warped_mov = self.tensor2fu(self.warped_mov, min_max=(0, 1))
        out_dict['warped_mov'] = warped_mov
        out_dict['img_show'] = combine_show(self.tensor2fu(self.data['ir'], min_max=(0, 1)), warped_mov[:,:,0:1])
        out_dict['fusion_result'] = self.tensor2fu(self.fusion_result, min_max=(0, 1))
        return out_dict

    def get_current_test_stage2(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)

        out_dict['vis'] = self.tensor2im(self.data['vis_rgb'], min_max=(0, 1))
        out_dict['ir_warp'] = self.tensor2im(self.data['ir_warp'], min_max=(0, 1))
        out_dict['ir'] = self.tensor2im(self.data['ir'], min_max=(0, 1))
        warped_mov = self.tensor2fu(self.warped_mov, min_max=(0, 1))
        out_dict['warped_mov'] = warped_mov
        out_dict['img_show'] = combine_show(self.tensor2fu(self.data['ir'], min_max=(0, 1)), warped_mov[:,:,0:1])
        out_dict['fusion_result'] = self.tensor2fu(self.fusion_result, min_max=(0, 1))
        return out_dict
        
    def tensor2im(self, image_tensor, imtype=np.float32, min_max=(-1, 1)):
        # (1, 3, 224, 224)===>(3, 224, 224)
        image_numpy = image_tensor[:1, :, :, :].squeeze(0).detach().clamp_(-1, 1).float().cpu().numpy()
        image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        nc, nh, nw = image_numpy.shape

        if nc == 1:
            tmp = np.zeros((nh, nw, 1))
            tmp = image_numpy.transpose(1, 2, 0)
            tmp = np.tile(tmp, (1, 1, 3))
            image_numpy = tmp
        elif nc == 3:
            tmp = np.zeros((nh, nw, 3))
            # 1 channel -> 3 channel
            tmp = image_numpy.transpose(1, 2, 0)
            image_numpy = tmp

        image_numpy -= np.amin(image_numpy)
        #image_numpy = (image_numpy / np.amax(image_numpy))
        image_numpy = (image_numpy /2.0)

        image_numpy = image_numpy * 255.0
        return image_numpy.astype(imtype)

    def tensor2fu(self, image_tensor, imtype=np.float32, min_max=(-1, 1)):
        # (1, 3, 224, 224)===>(3, 224, 224)
        image_numpy = image_tensor[:1, :, :, :].squeeze(0).detach().clamp_(-1, 1).float().cpu().numpy()
        image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  

        nc, nh, nw = image_numpy.shape

        if nc == 1:
            tmp = np.zeros((nh, nw, 1))
            tmp = image_numpy.transpose(1, 2, 0)
            tmp = np.tile(tmp, (1, 1, 3))
            image_numpy = tmp
        elif nc == 3:
            tmp = np.zeros((nh, nw, 3))
            # 1 channel -> 3 channel
            tmp = image_numpy.transpose(1, 2, 0)
            image_numpy = tmp
        image_numpy -= np.amin(image_numpy)
        #image_numpy = (image_numpy / np.amax(image_numpy))
        image_numpy = (image_numpy /2.0)

        image_numpy = image_numpy * 255.0
        return image_numpy.astype(imtype)

    def save_network(self, epoch, iter_step):
        genG_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_gen_G.pth'.format(iter_step, epoch))
        opt_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.Reg_Fus_net
        if isinstance(self.Reg_Fus_net, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, genG_path)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer1': None, 'optimizer2': None}
        opt_state['optimizer1'] = self.fusion_optG.state_dict()
        opt_state['optimizer2'] = self.Reg_optG.state_dict()
        
        torch.save(opt_state, opt_path)

    def load_network(self):
        
        load_path = self.opt['path']['resume_state']

            
            
        if load_path is not None:
            print(load_path)
            genG_path = load_path

            #opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.Reg_Fus_net
            if isinstance(self.Reg_Fus_net, nn.parallel.DistributedDataParallel):
                network = network.module
            #print(torch.load(genG_path))
            #network.load_state_dict(torch.load(genG_path), strict=(not self.opt['model']['finetune_norm']))
            network.load_state_dict(torch.load(genG_path), strict=False)
        




