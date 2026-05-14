import os
import torch
import model as Model
import argparse
import core.logger as Logger
import os
import numpy as np
from util.util  import  RGB2YCrCb,YCrCb2RGB
from torchvision.transforms import ToTensor
from PIL import Image
from util.img_read_save import img_save
from tqdm import tqdm
import time
import torch.nn.functional as F
import data as Data
from image_fuse_show import combine_show
import cv2
import torch.nn as nn
from Data_generate.utils_flow.pixel_wise_mapping import warp
from Data_generate.utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

import torch.nn.functional as F
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_2D.json',help='JSON file for configuration')
    parser.add_argument('-local_rank', '--local_rank', type=int, default=0)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    #=====================================================================
    #           Parsing command
    #======================================================================
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    resume_path='./checkpoint/NIR_fisher_affine.pth'
    opt['path']['resume_state']=resume_path
        
    for dataset_name in ["fisheye","affine"]:#MCubeS
        print("The test result of " + dataset_name + ' :')
        test_folder = os.path.join('./dataset/test/', dataset_name)
        test_out_folder1 = os.path.join('./Result/reg_result/', dataset_name)
        test_out_folder2 = os.path.join('./Result/fuse_result/',dataset_name)
        Mamba = Model.create_model(opt, args.local_rank)
        device = torch.device(f'cuda' if opt['gpu_ids'] is not None else 'cpu')
        Mamba.Reg_Fus_net.to(device)
        total = sum([param.nelement() for param in Mamba.Reg_Fus_net.parameters()])
        print("Number of parameters: %.2fM" % (total / 1e6))
        Mamba.Reg_Fus_net.eval()
        min_max = (-1, 1)
        start_time=time.time()
        with torch.no_grad():
            for img_name in tqdm(os.listdir(os.path.join(test_folder, "ir_warp"))):
                visible_image = Image.open(os.path.join(test_folder, "vi", img_name)).convert('RGB')
                ir = Image.open(os.path.join(test_folder, "ir", img_name)).convert('RGB')
                ir_warp = Image.open(os.path.join(test_folder, "ir_warp", img_name)).convert('RGB')

                visible_image = (ToTensor()(visible_image) * (min_max[1] - min_max[0]) +min_max[0]).unsqueeze(0).cuda()
                ir_warp = (ToTensor()(ir_warp)* (min_max[1] - min_max[0]) + min_max[0]).unsqueeze(0).cuda()
                ir = (ToTensor()(ir)* (min_max[1] - min_max[0]) + min_max[0]).unsqueeze(0).cuda()
                visible_image_YUV = RGB2YCrCb(visible_image)
                          
                vi=visible_image_YUV[:, 0:1, :, :]
                ir_warp=ir_warp[:, 0:1, :, :]
                ir=ir[:, 0:1, :, :]
                          
                vi_256 = F.interpolate(vi, (256, 256), mode='bilinear', align_corners=False)
                ir_warp_256 = F.interpolate(ir_warp, (256, 256), mode='bilinear', align_corners=False)
                #result
                Reg_result,Fusion_result=Mamba.Reg_Fus_net.Val_Fusion(vi_256, ir_warp_256,vi,ir_warp,device) 
                Fusion_result = YCrCb2RGB(torch.cat((Fusion_result, visible_image_YUV[:, 1:2, :, :], visible_image_YUV[:, 2:3, :, :]), dim=1))
                Fusion_result = Mamba.tensor2fu(Fusion_result, min_max=(0, 1)).astype(np.uint8)
                img_save(Fusion_result, img_name.split(sep='.')[0], test_out_folder2)
                Reg_result = Mamba.tensor2fu(Reg_result, min_max=(0, 1)).astype(np.uint8)
                ir = Mamba.tensor2fu(ir, min_max=(0, 1)).astype(np.uint8)
                fuse_img=combine_show(ir,Reg_result[:,:,0:1])
                  
                img_save(fuse_img, img_name.split(sep='.')[0], test_out_folder1)       
                        
                       
        end_time=time.time()
        print(dataset_name,":",start_time-end_time)
            