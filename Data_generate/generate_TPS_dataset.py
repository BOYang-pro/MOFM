import os

import torch
import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from imageio import imsave
import random
def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)
def get_coordinate(shape, det_uv):
    b, _, w, h = shape
    uv_d = np.zeros([w, h, 2], np.float32)

    for i in range(0, w):
        for j in range(0, h):
            uv_d[i, j, 0] = j
            uv_d[i, j, 1] = i

    uv_d = np.expand_dims(uv_d.swapaxes(2, 1).swapaxes(1, 0), 0)
    uv_d = torch.from_numpy(uv_d).cuda()
    uv_d = uv_d.repeat(b, 1, 1, 1)

    det_uv = uv_d + det_uv.to(uv_d.device)

    return det_uv

def uniform(shape, img_uv):
    b, _, w, h = shape
    x0 = (w - 1) / 2. 

    img_nor = (img_uv - x0)/x0             
    img_nor = img_nor.permute(0, 2, 3, 1)
    return img_nor

def resample_image(feature, flow):
    img_uv = get_coordinate(feature.shape, flow)
    grid = uniform(feature.shape, img_uv)
    target_image = F.grid_sample(feature, grid)
    return target_image

def get_coordinate_xy(shape, det_uv):
    b, _, h, w = shape
    uv_d = np.zeros([h, w, 2], np.float32)

    for j in range(0, h):
        for i in range(0, w):
            uv_d[j, i, 0] = i
            uv_d[j, i, 1] = j

    uv_d = np.expand_dims(uv_d.swapaxes(2, 1).swapaxes(1, 0), 0)
    uv_d = torch.from_numpy(uv_d).cuda()
    uv_d = uv_d.repeat(b, 1, 1, 1)
    det_uv = uv_d + det_uv
    return det_uv

def uniform_xy(shape, uv):
    b, _, h, w = shape
    y0 = (h - 1) / 2.
    x0 = (w - 1) / 2.

    nor = uv.clone()
    nor[:, 0, :, :] = (uv[:, 0, :, :] - x0) / x0 
    nor[:, 1, :, :] = (uv[:, 1, :, :] - y0) / y0
    nor = nor.permute(0, 2, 3, 1)  # b w h 2

    return nor

def resample_image_xy(feature, flow):
    uv = get_coordinate_xy(feature.shape, flow)
    grid = uniform_xy(feature.shape, uv)
    target_image = F.grid_sample(feature, grid)
    return target_image
    
def tensor2fu(image_tensor, imtype=np.float32, min_max=(-1, 1)):
    image_numpy = image_tensor[:1, :, :, :].squeeze(0).clamp_(-1, 1).float().cpu().numpy()
    image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  

    nc, nh, nw = image_numpy.shape

    if nc == 1:
        tmp = np.zeros((nh, nw, 1))
        tmp = image_numpy.transpose(1, 2, 0)
        tmp = np.tile(tmp, (1, 1, 3))
        image_numpy = tmp
    elif nc == 3:
        tmp = np.zeros((nh, nw, 3))
        tmp = image_numpy.transpose(1, 2, 0)
        image_numpy = tmp
    image_numpy -= np.amin(image_numpy)
    image_numpy = (image_numpy / np.amax(image_numpy))
    #image_numpy = (image_numpy /2.0)
    image_numpy = image_numpy * 255.0
    return image_numpy.astype(imtype)


def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' has been created.")
    else:
        print(f"Directory '{dir_path}' already exists.")      
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate warped datasets")
    parser.add_argument("--image_root", type=str, required=True, help="Path to input images")
    parser.add_argument("--save_root", type=str, required=True, help="Path to save generated images")
    parser.add_argument("--image_size", type=int, default=512, help="Output image size")
    args = parser.parse_args()
    image_root=args.image_root
    save_root=args.save_root
    image_size=args.image_size
    dataset_name=['M3FD_test']
    for dataset in dataset_name:
        test_folder = os.path.join(image_root)
        image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
        num_images = len([f for f in os.listdir(os.path.join(image_root,'ir')) if f.lower().endswith(image_extensions)])
        
        
        test_out_folder1= os.path.join(save_root,'vi_warp')
        test_out_folder2 = os.path.join(save_root,'ir_warp')
        test_out_folder3= os.path.join(save_root,'vi')
        test_out_folder4 = os.path.join(save_root,'ir')
        ensure_directory_exists(test_out_folder1)
        ensure_directory_exists(test_out_folder2)
        ensure_directory_exists(test_out_folder3)
        ensure_directory_exists(test_out_folder4)
        min_max=(0, 1)

        
        for img_name in tqdm(os.listdir(os.path.join(test_folder, "ir"))):
            if num_images == 0:
                raise ValueError(f"No images found in folder: {test_folder}")
            #If the image exceeds 500, the deformation type will be repeated
            if num_images <= 500:
                num = random.randint(0, num_images - 1)
                flow_name = f'./TPS/flow/{num}.npy'
                num2 = 1#random.uniform(1, 2)   #add  random perturbation
            else:
                num = random.randint(0, 499)
                flow_name = f'./TPS/flow/{num}.npy'
                num2 = 1#random.uniform(1, 2) 
            print(img_name)
            infrared_image = Image.open(os.path.join(test_folder, "ir", img_name)).convert('RGB')
            ir = (ToTensor()(infrared_image) * (min_max[1] - min_max[0]) + min_max[0]).unsqueeze(0)
            visble_image = Image.open(os.path.join(test_folder, "vi", img_name)).convert('RGB')
            vi = (ToTensor()(visble_image) * (min_max[1] - min_max[0]) + min_max[0]).unsqueeze(0)
        
            vi=vi.cuda()
            ir=ir.cuda()
            
            
            flow_gt=np.load(flow_name)
            flow_gt = torch.tensor(flow_gt)
            flow_gt = F.interpolate(flow_gt, size=(image_size, image_size), mode='bilinear', align_corners=False)
            
            #print(flow_gt.shape)
            
            #flow_gt = flow_gt.permute(2, 0, 1).unsqueeze(0)
            vi_warp = resample_image(vi, -flow_gt*num2)
            ir_warp = resample_image(ir, -flow_gt*num2)
            
            vi_warp = tensor2fu(vi_warp, min_max=(-1, 1)).astype(np.uint8)
            img_save(vi_warp, img_name.split(sep='.')[0], test_out_folder1)
            ir_warp = tensor2fu(ir_warp, min_max=(-1, 1)).astype(np.uint8)
            img_save(ir_warp, img_name.split(sep='.')[0], test_out_folder2)
            
            #vi_gt = tensor2fu(vi, min_max=(-1, 1)).astype(np.uint8)
            #img_save(vi_gt, img_name.split(sep='.')[0], test_out_folder3)
            #ir_gt = tensor2fu(ir, min_max=(-1, 1)).astype(np.uint8)
            #img_save(ir_gt, img_name.split(sep='.')[0], test_out_folder4)
            
            
