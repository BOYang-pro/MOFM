import numpy as np
import torch
import torch.nn.functional as F
import cv2
from Data_generate.utils_flow.pixel_wise_mapping import warp
import os
from Data_generate.utils_data.image_transforms import ArrayToTensor, PILToNumpy, ToTensor, RandomBlur
import torchvision.transforms as transforms

from util.util  import  RGB2YCrCb
def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path
    
def norm_image(source_img, mean_vector=[0.485, 0.456, 0.406],
                             std_vector=[0.229, 0.224, 0.225]):
    mean = torch.as_tensor(mean_vector, dtype=source_img.dtype, device=source_img.device)
    std = torch.as_tensor(std_vector, dtype=source_img.dtype, device=source_img.device)
    source_img.sub_(mean[:, None, None]).div_(std[:, None, None])
    return source_img
    

            
class WarpingDataset(torch.utils.data.Dataset):
    """Dataset applying random warps to single images to obtain pairs of matching images and their corresponding
    ground truth flow fields. The final image pair is composed of the central crops of the desired dimensions in both
    images.
    """

    def __init__(self, 
                      image_root1, 
                      image_root2, 
                      crop_size=256, 
                      output_size=256 
                      ):
   
        self.min_max=(-1,1)
        self.image_root1 = image_root1
        self.image_root2 = image_root2
        if not isinstance(crop_size, (tuple, list)):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        if not isinstance(output_size, (tuple, list)):
            output_size = (output_size, output_size)
            
        self.output_size = output_size

        
        self.data_list = self.build_list()
        self.img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    def __len__(self):
        return len(self.data_list)

    #========================
    #       The root directory of the generated image
    #========================
    def build_list(self):
        def collect_samples_from_root(root_path,task_id):
            vi_path = os.path.join(root_path, 'vi')
            ir_path = os.path.join(root_path, 'ir')
            vi_warp_path = os.path.join(root_path, 'vi_warp')
            ir_warp_path = os.path.join(root_path, 'ir_warp')
            samples = []
            for file_name_ext in os.listdir(vi_path):
                file_name = os.path.basename(file_name_ext)
                vi_p = fit_img_postfix(os.path.join(vi_path, file_name))
                ir_p = fit_img_postfix(os.path.join(ir_path, file_name))
                vi_warp_p = fit_img_postfix(os.path.join(vi_warp_path, file_name))
                ir_warp_p = fit_img_postfix(os.path.join(ir_warp_path, file_name))
                samples.append((vi_p, ir_p, vi_warp_p, ir_warp_p,task_id))
            return samples
        samples1 = collect_samples_from_root(self.image_root1,0)
        samples2 = collect_samples_from_root(self.image_root2,1)
        
        self.length = len(samples1) + len(samples2)
        return samples1 + samples2

        
    def read_img(self, image_path):
        img = cv2.imread(image_path)
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
        else:
            img = img[:, :, ::-1].copy()  # go from BGR to RGB
        img = self.img_transforms(img)   
        return img 
         
    def __getitem__(self, idx):
        
          
        vi_path, ir_path,vi_warp_path,ir_warp_path,task_id = self.data_list[idx]       #
        #Get image name
        img_name = os.path.basename(vi_path)
        
        #read image 
        vi_512 = self.read_img(vi_path).unsqueeze(0).float()
        ir_512 = self.read_img(ir_path).unsqueeze(0).float()
        vi_warp_512 = self.read_img(vi_warp_path).unsqueeze(0).float()
        ir_warp_512 = self.read_img(ir_warp_path).unsqueeze(0).float()

        vi = F.interpolate(vi_512, (self.output_size[0], self.output_size[0]), mode='bilinear', align_corners=False)
        ir = F.interpolate(ir_512, (self.output_size[0], self.output_size[0]), mode='bilinear', align_corners=False)
        vi_warp = F.interpolate(vi_warp_512, (self.output_size[0], self.output_size[0]), mode='bilinear', align_corners=False)
        ir_warp = F.interpolate(ir_warp_512, (self.output_size[0], self.output_size[0]), mode='bilinear', align_corners=False)
        
        vis_YUV=RGB2YCrCb(vi)
        ir_YUV=RGB2YCrCb(ir)
        vis_warp_YUV=RGB2YCrCb(vi_warp)
        ir_warp_YUV=RGB2YCrCb(ir_warp)
        
        vis_YUV_512=RGB2YCrCb(vi_512)
        ir_YUV_512=RGB2YCrCb(ir_512)
        vis_warp_YUV_512=RGB2YCrCb(vi_warp_512)
        ir_warp_YUV_512=RGB2YCrCb(ir_warp_512)
        
        
        #256
        vi_resized = vi.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_resized = ir.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        vi_warp_resized = vi_warp.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_warp_resized = ir_warp.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        vi_YUV_resized = vis_YUV.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_YUV_resized = ir_YUV.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        vi_warp_YUV_resized = vis_warp_YUV.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_warp_YUV_resized = ir_warp_YUV.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        #512
        vi_512_resized = vi_512.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_512_resized = ir_512.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        vi_warp_512_resized = vi_warp_512.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_warp_512_resized = ir_warp_512.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        vi_YUV_512_resized = vis_YUV_512.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_YUV_512_resized = ir_YUV_512.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        vi_warp_YUV_512_resized = vis_warp_YUV_512.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_warp_YUV_512_resized = ir_warp_YUV_512.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        
        #flow_gt_resized = flow_gt.squeeze(0).permute(1, 2, 0).numpy()

        
        #tensor[0,255]==>[0,1]==>[-1,1]
        vi_resized=self.img_transforms(vi_resized).div(255.0).mul(2.0).sub(1.0)
        ir_resized=self.img_transforms(ir_resized).div(255.0).mul(2.0).sub(1.0)
        
        ir_warp_resized=self.img_transforms(ir_warp_resized).div(255.0).mul(2.0).sub(1.0)
        vi_warp_resized=self.img_transforms(vi_warp_resized).div(255.0).mul(2.0).sub(1.0)
        vi_YUV_resized=self.img_transforms(vi_YUV_resized).div(255.0).mul(2.0).sub(1.0)
        ir_YUV_resized=self.img_transforms(ir_YUV_resized).div(255.0).mul(2.0).sub(1.0)
        
        ir_warp_YUV_resized=self.img_transforms(ir_warp_YUV_resized).div(255.0).mul(2.0).sub(1.0)
        vi_warp_YUV_resized=self.img_transforms(vi_warp_YUV_resized).div(255.0).mul(2.0).sub(1.0)
        
        
        #512
        vi_512_resized=self.img_transforms(vi_512_resized).div(255.0).mul(2.0).sub(1.0)
        ir_512_resized=self.img_transforms(ir_512_resized).div(255.0).mul(2.0).sub(1.0)
        
        ir_warp_512_resized=self.img_transforms(ir_warp_512_resized).div(255.0).mul(2.0).sub(1.0)
        vi_warp_512_resized=self.img_transforms(vi_warp_512_resized).div(255.0).mul(2.0).sub(1.0)
        vi_YUV_512_resized=self.img_transforms(vi_YUV_512_resized).div(255.0).mul(2.0).sub(1.0)
        ir_YUV_512_resized=self.img_transforms(ir_YUV_512_resized).div(255.0).mul(2.0).sub(1.0)
        
        ir_warp_YUV_512_resized=self.img_transforms(ir_warp_YUV_512_resized).div(255.0).mul(2.0).sub(1.0)
        vi_warp_YUV_512_resized=self.img_transforms(vi_warp_YUV_512_resized).div(255.0).mul(2.0).sub(1.0)
        
        task_id_tensor = torch.tensor(task_id, dtype=torch.int64)  
        output = {'vis': vi_YUV_resized[0:1,:,:],
                  'fuse_VU': vi_YUV_resized[1:3,:,:],
                  'ir': ir_YUV_resized[0:1,:,:],
                  'ir_warp': ir_warp_YUV_resized[0:1,:,:],
                  'vis_warp': vi_warp_YUV_resized[0:1,:,:],
                  'vis_512': vi_YUV_512_resized[0:1,:,:],
                  'ir_512': ir_YUV_512_resized[0:1,:,:],
                  'ir_warp_512': ir_warp_YUV_512_resized[0:1,:,:],
                  'vis_warp_512': vi_warp_YUV_512_resized[0:1,:,:],
                  'vis_rgb': vi_resized.squeeze(0),
                  'task_id':task_id_tensor,
                  #'flow_gt':  -flow_gt_resized,
                  }
        return output
        
        
        
