#coding:gbk
import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
import cv2
from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
from utils_flow.pixel_wise_mapping import warp
import os
from utils_data.image_transforms import ArrayToTensor
import torchvision.transforms as transforms
from utils_data.geometric_transformation_sampling.synthetic_warps_sampling import SynthecticAffHomoTPSTransfo,AddElasticTransforms
import torch.utils.data as data
import torchvision as tv



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
    def __init__(self, 
                      image_root, 
                      synthetic_flow_generator, 
                      elastic_flow_generator,
                      compute_mask_zero_borders=False,
                      min_percent_valid_corr=0.1, 
                      crop_size=256, 
                      output_size=256, 
                      image_transform=None):
   
        self.min_max=(-1,1)
        self.image_root = image_root
        self.apply_mask_zero_borders = compute_mask_zero_borders


        self.synthetic_flow_generator = synthetic_flow_generator
        self.elastic_flow_generator=elastic_flow_generator
        if not isinstance(crop_size, (tuple, list)):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        if not isinstance(output_size, (tuple, list)):
            output_size = (output_size, output_size)
            
        self.output_size = output_size
        self.min_percent_valid_corr = min_percent_valid_corr

        # processing of final images
        self.image_transform = image_transform
        
        self.data_list = self.build_list()
        self.img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    def __len__(self):
        return len(self.data_list)

    #========================
    #     Root directory for generated images
    #========================
    def build_list(self):
        image_root = os.path.abspath(self.image_root)
        vi_path = os.path.join(image_root, 'vi')
        ir_path = os.path.join(image_root, 'ir')
        samples = []
        self.length = len(os.listdir(vi_path))
        for file_name_ext in os.listdir(vi_path):
            file_name = os.path.basename(file_name_ext)
            vi_paths = fit_img_postfix(os.path.join(vi_path, file_name))
            ir_paths = fit_img_postfix(os.path.join(ir_path, file_name))
            samples.append((vi_paths, ir_paths))
        return samples
        
    def read_img(self, image_path):
        img = cv2.imread(image_path)
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
        else:
            img = img[:, :, ::-1].copy()  # go from BGR to RGB
        img = self.img_transforms(img)   
        return img 
         
    def __getitem__(self, idx):
        
          
        vi_path, ir_path = self.data_list[idx]       
        #read image name
        img_name = os.path.basename(vi_path)
        
        #read image
        vi = self.read_img(vi_path).unsqueeze(0).float()
        ir = self.read_img(ir_path).unsqueeze(0).float()
        
        vi = F.interpolate(vi, (self.output_size[0], self.output_size[0]), mode='bilinear', align_corners=False)
        ir = F.interpolate(ir, (self.output_size[0], self.output_size[0]), mode='bilinear', align_corners=False)
        
      
        # take original images, have them at resolution 520x520
        b, _, h, w = vi.shape
      
        # get synthetic homography transformation from the synthetic flow generator
        if self.elastic_flow_generator is not None:
            flow_gt = self.elastic_flow_generator(b, vi.device, flow=None).detach()
        elif self.synthetic_flow_generator is not None:
            flow_gt = self.synthetic_flow_generator(b, vi.device).detach()
        else:
            raise ValueError("No flow generator provided!")
        
        flow_gt.requires_grad_(False)
        bs, _, h_f, w_f = flow_gt.shape
        if h_f != h or w_f != w:
            # reshape and rescale the flow so it has the load_size of the images
            flow_gt = F.interpolate(flow_gt, (h, w), mode='bilinear', align_corners=False)
            flow_gt[:, 0] *= float(w) / float(w_f)
            flow_gt[:, 1] *= float(h) / float(h_f)
        
        #get warp image  ir: [0~255]  flow_gt: [-255~255]
        ir_warp, mask_zero_borders = warp(ir, flow_gt, padding_mode='zeros', return_mask=True)
        vi_warp, mask_zero_borders = warp(vi, flow_gt, padding_mode='zeros', return_mask=True)
        
        ir_warp = ir_warp.byte()
        vi_warp=vi_warp.byte()
        
        #Crop a central patch from the image and the real flow field so that the black borders are removed.
        x_start = w // 2 - self.crop_size[1] // 2
        y_start = h // 2 - self.crop_size[0] // 2
        vi_resized = vi[:, :, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]
        ir_resized = ir[:, :, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]
        
        vi_warp_resized = vi_warp[:, :, y_start: y_start + self.crop_size[0],x_start: x_start + self.crop_size[1]]
        ir_warp_resized = ir_warp[:, :, y_start: y_start + self.crop_size[0],x_start: x_start + self.crop_size[1]]
        
        flow_gt_resized = flow_gt[:, :, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]
        mask_zero_borders = mask_zero_borders[:, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]

        # To prevent cropping from deleting all public areas
        if self.output_size != self.crop_size:
            vi_resized = F.interpolate(vi_resized, self.output_size,mode='area')
            ir_resized = F.interpolate(ir_resized, self.output_size,mode='area')
            
            ir_warp_resized = F.interpolate(ir_warp_resized, self.output_size,mode='area')
            vi_warp_resized = F.interpolate(vi_warp_resized, self.output_size,mode='area')
            
            flow_gt_resized = F.interpolate(flow_gt_resized, self.output_size,mode='bilinear', align_corners=False)
            flow_gt_resized[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
            flow_gt_resized[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

            mask_zero_borders = F.interpolate(mask_zero_borders.float().unsqueeze(1), self.output_size,
                                              mode='bilinear', align_corners=False).byte().squeeze(1)
            mask_zero_borders = mask_zero_borders.bool() 
        #===========
        # put back the images and flow to numpy array, channel last
        # TO DO: change, this is not great
        vi_resized = vi_resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        ir_resized = ir_resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        ir_warp_resized = ir_warp_resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        vi_warp_resized = vi_warp_resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        flow_gt_resized = flow_gt_resized.squeeze(0).permute(1, 2, 0).numpy()
        mask_zero_borders = mask_zero_borders.squeeze(0).numpy()


        #Create a real-valued mask (at least for evaluation)
        mask_gt = get_gt_correspondence_mask(flow_gt_resized)

        if self.apply_mask_zero_borders:
            if mask_gt.sum() < mask_gt.shape[-1] * mask_gt.shape[-2] * self.min_percent_valid_corr:
                mask_zero_borders = mask_gt
            else:
                mask_gt *= mask_zero_borders  # also removes the black area from the valid mask
        
        #Data Augmentation: Grayscale, Blur
        if self.image_transform!=None:
            vi_resized = self.image_transform(vi_resized)
            ir_resized = self.image_transform(ir_resized)
            
            ir_warp_resized = self.image_transform(ir_warp_resized)
            vi_warp_resized = self.image_transform(vi_warp_resized)
          
        
        
        #np->tensor,[0,255]==>[0,1]==>[-1,1]
        vi_resized=self.img_transforms(vi_resized).div(255.0).mul(2.0).sub(1.0)
        ir_resized=self.img_transforms(ir_resized).div(255.0).mul(2.0).sub(1.0)
        
        ir_warp_resized=self.img_transforms(ir_warp_resized).div(255.0).mul(2.0).sub(1.0)
        vi_warp_resized=self.img_transforms(vi_warp_resized).div(255.0).mul(2.0).sub(1.0)
        flow_gt_resized=self.img_transforms(flow_gt_resized)
        
        

        #[2,h,w] 
        if flow_gt_resized.shape[0] != 2:
            flow_gt_resized = flow_gt_resized.permute(2, 0, 1)  
            
        #tensor[-255,255]==>[-1,1]
        _, H, W = flow_gt_resized.shape
        flow_gt_resized[0,:,:] /= W
        flow_gt_resized[1,:,:] /= H 
        flow_gt_resized = torch.clamp(flow_gt_resized, min=-1.0, max=1.0)
        output = {'vi': vi_resized,
                  'ir_warp': ir_warp_resized,
                  'vi_warp': vi_warp_resized,
                  'ir': ir_resized,
                  'flow_gt': -flow_gt_resized,
                  'correspondence_mask': mask_gt,
                  'img_name':img_name
                  }
        if self.apply_mask_zero_borders:
            output['mask_zero_borders'] = mask_zero_borders
        return output
        
        
def writeFlow(flow, name_to_save, save_dir):
    name=os.path.join(save_dir, name_to_save)
    f = open(name, 'wb')
    magic=202021.25
    np.array([magic], dtype=np.float32).tofile(f)
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f) 
    
def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' has been created.")
    else:
        print(f"Directory '{dir_path}' already exists.")      
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate warped datasets")
    parser.add_argument("--image_root", type=str, required=True, help="Path to input images")
    parser.add_argument("--save_root", type=str, required=True, help="Path to save generated images")
    parser.add_argument("--image_size", type=int, default=512, help="Output image size")
    parser.add_argument("--transform_type", type=str, default="elastic", choices=["affine","hom","tps","elastic"], help="Type of deformation")
    args = parser.parse_args()
    image_root=args.image_root
    save_root=args.save_root
    ensure_directory_exists(save_root+'vi/')
    ensure_directory_exists(save_root + 'vi_warp/')
    ensure_directory_exists(save_root + 'ir/')
    ensure_directory_exists(save_root + 'ir_warp/')
    #ensure_directory_exists(save_root + 'flow_gt/')
    #
    image_size=args.image_size
    transform_type=args.transform_type
    #Affine control parameters
    random_t=0.3                #The bigger, the more twisted
    random_s=0.3                #The bigger, the more twisted
    
    #Hom control parameters
    random_t_hom=0.2            #The bigger, the more twisted
    
    #Tps control parameters
    tps_center_value=(0.1, 0.2) # Deformation intensity of the four center 
    tps_edge_value=(0.05, 0.15) # Deformation intensity of the four edge 
    tps_corner_value=(0, 0.05)  # Deformation intensity of the four corners 
    
    #Elastic control parameters
    #sigma: Smoothness Intensity:   The smaller, the more twisted
    #alpha: deformation strength:   The bigger, the more twisted
    default_elastic_parameters = {"max_sigma": 0.048, 
                                  "min_sigma": 0.046, 
                                  "min_alpha": 1.05, 
                                  "max_alpha": 1.15} 
    if transform_type!='elastic':
        synthetic_flow_generator = SynthecticAffHomoTPSTransfo(
                size_output_flow=image_size, 
                random_t=random_t, 
                random_s=random_s,
                use_cuda=False,
                random_alpha=np.pi / 36, 
                random_t_hom=random_t_hom, 
                tps_center_value=tps_center_value,
                tps_edge_value=tps_edge_value,
                tps_corner_value=tps_corner_value,
                transformation_types=[transform_type])
        elastic_flow_generator=None
    else:
        synthetic_flow_generator=None           
        elastic_flow_generator =AddElasticTransforms(
                                        size_output_flow=image_size,
                                        max_nbr_perturbations=70, 
                                        min_nbr_perturbations=50,
                                        elastic_parameters=default_elastic_parameters, 
                                        max_sigma_mask=70, 
                                        min_sigma_mask=30) 
                                                 
    train_dataset = WarpingDataset(image_root=image_root,
                                    synthetic_flow_generator=synthetic_flow_generator,
                                    elastic_flow_generator=elastic_flow_generator,
                                    compute_mask_zero_borders=True,
                                    crop_size=image_size, 
                                    output_size=image_size,
                                    image_transform=None
                                    )
    dl_iter = data.DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
    
    for batch in dl_iter:
        vi_warp = (batch['vi_warp']+1)/2
        ir_warp = (batch['ir_warp']+1)/2
        vi = (batch['vi']+1)/2
        ir = (batch['ir']+1)/2
        #flow_map = batch['flow_gt']
        img_name=batch['img_name'][0]
        #pif_show(vi,ir_warp,flow_map)
        print(img_name)
        tv.utils.save_image(vi_warp, save_root+'vi_warp/'+img_name)
        tv.utils.save_image(ir_warp, save_root+'ir_warp/'+img_name)
        tv.utils.save_image(ir, save_root + 'ir/' + img_name)
        tv.utils.save_image(vi, save_root + 'vi/' + img_name)
        '''
        base, _ = os.path.splitext(img_name)
        name_flow = base + '.flo'
        flow_gt = flow_map[0].detach().permute(1, 2, 0).cpu().numpy()  # now shape is HxWx2
        writeFlow(flow_gt, name_flow, save_root + 'flow_gt/')
        '''
        
    