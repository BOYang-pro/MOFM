'''create dataset and dataloader'''
import logging
import torch.utils.data
import torchvision.transforms as transforms

#================================================================
#                          get dataset 
#=================================================================

def create_dataset_Fusion(dataset_opt, phase,opt):

        
    if opt['model']['task']=='VI-IR':           #
        path1=dataset_opt['dataroot_ir_hom']
        path2=dataset_opt['dataroot_ir_tps']
           
    if opt['model']['task']=='VI-NIR':           #
        path1=dataset_opt['dataroot_nir_affine']
        path2=dataset_opt['dataroot_nir_fisher'] 
        
        
    if opt['model']['task']=='Med':           #
        path1=dataset_opt['dataroot_med_pet']
        path2=dataset_opt['dataroot_med_ct']
        
            
    if phase=='train':
        from data.load_pre_made_dataset import WarpingDataset                                         
        train_dataset = WarpingDataset( image_root1=path1,
                                        image_root2=path2,
                                        crop_size=dataset_opt['crop_size'], 
                                        output_size=dataset_opt['crop_size']
                                        )
        
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(train_dataset.__class__.__name__,dataset_opt['name']))
    return train_dataset
