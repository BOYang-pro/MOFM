""" Modified from NC-Net. """
import random

import numpy as np
import torch

from utils_data.geometric_transformation_sampling.geometric_distortions import ElasticTransform
from utils_data.geometric_transformation_sampling.aff_homo_tps_generation import (AffineGridGen, HomographyGridGen, TpsGridGen)
from utils_flow.flow_and_mapping_operations import (convert_flow_to_mapping, convert_mapping_to_flow,get_gt_correspondence_mask, unormalise_and_convert_mapping_to_flow)
from utils_flow.pixel_wise_mapping import warp, warp_with_mapping


class CompositionOfFlowCreations:
    """Composes multiple flow creation classes. Will be applied sequentially. """
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, mini_batch, net, training=True, *args, **kwargs):
        flow_gt = None
        for transform in self.transforms_list:
            flow_gt = transform(mini_batch=mini_batch, net=net, training=training, flow=flow_gt)
        return flow_gt
def generate_tps_theta(center_strength_range=(0.1, 0.2),
                       edge_strength_range=(0.05, 0.15),
                       corner_strength_range=(0, 0.05),
                       use_cuda=False):
    # 
    corner_shift = np.random.uniform(-corner_strength_range[1],corner_strength_range[1],size=4)
    #corner_shift_value = np.random.uniform(-corner_strength_range[1], corner_strength_range[1])

    #corner_shift = np.array([corner_shift_value] * 4)
                                      

    # 
    edge_shift = np.random.uniform(-edge_strength_range[1], -edge_strength_range[0], size=4)
    #edge_shift_value = np.random.uniform(-edge_strength_range[1], -edge_strength_range[0])

    #edge_shift = np.array([edge_shift_value] * 4)
    
    # 
    center_shift = -np.random.uniform(center_strength_range[0],
                                      center_strength_range[1])


    # 
    base_y = np.array([-1, 0, 1,
                       -1, 0, 1,
                       -1, 0, 1], dtype=np.float32)

    base_x = np.array([-1, -1, -1,
                        0,  0,  0,
                        1,  1,  1], dtype=np.float32)

    # 
    new_x = base_x.copy()
    new_y = base_y.copy()

    # 
    corner_ids = [0, 2, 6, 8]
    for i, cid in enumerate(corner_ids):
        new_x[cid] += corner_shift[i] * (1 if base_x[cid] >= 0 else -1)
        new_y[cid] += corner_shift[i] * (1 if base_x[cid] >= 0 else -1)

    # 
    edge_ids = [1, 3, 5, 7]
    for i, eid in enumerate(edge_ids):
        direction_x = -np.sign(base_x[eid])
        direction_y = -np.sign(base_y[eid])
        new_x[eid] += edge_shift[i] * direction_x
        new_y[eid] += edge_shift[i] * direction_x

    # 
    new_x[4] += center_shift
    new_y[4] += center_shift

    
    theta = np.concatenate([new_x, new_y], axis=0).astype(np.float32)

    
    theta = torch.tensor(theta).unsqueeze(0)
    if use_cuda:
        theta = theta.cuda()

    return theta
    
class SynthecticAffHomoTPSTransfo:
    """Generates a flow field of given load_size, corresponding to a randomly sampled Affine, Homography, TPS or Affine-TPS
    transformation. """
    def __init__(self, size_output_flow=(512, 512), 
                        random_t=0.3, 
                        random_s=0.3, 
                        random_alpha=np.pi / 36,
                        random_t_hom=0.3, 
                        tps_center_value=(0.1, 0.2),
                        tps_edge_value=(0.05, 0.15),
                        tps_corner_value=(0, 0.05),
                        tps_grid_size=3, 
                        tps_reg_factor=0,
                        transformation_types=None, proba_horizontal_flip=0.0, use_cuda=True):
        """
        For all transformation parameters, image is taken as in interval [-1, 1]. Therefore all parameters must be
        within [0, 1]. The range of sampling is then [-parameter, parameter] or [1-parameter, 1+parameter] for the
        scale.
        Args:
            size_output_flow: desired output load_size for generated flow field
            random_t: max translation for affine transform.
            random_s: max scale for affine transform
            random_alpha: max rotation and shearing angle for the affine transform
            random_t_tps_for_afftps: max translation parameter for the tps transform generation, used for the
                         affine-tps transforms
            random_t_hom: max translation parameter for the homography transform generation
            random_t_tps: max translation parameter for the tps transform generation
            tps_grid_size: tps grid load_size
            tps_reg_factor:
            transformation_types: list of transformations to samples.
                                  Must be selected from ['affine', 'hom', 'tps', 'afftps']
            parametrize_with_gaussian: sampling distribution for the transformation parameters. Gaussian ? otherwise,
                                       uses a uniform distribution
            use_cuda: use_cuda?
        """

        if not isinstance(size_output_flow, (tuple, list)):
            size_output_flow = (size_output_flow, size_output_flow)
        self.out_h, self.out_w = size_output_flow

        self.proba_horizontal_flip = proba_horizontal_flip


        # for homo
        self.random_t_hom = random_t_hom

        # for affine
        self.random_t = random_t
        self.random_alpha = random_alpha
        self.random_s = random_s
        
        #for tps
        self.tps_center_value=tps_center_value
        self.tps_edge_value=tps_center_value
        self.tps_corner_value=tps_corner_value
        

        self.use_cuda = use_cuda
        if transformation_types is None:
            transformation_types = ['affine', 'hom', 'tps']
        self.transformation_types = transformation_types
        if 'hom' in self.transformation_types:
            self.homo_grid_sample = HomographyGridGen(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda)
        if 'affine' in self.transformation_types:
            self.aff_grid_sample = AffineGridGen(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda)
        if 'tps' in self.transformation_types:
            self.tps_grid_sample = TpsGridGen(out_h=self.out_h, out_w=self.out_w, grid_size=tps_grid_size,
                                              reg_factor=tps_reg_factor, use_cuda=use_cuda)

    def __call__(self, *args, **kwargs):
        """Generates a flow_field (flow_gt) from sampling a geometric transformation. """

        geometric_model = self.transformation_types[random.randrange(0, len(self.transformation_types))]
        # sample the theta
        theta_hom, theta_aff = 0.0, 0.0
        if geometric_model == 'affine':
            
            rot_angle =(np.random.rand(1) - 0.5) * 2 * self.random_alpha
            
            sh_angle =[0]# (np.random.rand(1) - 0.5) * 2 * self.random_alpha
            
            lambda_1 =[1]# 1 + (2 * np.random.rand(1) - 1) * self.random_s)  # between 0.75 and 1.25 for random_s = 0.25
            lambda_2 =[1]# 1 + (2 * np.random.rand(1) - 1) * self.random_s  # between 0.75 and 1.25
            #
            tx = (2 * np.random.rand(1) - 1) * self.random_t  # between -0.25 and 0.25 for random_t=0.25
            ty = (2 * np.random.rand(1) - 1) * self.random_t

            R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                                 [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
            R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                                    [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

            D = np.diag([lambda_1[0], lambda_2[0]])

            A = R_alpha @ R_sh.transpose() @ D @ R_sh

            theta_aff = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])
            theta_aff = torch.Tensor(theta_aff.astype(np.float32)).unsqueeze(0)
            theta_aff = theta_aff.cuda() if self.use_cuda else theta_aff
        if geometric_model == 'hom':
            theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
            theta_hom = theta_hom + (np.random.rand(8) - 0.5) * 2 * self.random_t_hom
            theta_hom = torch.Tensor(theta_hom.astype(np.float32)).unsqueeze(0)
            theta_hom = theta_hom.cuda() if self.use_cuda else theta_hom
        if geometric_model == 'tps':
            theta_tps = generate_tps_theta(center_strength_range=self.tps_center_value,
                                           edge_strength_range=self.tps_edge_value,
                                           corner_strength_range=self.tps_corner_value) 
        if geometric_model == 'hom':
            mapping = self.homo_grid_sample.forward(theta_hom)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 1, 2,h,w
        elif geometric_model == 'affine':
            mapping = self.aff_grid_sample.forward(theta_aff)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2,h,w
        elif geometric_model == 'tps':
            mapping = self.tps_grid_sample.forward(theta_tps)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2,h,w
        else:
            raise NotImplementedError

        # 1, 2, h, w
        if random.random() < self.proba_horizontal_flip:
            mapping_gt = convert_flow_to_mapping(flow_gt)
            mapping_gt = torch.from_numpy(np.copy(
                np.fliplr(mapping_gt.squeeze().permute(1, 2, 0).cpu().numpy()))).permute(2, 0, 1).unsqueeze(0)
            flow_gt = convert_mapping_to_flow(mapping_gt).cuda()
        return flow_gt


class AddElasticTransforms:
    """Generates batched dense elastic deformation fields of a given load_size. Optionally compose them with existing
    batched dense flow fields. The elastic deformation field is only applied in random regions of random sizes. """
    def __init__(self, size_output_flow, max_nbr_perturbations=13, min_nbr_perturbations=5,
                 elastic_parameters=None, max_sigma_mask=40, min_sigma_mask=10):
        """
        The elastic deformation field is applied only in small regions (each called a perturbation). More specifically,
        we create the residual flow by first generating an elastic deformation motion field on a dense grid
        of dimension given by size_output_flow. Since we only want to include elastic perturbations
        in multiple small regions, we generate binary masks, each delimiting the area on which to apply one
        local perturbation.  The final elastic flow is the sum of all masked elastic flow fields.

        The masks should be between 0 and 1 and offer a smooth transition between the two, so that the perturbations
        appear smoothly. To create each mask, we thus generate a 2D Gaussian centered at a random location
        and with a random standard deviation sampled between min_sigma_max and max_sigma_mask,
        on a dense grid of dimension size_output_flow. It is then scaled to 2.0 and clipped to 1.0,
        to obtain smooth regions equal to 1.0 where the perturbations will be applied, and transition
        regions on all sides from 1.0 to 0.0.
        Args:
            settings:
            size_output_flow: desired output load_size for generated flow field
            max_nbr_perturbations:
            min_nbr_perturbations:
            elastic_parameters: parameters for the deformation field generation
            max_sigma_mask: max sigma for creating the mask, where each perturbation is applied.
            min_sigma_mask: min sigma for creating the mask, where each perturbation is applied.
        """

        default_elastic_parameters = {"max_sigma": 0.08, "min_sigma": 0.1, "min_alpha": 1, "max_alpha": 1.0}
        

        self.max_nbr_perturbations = max_nbr_perturbations
        self.min_nbr_perturbations = min_nbr_perturbations
        self.elastic_parameters = default_elastic_parameters

        if elastic_parameters is not None:
            self.elastic_parameters.update(elastic_parameters)

        self.max_sigma_mask = max_sigma_mask  # can vary this load_size if we dont want small transformations
        self.min_sigma_mask = min_sigma_mask
        self.ElasticTrans = ElasticTransform(self.elastic_parameters, get_flow=True, approximate=True)

        if not isinstance(size_output_flow, (tuple, list)):
            size_output_flow = (size_output_flow, size_output_flow)
        self.size_output_flow = size_output_flow

    @staticmethod
    def get_gaussian(shape, mu, sigma):
            x = np.indices(shape)
            mu = np.float32(mu).reshape(2, 1, 1)
            n = sigma * np.sqrt(2 * np.pi) ** len(x)
            return np.exp(-0.5 * (((x - mu) / sigma) ** 2).sum(0)) / n

    def __call__(self, batch, device,flow=None):
        """Takes batched tensor and generates batched tensor flow fields.
        For each batch dimension, generates a random masked elastic flow field.
        If existing batch of flow field is provided, creates and returns composition of original flow and
        masked elastic flow. """

        if flow is None:
            # no previous flow, it will be only elastic transformations
            flow = np.zeros((batch, self.size_output_flow[0], self.size_output_flow[1], 2))
        else:
            # if it is torch, convert it to numpy
            if not isinstance(flow, np.ndarray):
                flow = flow.cpu().numpy()

        assert len(flow.shape) == 4  #
        # it is supposed to be batch of flow fields

        mapping = convert_flow_to_mapping(flow, output_channel_first=False)  # b, h, w, 2
        mask_valid_correspondences = get_gt_correspondence_mask(flow)  # b, h, w
        shape = mapping.shape[1:-1]

        flow_perturbations_ = []
        for b_ in range(mapping.shape[0]):
            # for each image in the batch
            nbr_perturbations = random.randint(self.min_nbr_perturbations, self.max_nbr_perturbations)

            # sample parameters of elastic transform
            sigma_, alpha = self.ElasticTrans.get_random_paremeters(shape, seed=None)

            # get the elastic transformation
            flow_x_pertu, flow_y_pertu = self.ElasticTrans.get_mapping_from_distorted_image_to_undistorted(shape, sigma_, alpha)
            flow_pertu = np.dstack((flow_x_pertu, flow_y_pertu))
            mask_final = np.zeros(shape, np.float32)

            # make the mask
            for i in range(nbr_perturbations):
                sigma = random.randint(self.min_sigma_mask, self.max_sigma_mask)
                coordinate_in_mask=False
                while coordinate_in_mask is False:
                    start=0 + sigma * 3
                    end=shape[1] - sigma * 3
                    end2=shape[0] - sigma * 3
                    if start>end:
                        x = random.randint(end, start)
                    else:
                        x = random.randint(start, end)
                    if start>end2:
                        y = random.randint(end, start)
                    else:
                        y = random.randint(start, end)
                    if mask_valid_correspondences[b_, y, x]:
                        coordinate_in_mask = True

                mask = self.get_gaussian(shape, mu=[x, y], sigma=sigma)

                max = mask.max()
                mask = np.clip(2.0 / max * mask, 0.0, 1.0)
                mask_final = mask_final + mask

            mask = np.clip(mask_final, 0.0, 1.0)
            # estimation final perturbation, shape is h,w,2
            flow_pertu = flow_pertu * np.tile(np.expand_dims(mask, axis=2), (1, 1, 2))
            flow_perturbations_.append(np.expand_dims(flow_pertu, axis=0))
        flow_pertu = np.concatenate(flow_perturbations_, axis=0)

        # get final composition
        final_mapping = warp(torch.Tensor(mapping).permute(0, 3, 1, 2),
                             torch.Tensor(flow_pertu).permute(0, 3, 1, 2))
        flow = convert_mapping_to_flow(final_mapping, output_channel_first=True).to(device)
        return flow



