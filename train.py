import os

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time
import random
from util.visualizer import Visualizer
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist



def adjust_learning_rate(opt,optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt['train']["optimizer"]["lr"] * (0.5 ** (epoch //15))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train_2D.json',help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--task', type=str, default=None, help='set task')
    parser.add_argument('--batch_size', type=int, default=None, help='set batch_size')
    parser.add_argument('--img_size', type=int, default=None, help='set img size')
    

    #=====================================================================
    #           Step 1: Analyze the command line and enable printing
    #======================================================================
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    opt['model']['task']=args.task
    opt['datasets']['train']['batch_size']=args.batch_size
    opt['model']['feat']['img_size']=args.img_size
    visualizer = Visualizer(opt)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    ##################################################
    #              Initialize distributed computing
    #################################################
    gpus=len(opt['gpu_ids'])
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    dist.init_process_group(backend='nccl',world_size=gpus,rank=local_rank)
    torch.cuda.set_device(local_rank)
    ##################################################
    #              read dataset
    #################################################
    phase = 'train'
    dataset_opt=opt['datasets']['train']
    batchSize = dataset_opt['batch_size']
    train_set = Data.create_dataset_Fusion(dataset_opt, phase,opt)
    if phase == 'train':
        train_sampler=DistributedSampler(train_set,shuffle=True)  
        train_loader =torch.utils.data.DataLoader(
            train_set,
            batch_size=batchSize,
            shuffle=dataset_opt['use_shuffle'],               
            sampler=train_sampler,
            num_workers=dataset_opt['num_workers'],           
            pin_memory=True,                                   
        )
    training_iters = int(ceil(train_set.length / float(batchSize*gpus)))
    
    original_list=opt['train']['scheduler']['periods']
    result_list = [x * training_iters for x in original_list]
    opt['train']['scheduler']['periods']=result_list
    if local_rank==0:
        logger.info('Initial Dataset Finished')
    
    seed = random.randint(0, 10000)
    if local_rank == 0:
        print("current random seed: ", seed)
    torch.cuda.manual_seed_all(seed)
   
    Mamba = Model.create_model(opt,local_rank)
    if local_rank == 0:
        logger.info('Initial Model Finished')
   
    total = sum([param.nelement() for param in Mamba.Reg_Fus_net.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    ################################################################
    ###                         train                            ###
    ################################################################
    current_step = Mamba.begin_step
    start_epoch = Mamba.begin_epoch
    n_epoch = opt['train']['n_epoch']
    epoch_gap=opt['train']['epoch_gap']
    epoch_gap2=opt['train']['epoch_gap2']
    flag_iter=0
    if opt['path']['resume_state']:
        if local_rank == 0:
            print('Resuming training from epoch: {}, iter: {}.'.format(start_epoch, current_step))

    for current_epoch in range (start_epoch,n_epoch):
        
        train_sampler.set_epoch(current_epoch)     # To ensure that the data obtained by each GPU  is random
        for istep, train_data in enumerate(train_loader):
            if current_epoch < epoch_gap or current_epoch >= epoch_gap2: #Phase I
                if current_epoch >= epoch_gap2:
                    flag_iter=1
                iter_start_time = time.time()
                current_step += 1
                Mamba.feed_data(train_data)
                Mamba.optimize_parameters_stage1(flag_iter)
                Mamba.update_learning_rate(current_epoch*train_set.length+istep*batchSize, warmup_iter=opt['train'].get('warmup_iter', -1))
    
                #          Print log
                if local_rank == 0:
                    if (istep+1) % opt['train']['print_freq'] == 0:
                        logs = Mamba.get_current_log_stage1()
                        t = (time.time() - iter_start_time) / batchSize
                        lr_log=Mamba.get_current_learning_rate()
                        visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, lr_log[0], 'Train')
                        visuals = Mamba.get_current_visuals_stage1()
                        visualizer.display_current_results(visuals, current_epoch, True)

                #                val
                #
                    if (istep+1) % opt['train']['val_freq'] == 0:
                        Mamba.test_stage1()
                        visuals = Mamba.get_current_test_stage1()
                        visualizer.display_current_results(visuals, current_epoch, True)
            else: #Phase II
                iter_start_time = time.time()
                current_step += 1
                Mamba.feed_data(train_data)
                Mamba.optimize_parameters_stage2()
                Mamba.update_learning_rate(current_epoch*train_set.length+istep*batchSize, warmup_iter=opt['train'].get('warmup_iter', -1))
    
                #          Print log
                if local_rank == 0:
                    if (istep+1) % opt['train']['print_freq'] == 0:
                        logs = Mamba.get_current_log_stage2()
                        t = (time.time() - iter_start_time) / batchSize
                        lr_log=Mamba.get_current_learning_rate()
                        visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, lr_log[0], 'Train')
                        visuals = Mamba.get_current_visuals_stage2()
                        visualizer.display_current_results(visuals, current_epoch, True)
    
                #              val    
                    if (istep+1) % opt['train']['val_freq'] == 0:
                        
                        Mamba.test_stage2()
                        visuals = Mamba.get_current_test_stage2()
                        visualizer.display_current_results(visuals, current_epoch, True)
        #save network
        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            if local_rank==0:
                Mamba.save_network(current_epoch, current_step)
    dist.destroy_process_group()  #