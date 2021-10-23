"""
Standalone validation file for the depth-completion-gan.
In order to invoke type:

python validate.py --gpus=0,1,2,3 --batch_size=8 --residual_blocks=17 --checkpoint_model=./logdir/train_test/saved_models/ -n val_test

1. The checkpoint model path has to have 2 files named generator_best.pth and discriminator_best.pth
2. -n --> give a name to the run
3. Modify the val dataloader path with appropriate data directory
4. Typically the directory has the following structure
   ----|->data.ShapeNetDepth|
                            |->train|
                                    |->image_lr
                                    |->image_hr
                                    |->meta_info.txt
                            |->val|
                                  |->image_lr
                                  |->image_hr
                                  |->meta_info.txt
                            |->sample|
                                     |->image_lr
                                     |->image_hr
                                     |->meta_info.txt
5. The image_hr and image_lr are the folder containing dense and sparse depth respectively
6. The meta_info.txt contains the file names of these folders. Refer to misc/ folder for sample meta_info file
7. The folder "sample" contains a few sparse samples. This is to track the model learning visually. 
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
import time

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

LOGDIR = "./logdir/"

def getOpt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ShapeNetSparseDepth", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=192, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--residual_blocks", type=int, default=17, help="number of residual blocks in the generator")
    parser.add_argument("--validation_interval", type=int, default=12, help="interval between two consecutive validations")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    parser.add_argument("--gpus", metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--meta_info_file', '-m', metavar='DIR', default="meta_info.txt", help='Meta file name')
    parser.add_argument("--checkpoint_model_path", type=str, required=True, help="Path to checkpoint model")

    return parser.parse_args()

def validate(generator, discriminator, opt, Tensor, val_dataloader, criterion_GAN, criterion_content, criterion_pixel, logger, val_image_save_path, writer, batches_done=0):
    
    total_val_batches = len(val_dataloader)
    
    # batch_to_be_saved = random.randint(0,total_val_batches)
    batch_to_be_saved = [25, 61, 124, 143] #it can be any numbers
    
    val_sample_path = os.path.join(val_image_save_path,"%06d"%batches_done)
    os.makedirs(val_sample_path, exist_ok=True)
    
    loss_dict = {'rmse':[],'mae':[],'irmse':[],'imae':[]}
    
    for i, imgs in enumerate(val_dataloader):
        
        # number of times the validation has been invoked
        val_n = batches_done//opt.validation_interval - 1
        iteration = val_n*total_val_batches + i
        
        # this will add channel axis: (4, 192, 256) --> (4, 1, 192, 256)
        lr_temp = torch.unsqueeze(imgs["lr"], 1)
        hr_temp = torch.unsqueeze(imgs["hr"], 1)
        
        # Configure model input
        imgs_lr = Variable(lr_temp.type(Tensor))
        imgs_hr = Variable(hr_temp.type(Tensor))
        
        # send equal batch partitions to differnt gpus
        imgs_lr, imgs_hr_nm = imgs_lr.to('cuda'), imgs_hr.to('cuda')
        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)
        
        gen_hr = generator(imgs_lr)
        
        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        writer.add_scalar("GAN_Loss/Validation", loss_GAN, iteration)

        # Content loss 
        # gen_features = feature_extractor(gen_hr)
        # real_features = feature_extractor(imgs_hr).detach()
        
        gen_features = imgrad_yx(gen_hr)
        real_features = imgrad_yx(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)
        writer.add_scalar("Content_Loss/Validation", loss_content, iteration)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)
        writer.add_scalar("Pixel_Loss/Validation", loss_pixel, iteration)
        
        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
        writer.add_scalar("Generator_Loss/Validation", loss_G, iteration)
        
        #new loss measures
        loss_rmse = rmse(gen_hr, imgs_hr)
        loss_dict['rmse'].append(loss_rmse.item())
        writer.add_scalar("RMSE/Validation", loss_rmse.item(), iteration)
        
        loss_mae = mae(gen_hr, imgs_hr)
        loss_dict['mae'].append(loss_mae.item())
        writer.add_scalar("MAE/Validation", loss_mae.item(), iteration)
        
        loss_irmse = irmse(gen_hr, imgs_hr)
        loss_dict['irmse'].append(loss_irmse.item())
        writer.add_scalar("iRMSE/Validation", loss_irmse.item(), iteration)
        
        loss_imae = imae(gen_hr, imgs_hr)
        loss_dict['imae'].append(loss_imae.item())
        writer.add_scalar("iMAE/Validation", loss_imae.item(), iteration)
        
        logger.info(
                "Validating [Batch %d/%d] [content: %f, pixel: %f, RMSE: %f, MAE: %f, iRMSE: %f, iMAE: %f]" #removed content loss
                % (
                    i+1,
                    len(val_dataloader),
                    loss_content.item(), # No content loss
                    loss_pixel.item(),
                    loss_rmse.item(),
                    loss_mae.item(),
                    loss_irmse.item(),
                    loss_imae.item(),
                )
            )
        
        if i in batch_to_be_saved:
            
            save_sample_images(imgs_hr, imgs_lr, gen_hr, val_sample_path, i)
            logger.info("Saved Validation Images...")
    
    avg_rmse = np.sqrt(np.mean(np.square(loss_dict['rmse'])))
    avg_mae = np.mean(loss_dict['mae'])
    avg_irmse = np.sqrt(np.mean(np.square(loss_dict['irmse'])))
    avg_imae = np.mean(loss_dict['imae'])
    
    writer.add_scalar("Final_RMSE_mean", avg_rmse, val_n)
    writer.add_scalar("Final_MAE_mean", avg_mae, val_n)
    writer.add_scalar("Final_iRMSE_mean", avg_irmse, val_n)
    writer.add_scalar("Final_iMAE_mean", avg_imae, val_n)
    
    logger.info(
                "Final Avg loss after %d batches [RMSE: %f, MAE: %f, iRMSE: %f, iMAE: %f]]" #removed content loss
                % (
                    batches_done,
                    avg_rmse,
                    avg_mae,
                    avg_irmse,
                    avg_imae,
                )
            )
    
    return avg_rmse, avg_mae
        
def main():
    
    opt = getOpt()

    # create the logdir if it does not exist
    os.makedirs(LOGDIR, exist_ok=True)
   
    val_image_save_path = os.path.join(LOGDIR,opt.name,"val_images")
    log_file_name = os.path.join(LOGDIR,opt.name,'%s.log'%opt.name)
    tensorboard_save_path = os.path.join(LOGDIR,opt.name)

    os.makedirs(val_image_save_path, exist_ok=True)

    # Create a logger
    logger = createLogger(log_file_name)

    # print(opt)
    logger.info(opt)
    
    # initiate tensorboard logger
    writer = SummaryWriter(log_dir=tensorboard_save_path)
    

    if opt.gpus is not None:
        try:
            opt.gpus = [int(s) for s in opt.gpus.split(',')]
        except ValueError:
            logger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
            exit(1)
        available_gpus = torch.cuda.device_count()
        for dev_id in opt.gpus:
            if dev_id >= available_gpus:
                logger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                .format(dev_id, available_gpus))
                exit(1)
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(opt.gpus[0])


    hr_shape = (opt.hr_height, opt.hr_width)


    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks)
    generator = nn.DataParallel(generator, device_ids = opt.gpus)
    generator.cuda()

    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    discriminator = nn.DataParallel(discriminator, device_ids = opt.gpus)
    discriminator.cuda()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_content = NormalLoss().cuda()
    criterion_pixel = torch.nn.L1Loss().cuda()

    # Load state dict for generator and discriminator
    saved_generator_chkpt = os.path.join(opt.checkpoint_model_path,"generator_best.pth")
    generator.load_state_dict(torch.load(saved_generator_chkpt))
    saved_discriminator_chkpt = os.path.join(opt.checkpoint_model_path,"discriminator_best.pth")
    discriminator.load_state_dict(torch.load(saved_discriminator_chkpt))
    
    # Only evaluate
    generator.eval()
    discriminator.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    ## Need to use PairedImageDataset Dataset class    
    val_dataloader = DataLoader(
        PairedImageDataset("/home/dataset/data.ShapeNetDepth/val/", opt, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )

    # final validation
    with torch.no_grad():
        avg_rmse, avg_mae =validate(generator, discriminator, opt, Tensor, val_dataloader, criterion_GAN, criterion_content, criterion_pixel, logger, val_image_save_path, writer)
    
    writer.flush()
    writer.close()
    
    logger.info("Validation Done! Check results.. Adios!")

if __name__=='__main__':
    main()