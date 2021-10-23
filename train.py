"""
This is the codebase for Depth Completion GAN paper titled
"Sparse to Dense Depth Completion using a Generative Adversarial Network with Intelligent Sampling Strategies"

Training file for the depth-completion-gan.
In order to invoke type:

python train.py --gpus=0,1,2,3 --batch_size=8 --n_epochs=10 --residual_blocks=17 --decay_epoch=5 -n train_test

1. -n --> give a name to the run
2. Modify the val dataloader path with appropriate data directory
3. Typically the directory has the following structure
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
4. The image_hr and image_lr are the folder containing dense and sparse depth respectively
5. The meta_info.txt contains the file names of these folders. Refer to misc/ folder for sample meta_info file
6. The folder "sample" contains a few sparse samples. This is to track the model learning visually.
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
from validate import validate

import torch.nn as nn
import torch.nn.functional as F
import torch

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import torch.optim.lr_scheduler as lr_scheduler 

LOGDIR = "./logdir/"

def getOpt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="ShapeNetSparseDepth", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=7, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=192, help="dense depth height")
    parser.add_argument("--hr_width", type=int, default=256, help="dense depth width")
    parser.add_argument("--channels", type=int, default=1, help="depth image has only 1 channel")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving image samples")
    parser.add_argument("--validation_interval", type=int, default=12, help="interval between two consecutive validations")
    parser.add_argument("--checkpoint_interval", type=int, default=15, help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=17, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=5, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    parser.add_argument("--gpus", metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--meta_info_file', '-m', metavar='DIR', default="meta_info.txt", help='Meta file name')
    parser.add_argument("--checkpoint_model_path", type=str, required=False, help="Path to checkpoint model")

    return parser.parse_args()
        
def main():
    
    # setting higher values initially
    best_mean = 9999
    best_mae = 9999
    
    opt = getOpt()

    # create the logdir if it does not exists
    os.makedirs(LOGDIR, exist_ok=True)
    
    # create addition log directories
    train_image_save_path = os.path.join(LOGDIR,opt.name,"train_images")
    val_image_save_path = os.path.join(LOGDIR,opt.name,"val_images")
    saved_model_path =  os.path.join(LOGDIR,opt.name,"saved_models")
    log_file_name = os.path.join(LOGDIR,opt.name,'%s.log'%opt.name)
    tensorboard_save_path = os.path.join(LOGDIR,opt.name)

    os.makedirs(train_image_save_path, exist_ok=True)
    os.makedirs(val_image_save_path, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)

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
    # criterion_pixel = torch.nn.MSELoss().cuda()

    if opt.resume_epoch != 0:
        # Load pretrained models
        saved_generator_chkpt = os.path.join(opt.checkpoint_model_path,"generator_%d.pth" % (opt.resume_epoch-1))
        generator.load_state_dict(torch.load(saved_generator_chkpt))
        saved_discriminator_chkpt = os.path.join(opt.checkpoint_model_path,"discriminator_%d.pth" % (opt.resume_epoch-1))
        discriminator.load_state_dict(torch.load(saved_discriminator_chkpt))
        logger.info("Loaded Checkpoint model from epoch %d"%(opt.resume_epoch-1))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    ## Need to use PairedImageDataset Dataset class
    train_dataloader = DataLoader(
        PairedImageDataset("/home/dataset/data.ShapeNetDepth/train", opt, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    
    val_dataloader = DataLoader(
        PairedImageDataset("/home/dataset/data.ShapeNetDepth/val", opt, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )
    
    sample_dataloader = DataLoader(
        PairedImageDataset("/home/dataset/data.ShapeNetDepth/sample", opt, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )
    
    milestones = [opt.decay_epoch, opt.decay_epoch+2, opt.decay_epoch+3]
    
    # ----------
    #  Training
    # ----------
    for epoch in range(opt.resume_epoch, opt.n_epochs):
    
        epoch_start_time = time.time()
        
        # Adjust LR
        if epoch in milestones:
            optimizer_G.param_groups[0]['lr'] *= 0.5 
            optimizer_D.param_groups[0]['lr'] *= 0.5 
        
        for i, imgs in enumerate(train_dataloader): #split the imgs to two arrays
            
            batches_done = epoch * len(train_dataloader) + i+1
            
            # this will add channel axis: (4, 192, 256) --> (4, 1, 192, 256)
            lr_temp = torch.unsqueeze(imgs["lr"], 1)
            hr_temp = torch.unsqueeze(imgs["hr"], 1)
            
            # Configure model input
            imgs_lr = Variable(lr_temp.type(Tensor))
            imgs_hr = Variable(hr_temp.type(Tensor))
            
            #send equal batch partitions to differnt gpus
            imgs_lr, imgs_hr = imgs_lr.to('cuda'), imgs_hr.to('cuda')
                    
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)
            
            valid, fake = valid.to('cuda'), fake.to('cuda')
            
            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
            writer.add_scalar("Pixel_Loss/Train", loss_pixel, batches_done)
            
            # log learning rate
            gen_lr = optimizer_G.param_groups[0]['lr']
            writer.add_scalar("Generateor_LR", gen_lr, batches_done)
            
            if batches_done < opt.warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, opt.n_epochs-1, i+1, len(train_dataloader), loss_pixel.item())
                )
                continue

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
            writer.add_scalar("GAN_Loss/Train", loss_GAN, batches_done)

            gen_features = imgrad_yx(gen_hr)
            real_features = imgrad_yx(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)
            writer.add_scalar("Content_Loss/Train", loss_content, batches_done)

            # Total generator loss
            loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
            writer.add_scalar("Generator_Loss/Train", loss_G, batches_done)
            # loss_G = opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            writer.add_scalar("Discriminator_RealLoss/Train", loss_real, batches_done)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
            writer.add_scalar("Discriminator_FakeLoss/Train", loss_fake, batches_done)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            writer.add_scalar("Discriminator_Loss/Train", loss_D, batches_done)
            
            #Discriminator LR
            disc_lr = optimizer_D.param_groups[0]['lr']
            writer.add_scalar("Discriminator_LR", disc_lr, batches_done)
            
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            logger.info(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f, lr: %f]" #removed content loss
                % (
                    epoch,
                    opt.n_epochs-1,
                    i+1,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(), # No content loss
                    loss_GAN.item(),
                    loss_pixel.item(),
                    gen_lr,
                )
            )

            if batches_done % opt.sample_interval == 0:
                
                #first create folder
                sample_path = os.path.join(train_image_save_path,"%06d"%batches_done)
                os.makedirs(sample_path, exist_ok=True)                

                for j, imgs in enumerate(sample_dataloader): #split the imgs to two arrays
                    
                    # this will add channel axis: (4, 192, 256) --> (4, 1, 192, 256)
                    lr_temp = torch.unsqueeze(imgs["lr"], 1)
                    hr_temp = torch.unsqueeze(imgs["hr"], 1)
                    
                    # Configure model input
                    imgs_lr = Variable(lr_temp.type(Tensor))
                    imgs_hr = Variable(hr_temp.type(Tensor))
                    
                    #send equal batch partitions to differnt gpus
                    imgs_lr, imgs_hr = imgs_lr.to('cuda'), imgs_hr.to('cuda')
                    
                    with torch.no_grad():
                        gen_hr = generator(imgs_lr)
                    
                    save_sample_images(imgs_hr, imgs_lr, gen_hr, sample_path, j)
                
            if batches_done % opt.checkpoint_interval == 0:
                # Save model checkpoints
                generator_chkpt = os.path.join(saved_model_path,"generator_%d.pth" % epoch)
                torch.save(generator.state_dict(), generator_chkpt)
                discriminator_chkpt = os.path.join(saved_model_path,"discriminator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(), discriminator_chkpt)
                logger.info("Saved Checkpoint at batch {}...".format(batches_done))
                
            
            if batches_done % opt.validation_interval == 0:
                with torch.no_grad():
                    avg_rmse, avg_mae = validate(generator, discriminator, opt, Tensor, val_dataloader, criterion_GAN, criterion_content, criterion_pixel, logger, val_image_save_path, writer, batches_done)
                
                # save best checkpoint
                if avg_rmse<best_mean and avg_mae<best_mae:
                    generator_chkpt = os.path.join(saved_model_path,"generator_best.pth")
                    torch.save(generator.state_dict(), generator_chkpt)
                    discriminator_chkpt = os.path.join(saved_model_path,"discriminator_best.pth")
                    torch.save(discriminator.state_dict(), discriminator_chkpt)
                    logger.info("Saved Best Checkpoint at batch {}...".format(batches_done))
                    best_mae = avg_mae
                    best_mean = avg_rmse
        
        logger.info("The last epoch took {} hrs... ok.".format((time.time()-epoch_start_time)/3600.0))
    
    # final validation
    with torch.no_grad():
        avg_rmse, avg_mae = validate(generator, discriminator, opt, Tensor, val_dataloader, criterion_GAN, criterion_content, criterion_pixel, logger, val_image_save_path, writer, batches_done)
    
    if avg_rmse<best_mean and avg_mae<best_mae:
        generator_chkpt = os.path.join(saved_model_path,"generator_best.pth")
        torch.save(generator.state_dict(), generator_chkpt)
        discriminator_chkpt = os.path.join(saved_model_path,"discriminator_best.pth")
        torch.save(discriminator.state_dict(), discriminator_chkpt)
        logger.info("Saved Best Checkpoint at batch {}...".format(batches_done))    
    
    writer.flush()
    writer.close()
    
    logger.info("Training Done! Check results.. Adios!")

if __name__=='__main__':
    main()