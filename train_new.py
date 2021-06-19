#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from data import *
from utils.augmentations import YOLACTAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from layers.modules import MultiBoxLoss
from yolact import Yolact

# Oof
import eval as eval_script

# global value to record the best val loss and COCO mask AP
best_val_loss = 0.0
best_AP = 0.0


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train(args, epoch, model, criterion, device, train_loader, optimizer, summary_writer):
    # all the possible loss type in MultiBoxLoss
    loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']
    loss_dict = {}
    epoch_loss = 0.0

    model.train()
    tbar = tqdm(train_loader)
    for i, (images, labels) in enumerate(tbar):

        # calculate iteration from epoch and steps
        iteration = epoch * len(train_loader) + i

        # Warm up by linearly interpolating the learning rate from some smaller value
        if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
            set_lr(optimizer, (cfg.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

        # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
        for i in range(len(cfg.lr_steps)):
            if iteration >= cfg.lr_steps[i]:
                set_lr(optimizer, cfg.lr * (cfg.gamma ** (i+1)))
                break
        #while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
            #step_index += 1
            #set_lr(optimizer, args.lr * (args.gamma ** step_index))

        # forward propagation
        optimizer.zero_grad()

        targets, masks, num_crowds = labels
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        masks = [mask.to(device) for mask in masks]

        preds = model(images)

        # calculate loss, here losses is a dict of loss tensors
        losses = criterion(model, preds, targets, masks, num_crowds)
        losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
        loss = sum([losses[k] for k in losses])

        # backward propagation
        loss.backward() # Do this to free up vram even if loss is not finite
        if torch.isfinite(loss).item():
            optimizer.step()

        # collect loss and accuracy
        batch_loss = loss.item()
        epoch_loss += batch_loss

        # sum up for each loss type
        for loss_type, loss_value in losses.items():
            if loss_type not in loss_dict:
                loss_dict[loss_type] = (loss_value).item()
            else:
                loss_dict[loss_type] += (loss_value).item()

        # prepare loss display labels
        loss_labels = sum([[k, v/(i + 1)] for k,v in loss_dict.items()], [])
        tbar.set_description((('%s:%.2f |' * len(losses)) + ' Train loss:%.2f')
                % tuple(loss_labels + [epoch_loss/(i + 1)]))

        # log train loss
        summary_writer.add_scalar('train loss', batch_loss, iteration)

    # decay learning rate every epoch
    #if lr_scheduler:
        #lr_scheduler.step()


def validate(args, epoch, step, model, criterion, device, val_loader, log_dir, summary_writer):
    global best_val_loss

    # all the possible loss type in MultiBoxLoss
    loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']
    loss_dict = {}
    val_loss = 0.0

    # Due to the YOLACT model output is different under
    # 'train' and 'eval' mode, we couldn't use model.eval()
    # here to get val loss. 'train' model is kept but gradient
    # will not be collected
    #model.eval()
    with torch.no_grad():
        tbar = tqdm(val_loader)
        for i, (images, labels) in enumerate(tbar):
            # forward propagation
            targets, masks, num_crowds = labels
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            masks = [mask.to(device) for mask in masks]

            preds = model(images)

            # calculate loss, here losses is a dict of loss tensors
            losses = criterion(model, preds, targets, masks, num_crowds)
            losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
            loss = sum([losses[k] for k in losses])

            # collect val loss
            val_loss += loss.item()

            # sum up for each loss type
            for loss_type, loss_value in losses.items():
                if loss_type not in loss_dict:
                    loss_dict[loss_type] = (loss_value).item()
                else:
                    loss_dict[loss_type] += (loss_value).item()

            # prepare loss display labels
            loss_labels = sum([[k, v/(i + 1)] for k,v in loss_dict.items()], [])
            tbar.set_description((('%s:%.2f |' * len(losses)) + ' Val loss:%.2f')
                    % tuple(loss_labels + [val_loss/(i + 1)]))

    val_loss /= len(val_loader.dataset)
    print('Validate set: Average loss: {:.4f}'.format(val_loss))

    # log validation loss and accuracy
    summary_writer.add_scalar('val loss', val_loss, step)

    # save checkpoint with best val loss
    if val_loss < best_val_loss:
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(log_dir, 'ep{epoch:03d}-val_loss{val_loss:.3f}.pth'.format(epoch=epoch+1, val_loss=val_loss))
        torch.save(model, checkpoint_dir)
        print('Epoch {epoch:03d}: val_loss improved from {best_val_loss:.3f} to {val_loss:.3f}, saving model to {checkpoint_dir}'.format(epoch=epoch+1, best_val_loss=best_val_loss, val_loss=val_loss, checkpoint_dir=checkpoint_dir))
        best_val_loss = val_loss
    else:
        print('Epoch {epoch:03d}: val_loss did not improve from {best_val_loss:.3f}'.format(epoch=epoch+1, best_val_loss=best_val_loss))



def evaluate(args, epoch, model, device, dataset, log_dir):
    global best_AP
    with torch.no_grad():
        model.eval()
        print("Computing validation mAP (this may take a while)...", flush=True)
        eval_info = eval_script.evaluate(model, dataset, device, train_mode=True)
        model.train()

    # check COCO mask AP to store best checkpoint
    eval_AP = eval_info['mask']['all']

    # save checkpoint with best val loss
    if eval_AP > best_AP:
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(log_dir, 'ep{epoch:03d}-eval_AP{eval_AP:.3f}.pth'.format(epoch=epoch+1, eval_AP=eval_AP))
        torch.save(model, checkpoint_dir)
        print('Epoch {epoch:03d}: eval_AP improved from {best_AP:.3f} to {eval_AP:.3f}, saving model to {checkpoint_dir}'.format(epoch=epoch+1, best_AP=best_AP, eval_AP=eval_AP, checkpoint_dir=checkpoint_dir))
        best_AP = eval_AP
    else:
        print('Epoch {epoch:03d}: eval_AP did not improve from {best_AP:.3f}'.format(epoch=epoch+1, best_AP=best_AP))




class ModelLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, model:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.model = model
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.model(images)
        losses = self.criterion(self.model, preds, targets, masks, num_crowds)
        return losses


def main():
    parser = argparse.ArgumentParser(description='Yolact Training Script')
    # Model definition options
    parser.add_argument('--config', type=str, required=False, default=None,
                        help='The config object to use.')

    # Data options
    parser.add_argument('--dataset', type=str, required=False, default=None,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')

    # Training settings
    parser.add_argument('--batch_size', type=int, required=False, default=8,
        help = "batch size for train, default=%(default)s")
    parser.add_argument('--lr', type=float, required=False, default=None,
        help = "Initial learning rate. Leave as None to read this from the config. default=%(default)s")
    parser.add_argument('--momentum', type=float, required=False, default=None,
        help='Momentum for SGD. Leave as None to read this from the config.')
    parser.add_argument('--decay', type=float, required=False, default=None,
        help='Weight decay for SGD. Leave as None to read this from the config.')
    #parser.add_argument('--gamma', type=float, required=False, default=None,
        #help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')

    parser.add_argument('--num_epoch', type=int,required=False, default=100,
        help = "Number of training epochs, default=%(default)s")

    parser.add_argument('--validation_size', type=int, required=False, default=5000,
                        help='The number of images to use for validation.')
    parser.add_argument('--validation_epoch', type=int, required=False, default=2,
                        help='Output validation information every n iterations. If -1, do no validation.')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Evaluation options
    parser.add_argument('--eval_online', default=False, action="store_true",
        help='Whether to do evaluation on validation dataset during training')
    parser.add_argument('--eval_epoch_interval', type=int, required=False, default=5,
        help = "Number of iteration(epochs) interval to do evaluation, default=%(default)s")
    parser.add_argument('--save_eval_checkpoint', default=False, action="store_true",
        help='Whether to save checkpoint with best evaluation result')

    args = parser.parse_args()
    log_dir = os.path.join('logs', '000')


    if args.config is not None:
        set_cfg(args.config)

    if args.dataset is not None:
        set_dataset(args.dataset)


    # Update training parameters from the config if necessary
    def replace(name):
        if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
    replace('lr')
    replace('decay')
    #replace('gamma')
    replace('momentum')


    # create running device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    # prepare train&val dataset loader
    train_dataset = COCOInstanceSegmentation(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=YOLACTAugmentation(MEANS))

    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                   num_workers=4,
                                   shuffle=True, collate_fn=detection_collate,
                                   pin_memory=True)

    if args.validation_epoch > 0:
        # setup eval script config
        eval_script.parse_args(['--max_images='+str(args.validation_size)])

        val_dataset = COCOInstanceSegmentation(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))
        val_loader = data.DataLoader(val_dataset, args.batch_size,
                                       num_workers=4,
                                       shuffle=True, collate_fn=detection_collate,
                                       pin_memory=True)

    # get tensorboard summary writer
    summary_writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))


    model = Yolact().to(device)
    model.train()
    model.init_weights(backbone_path='weights/' + cfg.backbone.path)

    # optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio).to(device)
    #train_model = ModelLoss(model, criterion).to(device)


    # Initialize everything
    if not cfg.freeze_bn: model.freeze_bn() # Freeze bn so we don't kill our means
    model(torch.zeros(1, 3, cfg.max_size, cfg.max_size).to(device))
    if not cfg.freeze_bn: model.freeze_bn(True)

    print(model)

    # Train loop
    for epoch in range(args.num_epoch):
        print('Epoch %d/%d'%(epoch, args.num_epoch))
        train(args, epoch, model, criterion, device, train_loader, optimizer, summary_writer)
        validate(args, epoch, epoch*len(train_loader), model, criterion, device, val_loader, log_dir, summary_writer)
        if args.eval_online and (epoch+1) % args.eval_epoch_interval == 0:
            # Do eval every eval_epoch_interval epochs
            evaluate(args, epoch, model, device, val_dataset, log_dir)

    # Finally store model
    torch.save(model, os.path.join(log_dir, 'trained_final.pth'))


if __name__ == '__main__':
    main()
