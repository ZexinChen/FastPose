import argparse
import time
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from network.fastpose import get_model, use_vgg
from training.datasets.coco import get_loader

# Hyper-params
parser = argparse.ArgumentParser(description='PyTorch rtpose Training')
parser.add_argument('--data_dir', default='./data/coco/images', type=str, metavar='DIR',
                    help='path to where coco images stored') 
parser.add_argument('--mask_dir', default='./data/coco/mask', type=str, metavar='DIR',
                    help='path to where coco images stored')    
parser.add_argument('--logdir', default='./tb_logs/default', type=str, metavar='DIR',
                    help='path to where tensorboard log restore')                                       
parser.add_argument('--json_path', default='./data/coco/COCO.json', type=str, metavar='PATH',
                    help='path to where coco images stored')                                      

parser.add_argument('--model_path', default='./network/weights/', type=str, metavar='DIR',
                    help='path to where the model saved') 
parser.add_argument('--ckpt', default='./network/weights/fastpose.pth', type=str, metavar='DIR',
                    help='path to where the model saved') 
parser.add_argument('--resume', default='', type=str, metavar='DIR',
                    help='path to where the model saved') 
parser.add_argument('--resume_opt', default='', type=str, metavar='DIR',
                    help='path to where the model saved') 

parser.add_argument('--lr', '--learning-rate', default=8e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  
parser.add_argument('--nesterov', dest='nesterov', action='store_true')     
                                                   
parser.add_argument('-o', '--optim', default='sgd', type=str)

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--freeze', default=0, type=int, metavar='N',
                    help='number of total epochs to freeze')
                    
                    
parser.add_argument('-b', '--batch_size', default=27, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')
from tensorboardX import SummaryWriter      
args = parser.parse_args()  
               

params_transform = dict()
params_transform['mode'] = 5
# === aug_scale ===
params_transform['scale_min'] = 0.5
params_transform['scale_max'] = 1.1
params_transform['scale_prob'] = 1
params_transform['target_dist'] = 0.6
# === aug_rotate ===
params_transform['max_rotate_degree'] = 40

# ===
params_transform['center_perterb_max'] = 40

# === aug_flip ===
params_transform['flip_prob'] = 0.5

params_transform['np'] = 56
params_transform['sigma'] = 7.0
params_transform['limb_width'] = 1.289

def build_names():
    names = []

    for j in range(1, 7):
        for k in range(2, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


def get_loss(saved_for_loss, heat_temp, heat_weight):
    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    total_loss = 0

    for j in range(6):

        pred2 = saved_for_loss[j] * heat_weight
        gt2 = heat_temp * heat_weight 
        
        loss2 = criterion(pred2, gt2) 

        total_loss += loss2

        saved_for_log[names[j]] = loss2.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, 0:-1, :, :])
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, 0:-1, :, :])
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data)
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data)

    return total_loss, saved_for_log
         

def train(train_loader, model, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target, heat_mask) in enumerate(train_loader):
        # measure data loading time       
        data_time.update(time.time() - end)

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        heat_mask = heat_mask.cuda()

        img_var = torch.autograd.Variable(img)
        heatmap_target = torch.autograd.Variable(heatmap_target)
        heat_mask = torch.autograd.Variable(heat_mask)
        
        # compute output
        _,saved_for_loss = model(img_var)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask)
        
        for name,_ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\n'.format( data_time=data_time)
            print_string += 'Loss loss.val:{} (loss.avg:{})\n'.format(losses.val.item(),losses.avg.item())

            for name, value in meter_dict.items():
                print_string+='{}: loss.val:{} (loss.avg:{})\n'.format(name, value.val,value.avg)
            print(print_string)
            for param_group in optimizer.param_groups:
                print('lr:',param_group['lr'])
                
            writer.add_scalars('data/scalar_group_iter', {'train loss iter avg': losses.avg,'train loss val': losses.val}, epoch*len(train_loader)+i)
            return losses.avg  
    return losses.avg  
        
def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossse_noBackward = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, heatmap_target, heat_mask) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img = torch.autograd.Variable(img.cuda(),volatile=True)
        heatmap_target = heatmap_target.cuda()
        heat_mask = heat_mask.cuda()
        
        # compute output
        _,saved_for_loss = model(img)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask)
        for name,_ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        
        losses.update(total_loss.item(), img.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\n'.format( data_time=data_time)
            print_string += 'Loss {loss.val} ({loss.avg})\n'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string+='{}: {} (avg:{})\n'.format(name, value.val, value.avg)
            print(print_string)
  

    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


print("Loading dataset...")
# load data
train_data = get_loader(args.json_path, args.data_dir,
                        args.mask_dir, 368, 8,
                        'vgg', args.batch_size,
                        shuffle=True, params_transform=params_transform, training=True, num_workers=8)
print('train dataset len: {}'.format(len(train_data.dataset)))

# validation data
valid_data = get_loader(args.json_path, args.data_dir, args.mask_dir, 368,
                            8, preprocess='vgg', params_transform=params_transform, training=False,
                            batch_size=20, shuffle=False, num_workers=4)
print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
model = get_model(trunk='vgg19')
model = torch.nn.DataParallel(model).cuda()

if args.ckpt != '':
    cpt_state_dict = torch.load(args.ckpt)

    new_state_dict = OrderedDict()
    for k, v in cpt_state_dict.items():
        # for pretrained checkpoint
        new_state_dict[k]=v
        # # for openpose checkpoint
        # if 'model0' in k:
        #     new_state_dict['module.'+k]=v
    model_dict = model.state_dict()
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('resumed from %s'%(args.ckpt))



# Fix the VGG weights first, and then the weights will be released
requires_grad_false = 0
for param in model.module.model0.parameters():
    requires_grad_false+=1
    param.requires_grad = False
print(requires_grad_false,' param requires_grad = false')   

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)
 
writer = SummaryWriter(log_dir=args.logdir)       
                                                                                          
for epoch in range(args.freeze):
    # train for one epoch
    train_loss = train(train_data, model, optimizer, epoch, writer)

    # evaluate on validation set
    val_loss = validate(valid_data, model, epoch)  
    lr_scheduler.step(val_loss)                        
                                 
    writer.add_scalars('data/scalar_group', {'train loss': train_loss,
                                             'val loss': val_loss}, epoch)

# Release all weights                                   
for param in model.module.parameters():
    param.requires_grad = True
trainable_vars = [
    {'params': model.module.model0.parameters(), 'lr': args.lr},
    {'params': model.module.model1_2.parameters(), 'lr': args.lr},
    {'params': model.module.model2_2.parameters(), 'lr': args.lr*4.0},
    {'params': model.module.model3_2.parameters(), 'lr': args.lr*4.0},
    {'params': model.module.model4_2.parameters(), 'lr': args.lr*4.0},
    {'params': model.module.model5_2.parameters(), 'lr': args.lr*4.0},
    {'params': model.module.model6_2.parameters(), 'lr': args.lr*4.0},
]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov) 

if args.resume != '':
    cpt_state_dict = torch.load(args.resume)
    model.load_state_dict(cpt_state_dict)
    print('load model from ',args.resume)
if args.resume_opt != '':
    cpt_state_dict_opt = torch.load(args.resume_opt)
    optimizer.load_state_dict(cpt_state_dict_opt)
    print('load optimizer from ',args.resume_opt)
                                      
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

best_val_loss = np.inf

model_save_filename = os.path.join(args.logdir,"best_pose-copy.pth")
opt_save_filename = os.path.join(args.logdir,"best_pose-copy-opt.pth")
for epoch in range(args.freeze, args.epochs):

    # train for one epoch
    train_loss = train(train_data, model, optimizer, epoch, writer)

    # evaluate on validation set
    val_loss = validate(valid_data, model, epoch)   
    
    writer.add_scalars('data/scalar_group', {'train loss': train_loss,
                                             'val loss': val_loss}, epoch)
    lr_scheduler.step(val_loss)                        
    
    is_best = val_loss<best_val_loss
    best_val_loss = max(val_loss, best_val_loss)
    if is_best:
        torch.save(model.state_dict(), model_save_filename)  
        torch.save(optimizer.state_dict(), opt_save_filename)  

        
writer.export_scalars_to_json(os.path.join(args.model_path,"tensorboard/all_scalars.json"))
writer.close()    
