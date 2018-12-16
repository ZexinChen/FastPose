import sys
sys.path.append('../FastPose/')
import unittest
import torch
from evaluate.coco_eval import run_eval_test,run_eval_test_nocuda
from network.fastpose import get_model, use_vgg
from torch import load
from collections import OrderedDict


with torch.autograd.no_grad():
    if torch.cuda.is_available():
        weight_name = './network/weights/fastpose.pth'
        model = get_model(trunk='vgg19')
        
        model = torch.nn.DataParallel(model).cuda()

        print('loading weight from ',weight_name)
        cpt_state_dict = torch.load(weight_name)
        new_state_dict = OrderedDict()
        model_dict = model.state_dict()
        for k, v in cpt_state_dict.items():
            if k in model_dict:
                # new_state_dict['module.'+k]=v
                new_state_dict[k]=v
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        print('resumed from %s'%(weight_name))

        model.eval()
        model.float()
        model = model.cuda()
        
        run_eval_test(image_dir= './data/coco/images/', 
            anno_dir = './data/coco', 
            # anno_dir = '/media/sda/coco2014/',
            vis_dir = './data/vis',
            image_list_txt='./evaluate/image_info_val2014_1k.txt', 
            model=model, preprocess='vgg')
    else:
        print('torch.cuda.is_available() is FALSE!!   Using CPU evaluating!!')
        weight_name = './tb_logs/vgg/vgg_copy_cmu_freeze0_sgd_b30_lrO7e-1/best_pose-copy.pth'
        state_dict = torch.load(weight_name,map_location='cpu')
        model = get_model(trunk='vgg19')
        
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model.eval()
        model.float()

        run_eval_test_nocuda(image_dir= './data/coco/images/', 
            anno_dir = './data/coco', 
            vis_dir = './data/vis',
            image_list_txt='./evaluate/image_info_val2014_1k.txt', 
            model=model, preprocess='vgg')



