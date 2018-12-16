import os
import time

import cv2
import numpy as np
import json
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from network.post import decode_pose_test
from network import im_transform
import demo.utils_eval
import network.post
from tqdm import tqdm

'''
MS COCO annotation order:
0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
14: r knee		15: l ankle		16: r ankle

The order in this work:
(0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
17-'left_ear' )
'''

ORDER_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
our_order = [0, 17, 6, 8, 10, 5, 7, 9,
             12, 14, 16, 11, 13, 15, 2, 1, 4, 3,
             18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

MID_1 = [1, 8,  9, 1,  11, 12, 1, 2, 3,
         2,  1, 5, 6, 5,  1, 0,  0,  14, 15]

MID_2 = [8, 9, 10, 11, 12, 13, 2, 3, 4,
         16, 5, 6, 7, 17, 0, 14, 15, 16, 17]


def eval_coco(outputs, dataDir, imgIds):
    """Evaluate images on Coco test set
    :param outputs: list of dictionaries, the models' processed outputs
    :param dataDir: string, path to the MSCOCO data directory
    :param imgIds: list, all the image ids in the validation set
    :returns : float, the mAP score
    """
    with open('results.json', 'w') as f:
        json.dump(outputs, f)  
    annType = 'keypoints'
    prefix = 'person_keypoints'

    # initialize COCO ground truth api
    dataType = 'val2014'
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
    cocoGt = COCO(annFile)  # load annotations
    cocoDt = cocoGt.loadRes('results.json')  # load model outputs

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    os.remove('results.json')
    # return Average Precision
    return cocoEval.stats[0]

def get_multiplier(img):
    """Computes the sizes of image at different scales
    :param img: numpy array, the current image
    :returns : list of float. The computed scales
    """
    scale_search = [0.5, 1., 1.5, 2, 2.5]
    return [x * 368. / float(img.shape[0]) for x in scale_search]

def get_coco_val(file_path):
    """Reads MSCOCO validation informatio
    :param file_path: string, the path to the MSCOCO validation file
    :returns : list of image ids, list of image file paths, list of widths,
               list of heights
    """
    val_coco = pd.read_csv(file_path, sep='\s+', header=None)
    image_ids = list(val_coco[1])
    file_paths = list(val_coco[2])
    heights = list(val_coco[3])
    widths = list(val_coco[4])

    return image_ids, file_paths, heights, widths

def get_outputs_test(multiplier, img, model, preprocess,use_cuda=True):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], 37))
    # paf_avg = np.zeros((img.shape[0], img.shape[1], 38))
    max_scale = multiplier[-1]
    max_size = max_scale * img.shape[0]
    # padding
    max_cropped, _, _ = im_transform.crop_with_factor(
        img, max_size, factor=8, is_ceil=True)
    batch_images = np.zeros(
        (len(multiplier), 3, max_cropped.shape[0], max_cropped.shape[1]))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_croped, im_scale, real_shape = im_transform.crop_with_factor(
            img, inp_size, factor=8, is_ceil=True)

        if preprocess == 'rtpose':
            im_data = rtpose_preprocess(im_croped)

        elif preprocess == 'vgg':
            im_data = vgg_preprocess(im_croped)

        elif preprocess == 'inception':
            im_data = inception_preprocess(im_croped)

        elif preprocess == 'ssd':
            im_data = ssd_preprocess(im_croped)

        batch_images[m, :, :im_data.shape[1], :im_data.shape[2]] = im_data

    # several scales as a batch
    if use_cuda:
        batch_var = torch.from_numpy(batch_images).cuda().float()
        predicted_outputs, _ = model(batch_var)
        output2 = predicted_outputs
        heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
    else:
        batch_var = torch.from_numpy(batch_images).float()
        predicted_outputs, _ = model(batch_var)
        output2 = predicted_outputs
        heatmaps = output2.data.numpy().transpose(0, 2, 3, 1)

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_cropped, im_scale, real_shape = im_transform.crop_with_factor(
            img, inp_size, factor=8, is_ceil=True)
        heatmap = heatmaps[m, :int(im_cropped.shape[0] /8), :int(im_cropped.shape[1] / 8), :]
        heatmap = cv2.resize(heatmap, None, fx=8, fy=8,interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
        heatmap = cv2.resize(
            heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)


        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

    return heatmap_avg

def append_result_test(image_id, persons, outputs):
    """Build the outputs to be evaluated
    :param image_id: int, the id of the current image
    :param person_to_joint_assoc: numpy array of joints associations
    :param joint_list: list, list of joints
    :param outputs: list of dictionaries with the following keys: image_id,
                    category_id, keypoints, score
    """
    for p_idx in range(len(persons)):
        if len(persons[p_idx])<=5:
            continue
        one_result = {
            "image_id": 0,
            "category_id": 1,
            "keypoints": [],
            "score": 0
        }
        one_result["image_id"] = image_id
        keypoints = np.zeros((17, 3))

        for point in persons[p_idx]:
            part = our_order[int(point[4])]
            if part==17:
                continue
            keypoints[part, 0] = point[0]
            keypoints[part, 1] = point[1]
            keypoints[part, 2] = 1


        one_result["score"] = point[2]
        one_result["keypoints"] = list(keypoints.reshape(51))

        outputs.append(one_result)

def handle_paf_and_heat_test(normal_heat, flipped_heat):
    """Compute the average of normal and flipped heatmap and paf
    :param normal_heat: numpy array, the normal heatmap
    :param normal_paf: numpy array, the normal paf
    :param flipped_heat: numpy array, the flipped heatmap
    :param flipped_paf: numpy array, the flipped  paf
    :returns: numpy arrays, the averaged paf and heatmap
    """

    # The order to swap left and right of heatmap
    swap_heat = np.array((0, 1, 5, 6, 7, 2, 3, 4, 11, 12,
                          13, 8, 9, 10, 15, 14, 17, 16, 
                          19,18,21,20,35,23,25,24,27,26,28,30,29,32,31,34,33,22,36)) 
    # swap_paf = np.array((1, 0, 3, 2, 4, 5, 7, 6, 9, 8, 10, 12, 11, 14, 13, 16, 15,
    #     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37))

    # paf's order
    # 0,1 2,3 4,5
    # neck to right_hip, right_hip to right_knee, right_knee to right_ankle

    # 6,7 8,9, 10,11
    # neck to left_hip, left_hip to left_knee, left_knee to left_ankle

    # 12,13 14,15, 16,17, 18, 19
    # neck to right_shoulder, right_shoulder to right_elbow, right_elbow to
    # right_wrist, right_shoulder to right_ear

    # 20,21 22,23, 24,25 26,27
    # neck to left_shoulder, left_shoulder to left_elbow, left_elbow to
    # left_wrist, left_shoulder to left_ear

    # 28,29, 30,31, 32,33, 34,35 36,37
    # neck to nose, nose to right_eye, nose to left_eye, right_eye to
    # right_ear, left_eye to left_ear So the swap of paf should be:
    # swap_paf = np.array((6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 20, 21, 22, 23,
    #                      24, 25, 26, 27, 12, 13, 14, 15, 16, 17, 18, 19, 28,
    #                      29, 32, 33, 30, 31, 36, 37, 34, 35))

    # flipped_paf = flipped_paf[:, ::-1, :]

    # # The pafs are unit vectors, The x will change direction after flipped.
    # # not easy to understand, you may try visualize it.
    # flipped_paf[:, :, swap_paf[1::2]] = flipped_paf[:, :, swap_paf[1::2]]
    # flipped_paf[:, :, swap_paf[::2]] = -flipped_paf[:, :, swap_paf[::2]]
    # averaged_paf = (normal_paf + flipped_paf[:, :, swap_paf]) / 2.
    # averaged_paf = (
    #     normal_paf + flipped_paf[:, ::-1, :][:, :, swap_paf]) / 2.
    averaged_heatmap = (
        normal_heat + flipped_heat[:, ::-1, :][:, :, swap_heat]) / 2.

    return averaged_heatmap


def handle_paf_and_heat(normal_heat, flipped_heat, normal_paf, flipped_paf):
    """Compute the average of normal and flipped heatmap and paf
    :param normal_heat: numpy array, the normal heatmap
    :param normal_paf: numpy array, the normal paf
    :param flipped_heat: numpy array, the flipped heatmap
    :param flipped_paf: numpy array, the flipped  paf
    :returns: numpy arrays, the averaged paf and heatmap
    """

    # The order to swap left and right of heatmap
    swap_heat = np.array((0, 1, 5, 6, 7, 2, 3, 4, 11, 12,
                          13, 8, 9, 10, 15, 14, 17, 16, 18))
    swap_paf = np.array((1, 0, 3, 2, 17, 5, 7, 6, 9, 8, 10, 12, 11, 14, 13, 16, 15, 4, 18))

    averaged_paf = (
        normal_paf + flipped_paf[:, ::-1, :][:, :, swap_paf]) / 2.
    averaged_heatmap = (
        normal_heat + flipped_heat[:, ::-1, :][:, :, swap_heat]) / 2.

    return averaged_paf, averaged_heatmap

def run_eval_test(image_dir, anno_dir, vis_dir, image_list_txt, model, preprocess):
    """Run the evaluation on the test set and report mAP score
    :param model: the model to test
    :returns: float, the reported mAP score
    """
    # This txt file is fount in the caffe_rtpose repository:
    # https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master
    img_ids, img_paths, img_heights, img_widths = get_coco_val(
        image_list_txt)
    # img_ids = img_ids[81:82]
    # img_paths = img_paths[81:82]
    print("Total number of validation images {}".format(len(img_ids)))

    # iterate all val images
    outputs = []
    print("Processing Images in validation set")
    for i in tqdm(range(len(img_ids))):
        # if i % 10 == 0 and i != 0:
        #     print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,"Processed {} images".format(i))

        oriImg = cv2.imread(os.path.join(image_dir, 'val2014/' + img_paths[i]))
        # Get the shortest side of the image (either height or width)
        shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image
        multiplier = get_multiplier(oriImg)
        orig_heat = get_outputs_test(
            multiplier, oriImg, model,  preprocess)
              
        # Get results of flipped image
        swapped_img = oriImg[:, ::-1, :]
        flipped_heat = get_outputs_test(multiplier, swapped_img,
                                                model, preprocess)

        # compute averaged heatmap and paf
        heatmap = handle_paf_and_heat_test(
            orig_heat, flipped_heat)
        # choose which post-processing to use, our_post_processing
        # got slightly higher AP but is slow.
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}


        joint_list, middle_joint_list = decode_pose_test(oriImg, param, heatmap, [])
        jl,mjl = network.post.outputChangeForm_test(joint_list.tolist())
        persons,pair,persons_withMiddlePoint,jointLength,peaks_1,peaks_2,_ = demo.utils_eval.person_group_test(jl,mjl,distance_tolerance=0.3)

        new_persons = demo.utils_eval.removeDouble(persons,jointLength)

        vis_path = os.path.join(vis_dir, img_paths[i])
        append_result_test(img_ids[i], new_persons, outputs)

    # Eval and show the final result!
    return eval_coco(outputs=outputs, dataDir=anno_dir, imgIds=img_ids)

def run_eval_test_nocuda(image_dir, anno_dir, vis_dir, image_list_txt, model, preprocess):
    """Run the evaluation on the test set and report mAP score
    :param model: the model to test
    :returns: float, the reported mAP score
    """
    # This txt file is fount in the caffe_rtpose repository:
    # https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master
    img_ids, img_paths, img_heights, img_widths = get_coco_val(
        image_list_txt)
    # img_ids = img_ids[81:82]
    # img_paths = img_paths[81:82]
    print("Total number of validation images {}".format(len(img_ids)))

    # iterate all val images
    outputs = []
    print("Processing Images in validation set")
    for i in tqdm(range(len(img_ids))):
        # if i % 10 == 0 and i != 0:
        #     print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,"Processed {} images".format(i))

        oriImg = cv2.imread(os.path.join(image_dir, 'val2014/' + img_paths[i]))
        # Get the shortest side of the image (either height or width)
        shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image
        multiplier = get_multiplier(oriImg)
        orig_heat = get_outputs_test(
            multiplier, oriImg, model,  preprocess,use_cuda=False)
              
        # Get results of flipped image
        swapped_img = oriImg[:, ::-1, :]
        flipped_heat = get_outputs_test(multiplier, swapped_img,
                                                model, preprocess,use_cuda=False)

        # compute averaged heatmap and paf
        heatmap = handle_paf_and_heat_test(
            orig_heat, flipped_heat)
        # choose which post-processing to use, our_post_processing
        # got slightly higher AP but is slow.
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        # canvas, to_plot, candidate, subset = decode_pose(
        #     oriImg, param, heatmap, paf)

        joint_list, middle_joint_list = decode_pose_test(oriImg, param, heatmap, [])
        jl,mjl = network.post.outputChangeForm_test(joint_list.tolist())
        persons,pair,persons_withMiddlePoint,jointLength,peaks_1,peaks_2,_ = demo.utils_eval.person_group_test(jl,mjl,distance_tolerance=0.3)
        # ann_forCOCO,all_socre,new_person = utils_eval.postProcess(persons_withMiddlePoint)
        new_persons = demo.utils_eval.removeDouble(persons,jointLength)

        vis_path = os.path.join(vis_dir, img_paths[i])
        # cv2.imwrite(vis_path, to_plot)
        # subset indicated how many peoples foun in this image.
        # append_result_test(img_ids[i], persons, outputs)
        append_result_test(img_ids[i], new_persons, outputs)

        # if i == 1:
        #     break


        # cv2.imshow('test', canvas)
        # cv2.waitKey(0)
    # Eval and show the final result!
    return eval_coco(outputs=outputs, dataDir=anno_dir, imgIds=img_ids)

