#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import time
import random
#import cProfile
import pickle
import json
import numpy as np
import cv2
#from PIL import Image
#import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from pathlib import Path
#import pycocotools
from tqdm import tqdm

import torch

from yolact import Yolact
from data import COCOInstanceSegmentation, get_label_map, MEANS, COLORS
from data import cfg, set_cfg, set_dataset
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
#from utils.functions import MovingAverage, ProgressBar
#from utils import timer
#from utils.functions import SavePath
#from layers.box_utils import jaccard, center_size, mask_iou
from layers.output_utils import postprocess, postprocess_np, undo_image_transformation


iou_thresholds = [x / 100 for x in range(50, 100, 5)]

class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """
    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """
        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def _mask_iou_np(mask1, mask2, iscrowd=False):
    '''
    mask1, mask2: [height, width] or [1, height, width]
    '''
    # If either set of masks is empty return empty result
    if mask1.shape[0] == 0 or mask2.shape[0] == 0:
        return 0.0

    # flatten masks and compute their areas
    mask1 = np.reshape(mask1 > .5, (sum(mask1.shape))).astype(np.float32)
    mask2 = np.reshape(mask2 > .5, (sum(mask2.shape))).astype(np.float32)

    area1 = np.sum(mask1)
    area2 = np.sum(mask2)

    # intersection and union
    inter = np.dot(mask1, mask2.T)
    union = area1 + area2 - inter

    if iscrowd:
        iou = inter / area1
    else:
        iou = inter / union
    return iou



def _bbox_iou_np(bbox1, bbox2, iscrowd=False):
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # check IoU
    inter_xmin = np.maximum(bbox1[0], bbox2[0])
    inter_ymin = np.maximum(bbox1[1], bbox2[1])
    inter_xmax = np.minimum(bbox1[2], bbox2[2])
    inter_ymax = np.minimum(bbox1[3], bbox2[3])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin)

    inter = inter_w * inter_h
    union = area1 + area2 - inter

    if iscrowd:
        iou = inter / area1
    else:
        iou = inter / union
    return iou


def predict_torch(model, device, image, height, width):
    model.eval()
    with torch.no_grad():
        image_tensor = torch.Tensor(image.unsqueeze(0)).to(device)

        # get predict output
        prediction = model(image_tensor)[0]

        # convert prediction to numpy array
        prediction = {'net':  prediction['net'],
                      #'detection': {
                      'class': prediction['detection']['class'].cpu().numpy(),
                      'box': prediction['detection']['box'].cpu().numpy(),
                      'score': prediction['detection']['score'].cpu().numpy(),
                      'mask': prediction['detection']['mask'].cpu().numpy(),
                      'proto': prediction['detection']['proto'].cpu().numpy(),
                                   #}
                       }

    classes, scores, boxes, masks = postprocess_np(prediction, width, height, crop_masks=args.crop, score_threshold=args.score_threshold)
    return classes, scores, boxes, masks


def get_prediction_ap_info(ap_record, model, model_format, device, image, gt, gt_masks, height, width, num_crowd):
    """ Returns a list of APs for this image, with each element being for a class  """
    # parse ground truth
    gt_boxes = gt[:, :4]
    gt_boxes[:, [0, 2]] *= width
    gt_boxes[:, [1, 3]] *= height
    gt_classes = list(gt[:, 4].astype(int))
    gt_masks = gt_masks.reshape(-1, height*width)

    # if there is crowd object in gt (at end), separate them out
    if num_crowd > 0:
        split = lambda x: (x[-num_crowd:], x[:-num_crowd])
        crowd_boxes  , gt_boxes   = split(gt_boxes)
        crowd_masks  , gt_masks   = split(gt_masks)
        crowd_classes, gt_classes = split(gt_classes)


    # predict with PyTorch pth model
    if model_format == 'PTH':
        classes, scores, boxes, masks = predict_torch(model, device, image, height, width)
    else:
        raise ValueError('invalid model format')

    # if there's no predict object, return
    # TODO: correct?
    if len(boxes) == 0:
        return

    # parse predict result
    classes = list(classes.astype(int))
    if isinstance(scores, list):
        box_scores = list(scores[0].astype(float))
        mask_scores = list(scores[1].astype(float))
    else:
        scores = list(scores.astype(float))
        box_scores = scores
        mask_scores = scores
    masks = masks.reshape(-1, height*width)

    # sort prediction bbox and mask index with scores
    pred_num = len(classes)
    gt_num   = len(gt_classes)
    box_indices = sorted(range(pred_num), key=lambda i: -box_scores[i])
    mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

    # entry list for calculating box & mask AP
    ap_types = [
        ('box',
         lambda i,j: _bbox_iou_np(boxes[i], gt_boxes[j]),
         lambda i,j: _bbox_iou_np(boxes[i], crowd_boxes[j]),
         box_scores,
         box_indices
        ),

        ('mask',
         lambda i,j: _mask_iou_np(masks[i], gt_masks[j]),
         lambda i,j: _mask_iou_np(masks[i], crowd_masks[j]),
         mask_scores,
         mask_indices
        )
    ]

    # loop on all the gt & predict classes for 1 single image
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        # gt number for one class
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        # loop on every iou_threshold in 0.5:0.05:0.95
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for ap_type, iou_func, crowd_iou_func, scores, indices in ap_types:
                # flag to mark gt has been matched to one prediction
                gt_used = [False] * len(gt_classes)

                # add gt number to class AP object
                ap_obj = ap_record[ap_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                # check every prediction object to match ground truth
                for i in indices:
                    if classes[i] != _class:
                        continue

                    # only gt with iou > iou_threshold will be used to match
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    # loop every gt to match prediction with best iou
                    for j in range(gt_num):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue

                        # calculate iou
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        # got matched gt, record a TP in AP object and mark it as used
                        gt_used[max_match_idx] = True
                        ap_obj.push(scores[i], True)
                    else:
                        # If the prediction matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            # check if prediction could match a crowd gt
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                # calculate iou with crowd gt
                                iou = crowd_iou_func(i, j)

                                # match a crowd gt, will ignore
                                # this prediction
                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            # no matching gt, record a FP in AP object
                            ap_obj.push(scores[i], False)

    return



def evaluate(args, model, model_format, dataset, device):
    #model.detect.use_fast_nms = args.fast_nms
    #model.detect.use_cross_class_nms = args.cross_class_nms
    #cfg.mask_proto_debug = args.mask_proto_debug


    # init AP record object lists
    ap_record = {
        'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
    }

    # prepare dataset
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    dataset_indices = list(range(len(dataset)))
    dataset_indices = dataset_indices[:dataset_size]

    # main eval loop
    pbar = tqdm(total=len(dataset_indices), desc='Eval model')
    for it, image_idx in enumerate(dataset_indices):
        pbar.update(1)
        #timer.reset()

        image, gt, gt_masks, height, width, num_crowd = dataset.pull_item(image_idx)

        get_prediction_ap_info(ap_record, model, model_format, device, image, gt, gt_masks, height, width, num_crowd)


    pbar.close()

    return calc_map(ap_record)



def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()




def load_eval_model(model_path, device):
    # support of PyTorch pth model
    if model_path.endswith('.pth'):
        model = Yolact()
        model.load_weights(model_path, device=device)
        model.eval()
        model_format = 'PTH'
    else:
        raise ValueError('invalid model file')

    return model, model_format



def main():
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='model file to predict')
    parser.add_argument('--config', type=str, required=True, help='config object to use.')
    parser.add_argument('--dataset', type=str, required=False, default=None,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')

    #parser.add_argument('--fast_nms', default=True, type=str2bool,
                        #help='Whether to use a faster, but not entirely correct version of NMS.')
    #parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        #help='Whether compute NMS cross-class or per-class.')

    #parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        #help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    #parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        #help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')

    parser.set_defaults(crop=True, detect=False)

    global args
    args = parser.parse_args()

    set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('results'):
        os.makedirs('results')

    dataset = COCOInstanceSegmentation(cfg.dataset.valid_images, cfg.dataset.valid_info,
                            transform=BaseTransform(), has_gt=cfg.dataset.has_gt)

    # get eval model
    model, model_format = load_eval_model(args.model_path, device)

    evaluate(args, model, model_format, dataset, device)



if __name__ == '__main__':
    main()
