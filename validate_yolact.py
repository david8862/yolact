#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import cv2, colorsys
#from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import torch

from yolact import Yolact
from data import cfg, set_cfg
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
#from utils import timer
from layers.output_utils import postprocess, postprocess_np


def get_colors(number, bright=True):
    """
    Get random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    if number <= 0:
        return []

    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / number, 1., brightness)
                  for x in range(number)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def apply_mask(image, mask, color, alpha=0.45):
    """Apply the given mask to RGB image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(0, 0, 0),
                lineType=cv2.LINE_AA)

    return image


def handle_prediction(args, prediction, image):
    height, width, channel = image.shape
    # prediction postprocess
    #classes, scores, boxes, masks = postprocess(prediction, width, height, crop_masks=True, score_threshold=0.3)
    classes, scores, boxes, masks = postprocess_np(prediction, width, height, crop_masks=True, score_threshold=0.3)

    # generate random colors by object number
    colors = get_colors(len(boxes))

    print('Found {} objects'.format(len(boxes)))
    # here we use reversed order to make sure highest
    # score object shows on top
    for i in reversed(range(len(boxes))):
        class_name = cfg.dataset.class_names[classes[i]]
        score = scores[i]
        label = '%s: %.2f' % (class_name, score)

        box = boxes[i]
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        print("Class: {}, Score: {}, Box: {},{}".format(class_name, score, (xmin, ymin), (xmax, ymax)))

        # draw box and label
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[i], 1, cv2.LINE_AA)
        image = draw_label(image, label, colors[i], (xmin, ymin))

        # draw segment mask on image
        image = apply_mask(image, masks[i], colors[i])

    return image


def validate_yolact_model_torch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = Yolact()
    model.load_weights(args.model_path, device=device)
    model.eval()

    with torch.no_grad():
        # prepare input image
        image = cv2.imread(args.image_file)
        image_tensor = torch.from_numpy(image).to(device).float()
        input_tensor = FastBaseTransform(device)(image_tensor.unsqueeze(0))

        # get predict output
        prediction = model(input_tensor)[0]

        # convert prediction to numpy array
        prediction = {'net':  prediction['net'],
                      'detection': {
                                    'class': prediction['detection']['class'].cpu().numpy(),
                                    'box': prediction['detection']['box'].cpu().numpy(),
                                    'score': prediction['detection']['score'].cpu().numpy(),
                                    'mask': prediction['detection']['mask'].cpu().numpy(),
                                    'proto': prediction['detection']['proto'].cpu().numpy(),
                                   }
                       }

        # postprocess and show result
        result_image = handle_prediction(args, prediction, image)
        cv2.imshow('Result', result_image)
        cv2.waitKey(0)



def main():
    parser = argparse.ArgumentParser(description='validate YOLACT instance segmentation model (pth/onnx/mnn) with image')
    parser.add_argument('--model_path', type=str, required=True, help='model file to predict')
    parser.add_argument('--config', type=str, required=True, help='config object to use.')
    parser.add_argument('--image_file', type=str, required=True, help='image file to predict')
    #parser.add_argument('--fast_nms', default=True, type=str2bool,
                        #help='Whether to use a faster, but not entirely correct version of NMS.')
    #parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        #help='Whether compute NMS cross-class or per-class.')
    #parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        #help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')

    args = parser.parse_args()

    set_cfg(args.config)

    #if args.detect:
        #cfg.eval_mask_branch = False

    # support of PyTorch pth model
    if args.model_path.endswith('.pth'):
        validate_yolact_model_torch(args)
    else:
        raise ValueError('invalid model file')




if __name__ == '__main__':
    main()
