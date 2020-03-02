#! /usr/bin/env python

import time
import os
import argparse
import json
import cv2
import sys
sys.path += [os.path.abspath('keras-yolo2-master')]

from preprocessing import parse_annotation
from utils import draw_boxes
from utils_y3.utils import get_yolo_boxes, makedirs
from frontend import YOLO
from keras.models import load_model
from tqdm import tqdm
import numpy as np


def _main_(args):

    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output
    threshold    = args.score_threshold

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = threshold, threshold

    ###############################
    #   Load the model
    ###############################
    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    ###############################

    if input_path[-4:] == '.mp4':
        video_out = input_path[:-4] + '_detected' + input_path[-4:]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'),
                               50.0,
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()

    else:

        image_paths = []

        if os.path.isdir(input_path):
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
        times = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)
            start = time.time()
		 # predict the bounding boxes
            boxes = yolo.predict(image)

            times.append(time.time() - start)
		 # draw bounding boxes on the image using labels
            image = draw_boxes(image, boxes, config['model']['labels'])

         # write the image with bounding boxes to file
            #print(len(boxes), 'boxes are found')

            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))
           #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

        print('Tiempo promedio:' + str(np.mean(times)))

        #file = open(args.output + '/time.txt','w')
        #file.write('Tiempo promedio:' + str(np.mean(times)))
        #file.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')
    argparser.add_argument('-s', '--score_threshold',       help='score threshold detection.', default=0.5, type=float)

    args = argparser.parse_args()
    _main_(args)
