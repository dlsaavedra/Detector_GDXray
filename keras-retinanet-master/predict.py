import keras
import sys
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)



def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')



    parser.add_argument('--weights',         help='model.', type=str)
    parser.add_argument('--backbone',         help='backbone model', default='resnet50', type=str)
    parser.add_argument('--input_path',       help='folder input.', type=str)
    parser.add_argument('--output_path',       help='folder output.', default='ouput/', type=str)
    parser.add_argument('--score_threshold',       help='score threshold detection.', default=0.5, type=float)
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    makedirs(args.output_path)

    def get_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    # use this environment flag to change which GPU to use
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())
    model_path = args.weights

    # load retinanet model
    model = models.load_model(model_path, backbone_name=args.backbone)


    labels_to_names = {0: 'Razor' , 1: 'Gun', 2: 'Knife', 3: 'Shuriken'}


    image_paths = []

    if os.path.isdir(args.input_path):
        for inp_file in os.listdir(args.input_path):
            image_paths += [args.input_path + inp_file]
    else:
        image_paths += [args.input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    times = []

    for image_path in image_paths:
        image = read_image_bgr(image_path)
        print(image_path)
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        times.append(time.time() - start)
        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color, thickness=5)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

        #plt.figure(figsize=(15, 15))
        #plt.axis('off')
        save_path = args.output_path + image_path.split('/')[-1]
        plt.imsave(save_path,draw)

    file = open(args.output_path + 'time.txt','w')

    file.write('Tiempo promedio:' + str(np.mean(times)))

    file.close()

if __name__ == '__main__':
    main()
