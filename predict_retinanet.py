import os
import argparse
import json

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(train_resnetpath):
            raise





def _main_(args):

    config_path = args.conf
    input_path = args.input
    output_path = args.output
    #config_path = 'config_resnet50.json'
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    os.system(' python keras-retinanet-master/setup.py build_ext --inplace')
    print ('Prediction RetinaNet')
    os.system('python keras-retinanet-master/predict.py --weights "' + config['train']['saved_weights_infer'] +
                '" --backbone ' + config['model']['backend'] +
                ' --input_path ' + input_path +
                ' --output_path ' + output_path +
                ' --score_threshold 0.5' )




if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')

    args = argparser.parse_args()
    _main_(args)
