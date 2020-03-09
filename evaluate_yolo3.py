import os
import argparse

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise





def _main_(args):

    config_path = args.conf
    #output_path = args.output

    #makedirs(output_path)

    print('Test full_yolo3')
    os.system('python keras-yolo3-master/evaluate.py -c ' + config_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate ssd model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    args = argparser.parse_args()
    _main_(args)
