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
    #config_path = 'config_resnet50.json'
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    if not os.path.isfile(config['train']['name_csv'] + '.csv'):
        print('Create csv Training')
        aux = "-".join(config['model']['labels'])
        os.system('python keras-retinanet-master/Create_csv.py '+ config['train']['train_annot_folder']+ ' --path_save ' + config['train']['name_csv'] + ' --labels ' + aux)

    if not os.path.isfile(config['valid']['name_csv'] + '.csv'):
        print('Create csv Validation')
        os.system('python keras-retinanet-master/Create_csv.py '+ config['valid']['valid_annot_folder']+ ' --path_save ' + config['valid']['name_csv'] + ' --labels ' + aux)

    if not os.path.isfile(config['test']['name_csv'] + '.csv'):
        print('Create csv Testing')
        os.system('python keras-retinanet-master/Create_csv.py '+ config['test']['test_annot_folder']+ ' --path_save ' + config['test']['name_csv'] + ' --labels ' + aux)

    csv_anns = config['train']['name_csv'] + '_anns.csv'
    csv_classes = config['train']['name_csv'] + '_class.csv'
    csv_anns_val = config['valid']['name_csv'] + '_anns.csv'
    csv_anns_test = config['test']['name_csv'] + '_anns.csv'
    csv_classes_test = config['test']['name_csv'] + '_class.csv'


    print('Evaluate')
    print(csv_anns_test)
    print(csv_classes_test)
    os.system('python keras-retinanet-master/keras_retinanet/bin/evaluate.py  --backbone '  + config['model']['backend']+
                ' --image-min-side ' + str(config['model']['image_min_side']) +
                ' --image-max-side ' + str(config['model']['image_max_side']) +
                ' --iou-threshold 0.5 --score-threshold 0.5 --model ' + config['train']['saved_weights_infer'] +
                ' csv ' + csv_anns_test + ' ' + csv_classes_test)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='train and evaluate ssd model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    #argparser.add_argument('-o', '--output', help='path to save the experiment')
    args = argparser.parse_args()
    _main_(args)
