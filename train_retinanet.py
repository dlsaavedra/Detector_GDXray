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

    os.system(' python keras-retinanet-master/setup.py build_ext --inplace')
    print ('Training Retina resnet50')
    os.system('python keras-retinanet-master/keras_retinanet/bin/train.py --weights '+ config['train']['based_weights'] +
                ' --snapshot ' + config['train']['saved_weights'] +
                ' --backbone ' + config['model']['backend']+
                ' --batch-size '+ str(config['train']['batch_size'])+
                ' --multi-gpu ' + str(config['train']['multi-gpu'])+
                ' --epochs '+ str(config['train']['nb_epochs'])+
                ' --steps '+ str(config['train']['steps']) +
                ' --image-min-side ' + str(config['model']['image_min_side']) +
                ' --image-max-side ' + str(config['model']['image_max_side']) +
                ' --compute-val-loss   csv ' + csv_anns + ' ' +csv_classes +
                ' --val-annotations ' + csv_anns_val)
                 #+ ' > ../Experimento_3/Resultados_retinanet/resnet50/retinanet_resnet.output 2> ../Experimento_3/Resultados_retinanet/resnet50/retinanet_resnet.err')

    #EL modelo se guarda keras-retinanet-master/snapshots/resnet50_csv.h5



    print('Convert Inference Model')
    os.system('keras-retinanet-master/keras_retinanet/bin/convert_model.py ' + config['train']['saved_weights'] + ' ' + config['train']['saved_weights_infer'])


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
