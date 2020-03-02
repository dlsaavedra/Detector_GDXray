import os
import json
import argparse
#os.mkdir('../Experimento_3/Resultados_retinanet')
#os.mkdir('../Experimento_3/Resultados_retinanet/resnet50')


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def main(args=None):
    # parse arguments
    config_path = args.conf
    #config_path = 'config_resnet50.json'
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    if not os.path.isfile(config['train']['name_csv'] + '.csv'):
        print('Create csv Training')
        os.system('python Create_csv.py '+ config['train']['train_annot_folder']+ ' --path_save ' + config['train']['name_csv'])

    if not os.path.isfile(config['valid']['name_csv'] + '.csv'):
        print('Create csv Validation')
        os.system('python Create_csv.py '+ config['valid']['valid_annot_folder']+ ' --path_save ' + config['valid']['name_csv'])

    if not os.path.isfile(config['test']['name_csv'] + '.csv'):
        print('Create csv Testing')
        os.system('python Create_csv.py '+ config['test']['test_annot_folder']+ ' --path_save ' + config['test']['name_csv'])


    csv_anns = config['train']['name_csv'] + '_anns.csv'
    csv_classes = config['train']['name_csv'] + '_class.csv'
    csv_anns_val = config['valid']['name_csv'] + '_anns.csv'
    csv_anns_test = config['test']['name_csv'] + '_anns.csv'
    csv_classes_test = config['test']['name_csv'] + '_class.csv'



    print ('Training Retina resnet50')
    os.system('python keras_retinanet/bin/train.py --weights '+ config['train']['based_weights'] +
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

    aux_weights = config['train']['saved_weights'].split('.')

    print('Convert Inference Model')
    os.system('keras_retinanet/bin/convert_model.py ' + config['train']['saved_weights'] + ' ' + aux_weights[0] + '_infer.h5')


    print('Evaluate')
    os.system('python keras_retinanet/bin/evaluate.py  --backbone ' + config['model']['backend']+
                ' --image-min-side ' + config['model']['image_min_side'] +
                ' --image-max-side ' + config['model']['image_max_side'] +
                ' --iou-threshold 0.5 --score-threshold 0.5 --model ' + aux_weights[0] + '_infer.h5' +
                ' csv ' + csv_anns_test + ' ' + csv_classes_test)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate ssd model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()

    main(args)
