from frontend import YOLO
from keras.models import load_model,save_model
import json

config_path = 'config.json'

with open(config_path) as config_buffer:
        config = json.load(config_buffer)

weights_path = 'experimento_0_cpu_yolo2.h5'

yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

yolo.load_weights(weights_path)
#   Load trained weights
###############################


#yolo.load_weights(weights_path)
#infer_model = load_model(weights_path)

