#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:37:58 2018

@author: dlsaavedra
"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2 as cv
import os
import pandas
import xml.etree.cElementTree as ET


def sum_image(Img, Img_obj, x, y):
    ## Función suma la imagen objeto a la imagen original desde el lugar x,y

    x = np.int16(x)
    y = np.int16(y)
    B = 0; C = 255 # Parametros del método
    Img_new = Img.copy()
    y_lim = min( max(0, y + Img_obj.shape[0]), Img.shape[0])
    x_lim = min( max(0, x + Img_obj.shape[1]), Img.shape[1])

    y_obj = min(Img.shape[0] - y, Img_obj.shape[0])
    x_obj = min(Img.shape[1] - x, Img_obj.shape[1])

    Img_new[max(0, y) : y_lim, max(0, x) : x_lim] = np.exp(np.log((Img[max(0, y) : y_lim, max(0, x) : x_lim] - B)/C) + np.log((Img_obj[ max(-y, 0): y_obj, max(-x, 0) : x_obj]- B)/C)) * C + B

    return Img_new

def crop_image(Img):
    # Recibe un imagen y la binariza y cropea el arma

    # Otsu's thresholding after Gaussian filtering
    ret3,th3 = cv.threshold(cv.GaussianBlur(Img,(5,5),0) ,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #Im_logic = th3 == 0 # pone blanco el shuriken

    #Crop
    mask = th3 == 0
    y_0, x_0 = mask.any(1).argmax(), mask.any(0).argmax()
    y_fin, x_fin = Img.shape[0]-np.flipud(mask.any(1)).argmax(), Img.shape[1]-np.flipud(mask.any(0)).argmax()


    Img_crop = Img[y_0 : y_fin, x_0 : x_fin].copy()
    th3 = th3[y_0 : y_fin, x_0 : x_fin]
    Img_crop[th3 > 0] = 255

    return Img_crop

def create_ann_xml(path, boxes, size_baggage):
    #Transforma un string de la forma box1 box2 box3
    #donde box1 = x_min, y_min, x_max, y_max, obj
    # a un archivo xml en VOC format

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = 'Training'
    ET.SubElement(root, "filename").text = name
    ET.SubElement(root, "path").text = path_save_train + 'images/' + name
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = 'Unknown'
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(size_baggage)
    ET.SubElement(size, "height").text = str(size_baggage)
    ET.SubElement(size, "depth").text = str(1)
    ET.SubElement(root, "segmented").text = '0'

    for box in boxes.split(' '):

        if box != '':
            box = box.split(',')
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = box[4]
            ET.SubElement(obj, "pose").text = 'Unspecified'
            ET.SubElement(obj, "truncated").text = str(0)
            ET.SubElement(obj, "difficult").text = str(0)
            bx = ET.SubElement(obj, "bndbox")
            ET.SubElement(bx, "xmin").text = box[0]
            ET.SubElement(bx, "ymin").text = box[1]
            ET.SubElement(bx, "xmax").text = box[2]
            ET.SubElement(bx, "ymax").text = box[3]

    tree = ET.ElementTree(root)
    tree.write(path + '.xml')

#%%

name_experiment = 'Experiment_1'

data_path = 'Data/'
path_save = os.getcwd() + '/' + name_experiment + '/'
#path_save_train = os.getcwd() + '/'+ name_experiment + '/Training_2/'


size_baggage = 448 # size of train images
margen = 30 #edge margin

num_image_per_baggage = 2 # numeber of images to create
    #total images =  49*(num_image_per_baggages) = 98
max_num_obj = 4 # maxima cantidad de objetos por imagen

type_object = {0: 'Razor' , 1: 'Gun', 2: 'Knife', 3: 'Shuriken'}
# Razor : 0
# Gun : 1
# Knife : 2
# Shuriken : 3

scales_obj = {'Razor': [0.07, 0.11 ,0.15],
          'Gun': [0.4, 0.5 ,0.6, 0.7],
          'Knife': [0.4, 0.5 ,0.6, 0.7],#[0.3, 0.35 ,0.4],
          'Shuriken': [0.4, 0.5 ,0.6, 0.7],
          'Other': [0.1, 0.125,0.15]}  # escala de objeto con referencia al tamaño del empaque

## Training
folders_obj = {'Razor': ['B0051'],#, 'B0007'], #B0007: 20 img #B0051: 100 img
          'Gun': ['B0049'],                 #B0049: 200 img
          'Knife': ['B0076'],#'B0008'],       #B0008: 361 img #B0076: 576 img
          'Shuriken': ['B0052']} #,'B0050']}   #B0050: 100 img #B0052: 144 img
          #cantidad total 1501 img
          #cantidad parcial
folders_other = ['B0082']
path_other = []
#diccionario de listas con path de los objetos
path_objects = {'Razor': [],
          'Gun': [],
          'Knife': [],
          'Shuriken': []}



folders_baggages = ["BX_100"]#['B0046']#['B0044', 'B0045', 'B0046', 'B0047']


## Crear diccionario de listas con path de los objetos
for k in folders_obj.keys():

    for folder_training in folders_obj[k]:

        list_dir = os.listdir(data_path + folder_training)
        list_dir = list(filter(lambda x : x[-3:]=='png', list_dir))
        for l in range(len(list_dir)): list_dir[l] = data_path  + folder_training + '/' + list_dir[l]
        path_objects[k] += list_dir.copy()

for other in folders_other:
    list_dir = os.listdir(data_path + other)
    list_dir = list(filter(lambda x : x[-3:]=='png', list_dir))
    for l in range(len(list_dir)): list_dir[l] = data_path  + other + '/' + list_dir[l]
    path_other += list_dir.copy()

#Creación de carpetas
try:
    os.makedirs(path_save)
    os.makedirs(path_save + 'Training/')
    os.makedirs(path_save + 'Training/images') ## Carpeta de imagenes
    os.makedirs(path_save + 'Training/anns')  ## Carpeta de anotaciones
    os.makedirs(path_save + 'Validation/')
    os.makedirs(path_save + 'Validation/images') ## Carpeta de imagenes
    os.makedirs(path_save + 'Validation/anns')  ## Carpeta de anotaciones

except OSError:
    pass


## Iteracion sobre las carpetas de Training
for train_val in ['Training/', 'Validation/']:
    path_save_train = path_save + train_val
    image_set_filenames = []
    num_objects = {'Razor': 0,
              'Gun': 0,
              'Knife': 0,
              'Shuriken': 0}
              
    for folder_b in folders_baggages:

        ## Iteracion sobre las imagen de la carpeta
        list_path_img = os.listdir(data_path + folder_b)
        list_path_img.sort()

        for path_img in list_path_img:

            print(path_img)

            if path_img[-3:]=='png':

                real_size = (imread(data_path + folder_b + '/' + path_img)).shape
                Img_B = np.uint8(resize(imread(data_path + folder_b + '/' + path_img),(size_baggage, size_baggage), preserve_range = True))
                if len(Img_B.shape) == 3:
                    Img_B = Img_B[:,:,0]


                ### Leer bounding box previo de la imagen

                ## Guardar solo las imagenes
                #image_set_filenames.append(path_img[:-4])
                #name = path_img[:-4] + '.png'
                #plt.imsave(path_save_train + 'images/' + name, Img_B, cmap = 'gray')
                #create_ann_xml(path_save_train + 'anns/' + name[:-4], '', size_baggage)

                for i in range(num_image_per_baggage):

                    Im_F = np.copy(Img_B)

                    string_box = ' ' #string que dice las coordenadas y tipo de objeto en el box. cada 5 elementos es una caja

                    ##Random para ver la cantidad de objetos a pegar

                    num_obj = np.random.randint(1,max_num_obj + 1) # cantidad de objetos a pegar en la Imagen
                    #print(num_obj)
                    objects = np.random.randint(0, len(type_object) , num_obj) # tipos de objetos que se van a pegar

                    num_obj_others = np.random.randint(1,max_num_obj*2 + 1) # cantidad de objetos a pegar en la Imagen

                    # Pegar los objetos
                    coord_list = []
                    for obj in objects:


                        name_obj = type_object[obj]
                        num_objects[name_obj] += 1 #cuenta una anotación del objeto

                        scale = np.random.choice(scales_obj[name_obj])
                        size_obj = int(size_baggage * scale)
                        x_rd, y_rd = np.random.randint(size_baggage-size_obj, size = 2)
                        # Asegurar que las posiciones no queden cercas.(cerca < margen)
                        if coord_list == []:
                            coord_list.append([x_rd,y_rd])
                        else:
                            while min(list(map(lambda x: (x[0]-x_rd)**2 + (x[1]-y_rd)**2, coord_list))) < margen**2:
                                x_rd, y_rd = np.random.randint(size_baggage-size_obj, size = 2)
                            coord_list.append([x_rd,y_rd])


                        path_obj = np.random.choice(path_objects[name_obj])

                        Im_C = crop_image(imread(path_obj))
                        prop = max(Im_C.shape[0],Im_C.shape[1])
                        Img_O = np.uint8(resize(Im_C, (int(size_obj * Im_C.shape[0] / prop) ,int(size_obj * Im_C.shape[1] / prop)), preserve_range = True))

                        new_box =  np.int16(( x_rd  ,  y_rd , min(size_baggage,   x_rd  + Img_O.shape[1]) , min(size_baggage, y_rd  + Img_O.shape[0]) ))

                        # string es xmin1,ymin1,xmax1,ymax1,class1 xmin2,ymin2,xmax2,ymax2,class2

                        string_box  = string_box + str(new_box[0]) + ',' + str(new_box[1]) + ',' + str(new_box[2]) + ',' + str(new_box[3]) + ',' + name_obj + ' '

                        Im_F = sum_image(Im_F, Img_O, new_box[0], new_box[1])

                    #Pegar los objetos Otros (Ruido)
                    for obj in range(num_obj_others):

                        scale = np.random.choice(scales_obj['Other'])
                        size_obj = int(size_baggage * scale)
                        x_rd, y_rd = np.random.randint(size_baggage-size_obj, size = 2)

                        path_obj = np.random.choice(path_other)
                        Im_C = crop_image(imread(path_obj))
                        prop = max(Im_C.shape[0],Im_C.shape[1])
                        Img_O = np.uint8(resize(Im_C, (int(size_obj * Im_C.shape[0] / prop) ,int(size_obj * Im_C.shape[1] / prop)), preserve_range = True))

                        new_box =  np.int16(( x_rd  ,  y_rd , min(size_baggage,   x_rd  + Img_O.shape[1]) , min(size_baggage, y_rd  + Img_O.shape[0]) ))

                        Im_F = sum_image(Im_F, Img_O, new_box[0], new_box[1])


                    image_set_filenames.append(path_img[:-4] + '_' + str(i))
                    name = path_img[:-4] + '_' + str(i) + '.png'
                    plt.imsave(path_save_train + 'images/' + name, Im_F, cmap = 'gray')
                    create_ann_xml(path_save_train + 'anns/' + name[:-4], string_box[1:-1], size_baggage)

    # ## Numero de anotaciones

    f = open( path_save_train + 'num_annotation.txt','w')
    print(num_objects, file= f)

    ####
    if train_val == 'Training/':
        with open(path_save_train + 'train.txt', 'w') as f:
            for item in image_set_filenames:
                f.write("%s\n" % item)

    elif train_val == 'Validation/':
        with open(path_save_train + 'val.txt', 'w') as f:
            for item in image_set_filenames:
                f.write("%s\n" % item)
