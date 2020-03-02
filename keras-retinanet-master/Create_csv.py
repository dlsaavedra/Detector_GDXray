#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:52:39 2019

@author: dlsaavedra
"""

import csv
import os
import xml.etree.cElementTree as ET
import argparse
import sys

#name_experiment = 'Test_B0046'#'Experimento_3'
#path_anns  = '../Experimento_3/Baggages/Testing/anns/'#''../'+ name_experiment +'/Training/anns/'




def _main_(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]

    name_experiment = args.path_save
    path_anns = args.path_anns
    labels = args.labels
    labels = labels.split("-")
    #print(labels)
    type_object = { i : labels[i] for i in range(0, len(labels) )}
    classes = { labels[i] : i for i in range(0, len(labels) )}

    #type_object = {0: 'Razor' , 1: 'Gun', 2: 'Knife', 3: 'Shuriken'}
    #classes = {'Razor': 0, 'Gun': 1, 'Knife': 2, 'Shuriken': 3}
    #%%
    with open(name_experiment + '_anns.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for xml in list(filter(lambda x : x[-3:]=='xml', os.listdir(path_anns))):



            tree = ET.parse(path_anns + xml)
            root = tree.getroot()

            path_image = '/'.join(path_anns.split('/')[1:-2]) #root[2].text
            path_image += '/images/' + xml[:-3] + 'png'

            if len(root) > 6:
                for child in root[6:]:

                    label = child[0].text
                    x_1 = child[4][0].text
                    y_1 = child[4][1].text
                    x_2 = child[4][2].text
                    y_2 = child[4][3].text

                    employee_writer.writerow([path_image, x_1, y_1, x_2, y_2, label])
            else:
                employee_writer.writerow([path_image,None,None,None,None,''])

    with open(name_experiment + '_class.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for key in classes.keys():
            employee_writer.writerow([key, classes[key]])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path_anns', type=str, help='path annotations')
    parser.add_argument('--path_save', type=str, help='path save csv')
    parser.add_argument('--labels', type=str, help='path save csv')
    args = parser.parse_args()

    _main_(args)
