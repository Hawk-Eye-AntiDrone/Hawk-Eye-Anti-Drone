import tensorflow as tf
import numpy as np
import datetime

import math
import timeit
#import matplotlib.pyplot as plt
import cv2
#matplotlib inline
import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import random



height = 600
width = 800
new_size = 512

num_folders = 3
groundtruth_files = []

# 
groundtruth_dict = {}


for ground_filename in groundtruth_files:
    valid_image_list = []
    
    # image 'xxx.jpg'
    # groundtruth_folder_name 'XOXO.xml'
    tree = ET.parse( 'groundtruth/' + ground_filename)
    root = tree.getroot()

    # Data Parsing

    # define output label matrix
    
    for frame in root:
        filename =  frame.attrib['number'] + '.jpg'
        if (len(frame) != 0):  # has person in the frame
            valid_image_list.append(filename)
        elif (random.random() < 0.03):
            valid_image_list.append(filename)

            
    groundtruth_dict[ground_filename] = valid_image_list

#print groundtruth_dict







