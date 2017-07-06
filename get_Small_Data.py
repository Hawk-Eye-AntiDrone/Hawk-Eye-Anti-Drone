import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2


# get the images from the small data folder 
###### Change Path here
image_path = 'small_data' 
allfiles = [ f for f in listdir(image_path) if isfile(join(image_path,f)) ]

# scan through the folder to make sure that are images

imagefiles = []
for image in allfiles:
	if image[-3:] == 'jpg':
		imagefiles.append(image)

# 
height = 600
width = 800
new_size = 512


# for each image file, get its x, y, w, h
groundtruth_matrix = np.zeros((len(imagefiles), 4))
image_matrix = np.zeros((len(imagefiles),new_size,new_size,3))

########### Change label directory
# define path directory of the groundtruth label
rel_path = "groundtruth/folder1.xml" # directory of the groundtruth label folder
abs_file_path = os.path.join(rel_path) # absolute director of the groundtruth files
tree = ET.parse(abs_file_path)
root = tree.getroot()


# define output label matrix
index = 0

for frame in root:
	filename =  frame.attrib['number'] + '.jpg'
	if filename in imagefiles:
		fullpath_name = join(image_path,filename)
	    
		for person in frame:
			left_eye = person[0]
			right_eye = person[1]

			left_x = float(left_eye.attrib['x'])*new_size/width
			right_x = float(right_eye.attrib['x'])*new_size/width

			left_y = float(left_eye.attrib['y'])*new_size/height
			right_y = float(right_eye.attrib['y'])*new_size/height

			groundtruth_matrix[index, 0] = (left_x + right_x)/2 
			groundtruth_matrix[index, 1] = (left_y + right_y)/2 
			eye_distance = np.sqrt((float(left_x) - float(right_x))**2 + (float(left_y) - float(right_y))**2)
			groundtruth_matrix[index, 2] = 3.0 * eye_distance # bounding box width is 3 times the eye width
			groundtruth_matrix[index, 3] = 5.0 * eye_distance # bounding box height is 5 times the eye width
			
		# draw the rectangle in teh picture to double check 
		#print fullpath_name
		x = int(groundtruth_matrix[index, 0])
		y = int(groundtruth_matrix[index, 1])
		w = int(groundtruth_matrix[index, 2])
		h = int(groundtruth_matrix[index, 3])

		image = cv2.imread(fullpath_name)
		resized_image = cv2.resize(image, (new_size, new_size))
		image_matrix[index,:,:,:] = resized_image
		index +=1

		
		cv2.rectangle(resized_image, (x - w/2, y - h/2), (x + w/2, y + h/2), (0, 255, 0), 2)
		cv2.imshow('Image', resized_image)
        cv2.waitKey(10)
        
print groundtruth_matrix

		