import tensorflow as tf
import numpy as np
import datetime
import math
import timeit
import cv2
import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import random
import timeit

##################################################################################################
# Naming Convention: each folder name is "XXX"
# in the foler groundtruth, each "XXX.txt" corresponds to the groundtruth in XXX flder
###################################################################################################

####################################################################################
tf.reset_default_graph()  #Always put it at the beginning
####################################################################################


# define parameters 
# Original height and weight
height = 1080
width = 1920
# new_size of height and weight after resize
new_size = 1024
# number of class
num_class = 4

# class num: meaning
# 0: None, 1: airplane, 2: drone, 3: bird, 4: others

# there are numbers of folders; in each folder, there are images  
num_folders = 1
folder_name = ['folder1']


# With a folder name as input, output a list of string corresponded to the image name
# in the folder as well as the number of image.
def get_list_filenames(nameOfFolder):
    outputList = []
    count = 0

    for filename in os.listdir(nameOfFolder):
        if filename[-3:] == 'jpg':
            outputList.append(filename)
            count = count + 1
    return np.array(outputList), count # numpy array is better than a python list since it has better index access


# randomly pick an image folder; from that folder, randomly pick batch_size of image
# output the resized image as well as its groundtruth values
def get_data(batch_size=16):

    # choose a random folder 
    folder_idx = random.randint(0, num_folders-1)
    groundtruth_file_name = 'groundtruth/' + folder_name[folder_idx] + '.txt'
    image_folder_name = folder_name[folder_idx]
    full_image_list, num_images = get_list_filenames(image_folder_name)

    # get the batch_size of images from the image folder 
    index_of_images = np.random.choice(num_images, batch_size)
    image_list_batch = full_image_list[index_of_images]

    # for each image file, get its x, y, w, h, class_ID
    full_groundtruth_matrix = np.loadtxt(groundtruth_file_name)
    groundtruth_matrix = np.ones((batch_size, 5))
    # use ones because assuming all picturs have airplane class
    image_matrix = np.zeros((batch_size,new_size,new_size,3))

    # get the x,y,w,h from the full_groundtruth_matrix
    groundtruth_matrix[:, :4] = full_groundtruth_matrix[index_of_images]

    # iterate over all chosen images, resize, and put them into image matrix
    image_matrix = np.zeros((batch_size,new_size,new_size,3))
    for index in range(batch_size):
        fullpath_name = join(image_folder_name, image_list_batch[index])
        image = cv2.imread(fullpath_name)
        resized_image = cv2.resize(image, (new_size, new_size))
        image_matrix[index,:,:,:] = resized_image

    return groundtruth_matrix, image_matrix    

# define our loss layer
def loss_layer(coord_predicts, class_predicts, ground_truth, lam_wh=.5, lam_class=2500., scope='loss_layer'):
    # coord_predicts batch_size x 4
    # class_predicts batch_size x 1
    # ground_truth = batch_size x 5

    with tf.variable_scope(scope):
        #x1h,y1h,w1h,h1h,cf1h = ground_truth[:,:,:,:5]
        xh = tf.cast(ground_truth[:,0], tf.float32)
        yh = tf.cast(ground_truth[:,1], tf.float32)
        wh = tf.cast(ground_truth[:,2], tf.float32)
        hh = tf.cast(ground_truth[:,3], tf.float32)
        
        x = tf.cast(coord_predicts[:,0], tf.float32)
        y = tf.cast(coord_predicts[:,1], tf.float32)
        w = tf.cast(coord_predicts[:,2], tf.float32)
        h = tf.cast(coord_predicts[:,3], tf.float32)
        class_ground_truth = tf.cast(ground_truth[:,4], tf.uint8)
        
        mask = tf.cast(ground_truth[:,4]>0, tf.float32)
        none_mask = tf.cast(ground_truth[:,4]==0, tf.float32)

        xy_loss = tf.square(x-xh)+tf.square(y-yh)
        xy_loss = tf.multiply(xy_loss, mask) # not penalize x y with no objects
        wh_loss = tf.square(w-wh)+tf.square(h-hh)
        coord_loss = tf.reduce_mean(xy_loss + lam_wh*wh_loss)
        '''
        class_none_loss = lam_class*tf.reduce_mean(tf.multiply(none_mask, tf.losses.softmax_cross_entropy(tf.one_hot(class_ground_truth,26), class_predicts)))
        class_yes_loss = lam_class*tf.reduce_mean(tf.multiply(mask, tf.losses.softmax_cross_entropy(tf.one_hot(class_ground_truth,26), class_predicts)))
        class_loss = class_yes_loss + class_none_loss
        '''
        class_loss = lam_class*tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(class_ground_truth,num_class), class_predicts))
        return coord_loss + class_loss # only coordinate loss, ignore the error in width and height for now


# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, new_size, new_size, 3])
y = tf.placeholder(tf.float32, [None, 5])
is_training = tf.placeholder(tf.bool)

def leaky_relu(inputs, alpha=0.1):
    return tf.maximum(alpha*inputs, inputs)

def simple_model(X,y, is_training):
    conv0 = tf.layers.conv2d(X, 32, 3, padding='SAME', activation=leaky_relu)
    maxpool1 = tf.layers.max_pooling2d(conv0, 2, 2) #512,32
    
    conv2 = tf.layers.conv2d(maxpool1, 64, 3, padding='SAME', activation=leaky_relu)
    maxpool3 = tf.layers.max_pooling2d(conv2, 2, 2) #256,64

    conv4 = tf.layers.conv2d(maxpool3, 128, 3, padding='SAME', activation=leaky_relu)
    maxpool5 = tf.layers.max_pooling2d(conv4, 2, 2) #128, 128
    
    conv6 = tf.layers.conv2d(maxpool5, 64, 3, padding='SAME', activation=leaky_relu)
    maxpool7 = tf.layers.max_pooling2d(conv6, 2, 2) #64,64

    conv8 = tf.layers.conv2d(maxpool7, 32, 3, padding='SAME', activation=leaky_relu)
    maxpool9 = tf.layers.max_pooling2d(conv8, 2, 2) #32,32 

    conv10 = tf.layers.conv2d(maxpool9, 16, 3, padding='SAME', activation=leaky_relu)
    maxpool11 = tf.layers.max_pooling2d(conv10, 2, 2) #16,16 

    conv12 = tf.layers.conv2d(maxpool11, 8, 3, padding='SAME', activation=leaky_relu)
    maxpool13 = tf.layers.max_pooling2d(conv12, 2, 2) #8,8 

    pool_flat = tf.reshape(maxpool13, [-1, 8*8*8]) # 8*8*8

    # Prediction layer for coordinate 
    corrdiante_dense1 = tf.layers.dense(inputs=pool_flat, units=64, activation=leaky_relu)
    corrdiante_dense2 = tf.layers.dense(inputs=corrdiante_dense1, units=4)

    # Prediction layer for class error 
    class_dense1 = tf.layers.dense(inputs=pool_flat, units=32, activation=leaky_relu)
    class_dense2 = tf.layers.dense(inputs=class_dense1, units=num_class) # scores of every class

    # loss layer
    total_loss = loss_layer(corrdiante_dense2, class_dense2, y)    
    mean_loss = tf.reduce_mean(total_loss)

    return corrdiante_dense2, class_dense2, mean_loss


coordinate_predict, class_predict, mean_loss = simple_model(X,y, is_training)  

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-3) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)


# run the network session
def run_model(session, coordinate_predict, class_predict, loss_val, Xd, yd,
              epochs=1, batch_size=16, print_every=100,
              training=None, plot_losses=False):
    
    # compute accuracy
    correct_prediction = tf.equal(tf.argmax(class_predict,1), tf.cast(y[:,4], tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    mask_true = tf.equal(tf.cast(y[:,4], tf.int64), 0)
    mask_pred = tf.equal(tf.cast(tf.argmax(class_predict,1), tf.int64), 0)
    TF_precision = tf.reduce_mean(tf.cast(tf.equal(mask_true,mask_pred), tf.float32))
    
    mask = tf.cast(tf.greater(tf.cast(y[:,4], tf.int64), 0), tf.float32)

    temp = tf.multiply(tf.abs(coordinate_predict[:,0]-y[:,0]), tf.abs(coordinate_predict[:,0]-y[:,0])) + tf.multiply(tf.abs(coordinate_predict[:,1]-y[:,1]), tf.abs(coordinate_predict[:,1]-y[:,1]))
    center_deviation = tf.reduce_sum(tf.multiply(mask, tf.sqrt(temp)))/tf.reduce_sum(mask)
    
    groundtruth_upper_left_x, groundtruth_upper_left_y = y[:,0]-0.5*y[:,2], y[:,1]-0.5*y[:,3]
    groundtruth_bottom_right_x, groundtruth_bottom_right_y = y[:,0]+0.5*y[:,2], y[:,1]+0.5*y[:,3]
    predict_upper_left_x, predict_upper_left_y = coordinate_predict[:,0]-0.5*coordinate_predict[:,2], coordinate_predict[:,1]-0.5*coordinate_predict[:,3]
    predict_bottom_right_x, predict_bottom_right_y = coordinate_predict[:,0]+0.5*coordinate_predict[:,2], coordinate_predict[:,1]+0.5*coordinate_predict[:,3]
    
    upper_left_x, upper_left_y = tf.maximum(groundtruth_upper_left_x,predict_upper_left_x), tf.maximum(groundtruth_upper_left_y,predict_upper_left_y)
    bottom_right_x, bottom_right_y = tf.minimum(groundtruth_bottom_right_x,predict_bottom_right_x), tf.minimum(groundtruth_bottom_right_y,predict_bottom_right_y)
    groundtruth_area = tf.multiply(y[:,2], y[:,3])
    predict_area = tf.multiply(coordinate_predict[:,2], coordinate_predict[:,3])
    intersection = tf.multiply(bottom_right_x-upper_left_x, bottom_right_y-upper_left_y)
    IOU = tf.reduce_sum(tf.multiply(mask, tf.div(intersection, groundtruth_area+predict_area-intersection+1e-6)))/tf.reduce_sum(mask)
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    training_now = training is not None

    #  when in training, it will be [loss_val, predict, training]
    # in testing, it will just be [loss_val, predict, predict]  
    variables = [loss_val, coordinate_predict, class_predict, accuracy, TF_precision, center_deviation, IOU, accuracy] 
    
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0

    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx],
                         y: yd[idx],
                         is_training: training_now}

            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step

            #loss, coordinate_Predictions, class_Predictions, accuracy_result, TF_precisions, X_errors,Y_errors,W_errors,H_errors, _ = session.run(variables,feed_dict=feed_dict)
            loss, coordinate_Predictions, class_Predictions, accuracy_result, TF_precisions, center_deviations, IOUs, _ = session.run(variables,feed_dict=feed_dict)
        
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Minibatch training loss = ", loss)
                print ('accuracy = ', accuracy_result)
                print('TF_precisions = ', TF_precisions)
                print('center deviation = ', center_deviations)
                print('IOU = ', IOUs)
                #print('X error = ', X_errors)
                #print('Y error = ', Y_errors)
                #print('W error = ', W_errors)
                #print('H error = ', H_errors)
            iter_cnt += 1

            A = coordinate_Predictions
            B = np.argmax(class_Predictions, axis = 1).T.reshape((batch_size,1))
            #print (np.hstack((A,B)))
            
        total_loss = np.sum(losses)/Xd.shape[0]

    return coordinate_Predictions, class_Predictions, accuracy_result, TF_precisions, center_deviations, IOUs, total_loss

loss_list = []

# 
train_batch_size = 2
test_batch_size = 2

with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        sess.run(tf.global_variables_initializer())
        print('Training')
        
        # delete this line for actual training
        y_train, X_train = get_data(batch_size=train_batch_size)

        for iter_index in range(150):
            print('================================')
            print('iter: ',iter_index)
            # uncommin this line for actual training
            #y_train, X_train = get_data(batch_size=train_batch_size)
            coordinate_output, class_output, _,_,_,_, epoch_loss = run_model(sess,coordinate_predict, class_predict,mean_loss,X_train,y_train, 1,train_batch_size,1,train_step)
            print('coordinate_output', coordinate_output)
            print('class_output', class_output)

            loss_list.append(epoch_loss)

        # save the weights
        variable_to_restore = tf.global_variables()
        saver = tf.train.Saver(variable_to_restore, max_to_keep=None)
        output_dir = os.path.join(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        ckpt_file = 'save.ckpt'
        global_step = tf.get_variable( 'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        saver.save(sess, ckpt_file)
        
        print('Test')
        y_test, X_test = get_data(batch_size=test_batch_size) # Change 16!!!!!!!


        #########################################################
        # use the same test and training for over fit
        y_test, X_test = y_train, X_train

        #########################################################

        #print(y_test)
        start = timeit.timeit()
        coordinate_output, class_output, accuracy_result, TF_precisions, center_deviations, IOUs, _ = run_model(sess,coordinate_predict, class_predict,mean_loss,X_test,y_test,1,test_batch_size,1) # Change 16!!!!!!!
        end = timeit.timeit()
        print('time: ', end-start)
        #print(np.hstack((coordinate_output, class_output)))     

        print('coordinate_output', coordinate_output)
        print('class_output', class_output)
        print('y', y_test)

        print('accuracy = ', accuracy_result)
        print('TF_precision = ', TF_precisions)
        print('center deviation = ', center_deviations)
        print('IOU = ', IOUs)
#print(loss_list)
    
