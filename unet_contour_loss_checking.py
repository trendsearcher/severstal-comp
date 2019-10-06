# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:52:59 2019

@author: user_PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import  SGD
import cv2
import tensorflow.keras
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D, Activation
#from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from PIL import Image 
from tensorflow.keras.layers import Input,  Convolution2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Conv2DTranspose, concatenate
smooth = K.epsilon()#1
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import random
from tensorflow.keras.models import Model

df_path = 'C:\\Users\\user_PC\\Desktop\\severstal\\train.csv'
img_path = 'C:\\Users\\user_PC\\Desktop\\severstal\\input\\train_images\\'
weights_vgg16_path = 'C:\\Users\\user_PC\\Desktop\\severstal\\vgg16_weights_tf.h5'
###############################################################################
rand_state = 1
train_val_ratio = 0.8
img_sizex = 256
img_sizey = 256
epochss = 160
dice = [0]*epochss
epoch_list = list(range(epochss))
val_dice = [0]*epochss
loss = [0]*epochss
val_loss = [0]*epochss
iterations = 1
##############################################################################
tr = pd.read_csv(df_path)
# Only ClassId=4
df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
df_4type = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')].reset_index(drop=True)


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )

def fetch_photo_lables_sets (df, rand_state, train_val_ratio):
    
    names = []
    originals = []
    masks = []
    
    for i in range(len(df)):       
        fn = df['ImageId_ClassId'].iloc[i].split('_')[0]
        img = cv2.imread(img_path + fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        mask = rle2mask(df['EncodedPixels'].iloc[i], img.shape)
        img = cv2.resize(img, (img_sizex, img_sizey))
        img = img.reshape(img_sizex, img_sizey, 1)
        mask = cv2.resize(mask, (img_sizex, img_sizey))
        
        names.append(fn)
        masks += [mask]
        originals += [img]
    
    
    originals = np.array(originals)
    masks = np.array(masks)
    
    return names, originals, np.expand_dims(masks, -1)

###############################################################################
''' AUGMENTATION'''
def vertical_flip(image, fmap):
    'вертикальное зеркальное отображение'    
    image = image[::-1, :, :]
    fmap = fmap[::-1, :, :]
    return image, fmap

def image_translation(img, fmap): 
    'cмещения параллельным переносом '
    params = np.random.randint(-50, 51)
    if not isinstance(params, list):
        params = [params, params]
    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    fmap = cv2.warpAffine(fmap, M, (cols, rows))
    return np.expand_dims(dst, axis=-1), fmap


def image_stright_rotation(img, fmap):
    'вертикальный поворот'
    params = 180#np.random().randint(-50, 50)
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    fmap = cv2.warpAffine(fmap, M, (cols, rows))
    return np.expand_dims(dst, axis=-1), fmap

def image_slight_rotation(img, fmap):
    'незначительное отклонение на +-5 градусов'
    params = np.random.randint(-5, 6)
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    fmap = cv2.warpAffine(fmap, M, (cols, rows))
    return np.expand_dims(dst, axis=-1), fmap

def image_contrast(img):
    'изм котнраста на +-30%'
    params = np.random.randint(7, 13)*0.1
    new_img = cv2.multiply(img, np.array([params]))                    # mul_img = img*alpha
    return np.expand_dims(new_img, axis=-1)

def image_blur(img):
    'наложение тумана'
    params = params = np.random.randint(1, 11)
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
            
    return blur.reshape([blur.shape[0],blur.shape[1],1])
###############################################################################
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)#relu
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)#relu
    return x
def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#print(np.shape(train_origin[0]))
#plt.imshow(np.squeeze(train_origin[0]))  
#plt.pause(0.05)
#plt.imshow(np.squeeze(train_masks[0])) 
#plt.pause(0.05)
#print('-------------')
#print(np.shape(image_translation(train_origin[0])))
#plt.imshow(np.squeeze(image_translation(train_origin[0])))  
#plt.pause(0.05)
#plt.imshow(np.squeeze(image_slight_rotation(train_masks[0]))) 
#plt.pause(0.05)
    

sobel_operator_y = K.constant([-1, -2, -1, 0, 0, 0, 1, 2, 1], shape = [3, 3,1, 1])
sobel_operator_x = K.constant([-1, 0, 1, -2, 0, 2, -1, 0, 1], shape = [3, 3,1, 1])

def Active_Contour_Loss(y_true, y_pred):
    coef = 1
    
    sober_y_len = K.sum(K.relu(K.conv2d(y_pred, sobel_operator_y, padding='same')))
    sober_x_len = K.sum(K.relu(K.conv2d(y_pred, sobel_operator_x, padding='same')))
    sober_len = sober_y_len + sober_x_len
    area = K.sum(y_pred) + K.epsilon()
    shape_metric = sober_len/ area
    
#    shape_metric = tensorflow.keras.backend.print_tensor(shape_metric)
    binary_cross_loss = tf.keras.losses.binary_crossentropy(y_pred, y_true)

    return binary_cross_loss + coef*shape_metric


class Custom_Generator(tensorflow.keras.utils.Sequence) :
  
    def __init__(self, image_filenames, image, labels, batch_size, mode) :
        self.image_filenames = image_filenames
        self.image = image
        
        self.labels = labels
        self.batch_size = batch_size
        self.mode = mode
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
    def __getitem__(self, idx) :
        batch_x = self.image[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        if self.mode == 'train': # рандомная аугментация только на трейне
            gambling = random.choice([1,2,3,4,5,6,7])#np.random.randint(7, 8)
            if gambling <=6: # делаем аугментацию
                batch_x = batch_x.reshape(img_sizex, img_sizey, 1)
                if gambling <= 4:
                    batch_y = batch_y.reshape(img_sizex, img_sizey, 1)
                if gambling == 1:
                    batch_x, batch_y = vertical_flip(batch_x, batch_y)
                    batch_x = batch_x.reshape(1, img_sizex, img_sizey, 1)
                    batch_y = batch_y.reshape(1, img_sizex, img_sizey, 1)
                if  gambling == 2:  
                    batch_x, batch_y = image_translation(batch_x, batch_y)
                    batch_x = batch_x.reshape(1, img_sizex, img_sizey, 1)
                    batch_y = batch_y.reshape(1, img_sizex, img_sizey, 1)
                if  gambling == 3:  
                    batch_x, batch_y = image_stright_rotation(batch_x, batch_y)
                    batch_x = batch_x.reshape(1, img_sizex, img_sizey, 1)
                    batch_y = batch_y.reshape(1, img_sizex, img_sizey, 1)
                if  gambling == 4:  
                    batch_x, batch_y = image_slight_rotation(batch_x, batch_y)
                    batch_x = batch_x.reshape(1, img_sizex, img_sizey, 1)
                    batch_y = batch_y.reshape(1, img_sizex, img_sizey, 1)        
                if  gambling == 5:  
                    batch_x = image_contrast(batch_x)
                    batch_x = batch_x.reshape(1, img_sizex, img_sizey, 1)
                if  gambling == 6:  
                    batch_x = image_blur(batch_x)
                    batch_x = batch_x.reshape(1, img_sizex, img_sizey, 1)
                    
        return batch_x, batch_y
# COMPETITION METRIC
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
##################################################################################################    
batch_size = 1 # сделал все только для батча = 1 аугмент на лету
for it in range(iterations):
    working_df, val_df = train_test_split(df_4type, test_size=0.2, random_state= it)
    
    val_names, val_origin, val_masks = fetch_photo_lables_sets(val_df, rand_state, train_val_ratio)
    train_names, train_origin, train_masks = fetch_photo_lables_sets(working_df, rand_state, train_val_ratio)
    
    validation_batch_generator = Custom_Generator(val_names, val_origin, val_masks, batch_size, 'val')
    training_batch_generator = Custom_Generator(train_names, train_origin, train_masks, batch_size, 'train')        
    ############################################################################################
    model = get_unet(Input((img_sizex, img_sizey, 1), name='img'))
    #model = load_model('C:\\Users\\user_PC\\Desktop\\severstal\\keras_model2.h5')
    #model.load_weights('C:\\Users\\user_PC\\Desktop\\severstal\\keras_model_weights3.h5')
    model.compile(tf.keras.optimizers.Adam(lr=0.0001), loss= 'binary_crossentropy', metrics=[dice_coef])#'binary_crossentropy' tf.keras.optimizers.Adam(lr=0.0001)
    print('==================================================================')
    history = model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch=(len(training_batch_generator)// batch_size),
                        epochs=epochss,
                        verbose=1,
                        validation_data = validation_batch_generator,
                        validation_steps=1)
    
    #model.save('C:\\Users\\user_PC\\Desktop\\severstal\\keras_model2.h5')
    #model.save_weights('C:\\Users\\user_PC\\Desktop\\severstal\\keras_model_weights2.h5')
    
    metric_df_path = 'C:\\Users\\user_PC\\Desktop\\severstal\\metric_df_'+str(it)+'.csv'
#
    metric_df = pd.DataFrame.from_dict(history.history)
    metric_df.to_csv(metric_df_path, index=False)
    
    for indx, (i, j, k, g) in enumerate(zip(history.history['dice_coef'], history.history['val_dice_coef'], history.history['loss'], history.history['val_loss'])):
        dice[indx] += i 
        val_dice[indx]  += j
        loss[indx] +=k
        val_loss[indx] +=g
        
dice = [x/iterations for x in dice]
val_dice = [x/iterations for x in val_dice]
loss = [x/iterations for x in loss]
val_loss = [x/iterations for x in val_loss]

lines = plt.plot(epoch_list, dice, epoch_list, val_dice)
l1, l2= lines
plt.setp(lines, linestyle='-')
plt.setp(l1, linewidth=1, color='b')
plt.setp(l2, linewidth=1, color='r')
plt.title('dice-b, val_dice-r' )
plt.grid()
plt.show()
plt.pause(0.05)

lines = plt.plot(epoch_list, loss,epoch_list, val_loss)
l1, l2= lines
plt.setp(lines, linestyle='-')
plt.setp(l1, linewidth=1, color='g')
plt.setp(l2, linewidth=1, color='y')
plt.title('loss-green, val_loss -yellow' )
plt.grid()
plt.show()

# сохраняю усредненные метрики
np.savetxt("C:\\Users\\user_PC\\Desktop\\severstal\\val_dice_binarycross.csv", val_dice, delimiter=",")















# uhfabrb
#plt.figure(figsize=(30, 5))
#plt.subplot(121)
#plt.plot(history.history['dice_coef'])
#plt.plot(history.history['val_dice_coef'])
#plt.title('Model iou_score')
#plt.ylabel('dice_coef')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#
## Plot training & validation loss values
#plt.subplot(122)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('binary_crossentropy')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
## пикчи для человека
#pred = model.predict(val_origin[0].reshape(1, img_sizex, img_sizey, 1))
#plt.imshow(np.squeeze(val_origin[0]))
#plt.pause(0.5)
#
#plt.imshow(np.squeeze(val_masks[0]))
#plt.pause(0.5)

#plt.imshow(np.squeeze(pred))
#plt.pause(0.5)
