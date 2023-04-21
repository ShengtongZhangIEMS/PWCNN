# 3/28/2021 Shengtong Zhang

import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt



# In[4]:


def load_image(path):
    im = sio.loadmat(path)
    return im


# In[5]:


def load_mat(path):
    mat_contents = sio.loadmat(path)
    return mat_contents


# In[6]:

import os
import time

# import cv2
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (AveragePooling2D, Concatenate, Conv1D, Conv2D, Dense, Activation, BatchNormalization,
                          Flatten, Input, Lambda, MaxPooling2D, Permute, GlobalAveragePooling2D,ZeroPadding2D)
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
# from keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import linalg as LA
from tensorflow.keras.utils import Sequence
import h5py
from helper.DataGenerator6 import DataGeneratorPhase6

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


im = load_image('./round_inclusion_data/im.mat')
im = im['newMS']
f = h5py.File('./round_inclusion_data/2001x2001_result.mat', 'r')
arrays = {}
plt.imshow(im[1097-100:1097+100,1098-100:1098+100])
for k, v in f.items():
    arrays[k] = np.array(v)
Out = arrays['Out']
ee = arrays['ee']

#%%
def Elas3D(E,v):
    # Return the 3D elastic tensor in Voigt notation for given young's modulus and poisson ratio
    C = np.array([[1-v, v, v, 0, 0, 0], [v, 1-v, v, 0, 0, 0], [v, v, 1-v, 0, 0, 0], [0, 0, 0, 1/2-v, 0, 0], [0, 0, 0, 0, 1/2-v, 0], [0, 0, 0, 0, 0, 1/2-v]])
    C = E/(1+v)/(1-2*v)*C
    return C

E = 38000; v = 0.2
C = Elas3D(E, v)


#%%
num_load = 10
ew_length = 9 
win = 100

# In[8]:


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true, axis = 0) ) )
    return 1 - SS_res/(SS_tot + K.epsilon())
  
dependencies = {
    'coeff_determination': coeff_determination
}
model_mat = load_model('./save_data/model_mat_constitutive.h5', custom_objects=dependencies)  
for layer in model_mat.layers:
  print(layer)
  layer.trainable = False
  
class ComputeStress(keras.layers.Layer):
    def __init__(self,):
        super(ComputeStress, self).__init__()
        self.C = tf.convert_to_tensor(C, dtype=tf.float32)

    def call(self, inputs):
        phase = tf.repeat(inputs[0][:,0,], 6, axis = -1)
        stress_par = K.dot(inputs[1], self.C)
        stress_mat = model_mat(inputs[1])
        stress_pred = phase * stress_par  + (1-phase)*stress_mat
        # print(stress_pred.shape)
        
        return stress_pred
      

def ExpandConv(num_patch_1d, im_input = im, ew_length=9, stride=2,BATCHSIZE=1,num_load = num_load,weights_path=None):
  # ew_length: expanding window length

  inputs = {}  
  # preprocess
  inputs['input1'] = tf.cast(im_input.reshape(1,2000,2000,1),dtype='float')
  # inputs['input1'] = Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu", name = 'FEATURE_ST')(inputs['input1'])
  # inputs['input1'] = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", name = 'FEATURE_ST')(inputs['input1'])
  for i in range(1,6):
    inputs['input'+str(i+1)] = AveragePooling2D(pool_size=(2,2),strides=(2,2), padding='same')(inputs['input'+str(i)])
  padding = ew_length//2+1 - (2000//(2**5) - (2000-win)//(2**5))
  st_padding = ew_length//2 - win//(2**5)
  inputs['input6'] = ZeroPadding2D(padding=((st_padding,padding),(st_padding,padding)))(inputs['input6'])

  n_input = 12 # The number of numerical features.
  num_patch_2d = num_patch_1d**2
  input1 = tf.keras.Input(shape = (2,),batch_size = BATCHSIZE, dtype = 'int32') # (begin position at th first levels)
  input2 = tf.keras.Input(shape = (num_patch_2d, n_input),batch_size = BATCHSIZE) # number of loads = 5
  input3 = tf.keras.Input(shape = (num_patch_2d,1),batch_size = BATCHSIZE) # (phase of the pixel, matrix or particle)

  # Calculate the starting point and size of the feature map
  ew_begin = []; ew_size = []
  for i in range(6):
    feature_beg = input1 // (2**i)-ew_length//2
    if i == 5:
      feature_beg += st_padding
    length = ((num_patch_1d-1)*stride) // (2**i) + ew_length
    ew_begin += [tf.concat([feature_beg, tf.zeros([BATCHSIZE,1],dtype = 'int32')],1)]
    ew_size += [[length,length,1]]
          

  ### Extract features from different scale
  feature = {}
  patch_feature = {}
  filters = 64


  for i in range(6):
    expanding_window = tf.stack([tf.slice(inputs['input'+str(i+1)][0,],begin=ew_begin[i][batch,], size=ew_size[i], name = 'SLICE'+str(i)) for batch in range(BATCHSIZE)],name = 'STACK'+str(i))
    x = Conv2D(filters=filters,kernel_size=(3,3),padding="valid", name = 'Conv_1_Scale'+str(i))(expanding_window)
    x = BatchNormalization(trainable=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters,kernel_size=(3,3),padding="same", name = 'Conv_2_Scale'+str(i))(x)
    x = BatchNormalization(trainable=True)(x)
    feature['feature_'+str(i+1)] = Activation('relu')(x)

    # cut the feature domain for each patch
    size = [7,7,filters]
    patch_feature_i = {}
    for j in range(num_patch_1d):
      for k in range(num_patch_1d):
        center = input1 + tf.concat([tf.ones([BATCHSIZE,1],dtype = 'int32')*j*stride,tf.ones([BATCHSIZE,1],dtype = 'int32')*k*stride],1)
        beg = center//(2**i) - input1//(2**i)
        beg = tf.concat([beg, tf.zeros([BATCHSIZE,1],dtype = 'int32')],1)
        patch_feature_i[str(j)+'_'+str(k)] = tf.stack([tf.slice(feature['feature_'+str(i+1)][batch,],begin=beg[batch,], size=size, name = 'SLICE'+str(i)) for batch in range(BATCHSIZE)],name = 'STACK'+str(i))
        patch_feature_i[str(j)+'_'+str(k)] = GlobalAveragePooling2D(data_format='channels_last')(patch_feature_i[str(j)+'_'+str(k)])
        
    patch_feature[str(i+1)] = tf.stack([patch_feature_i[str(j)+'_'+str(k)] for j in range(num_patch_1d) for k in range(num_patch_1d)], axis=1)


  im_feature = tf.concat([patch_feature[str(i+1)] for i in range(6)], axis = -1)

  r1 = im_feature
  r2 = input2
  r2 = Dense(64, activation = 'relu', name = 'input_dense_2')(r2)
  r3 = input3
  r3 = Dense(64, activation = 'relu', name = 'input_dense_3')(r3)
  merge = Concatenate(axis = -1)([r1, r2, r3])
  fc = Dense(128, activation='relu', name = 'dense_1_load_'+str(i))(merge)
  strain_output = Dense(6, activation='linear', name = 'dense_2_load_'+str(i))(fc)[:,0,]
  
  stress_output = ComputeStress()([input3, strain_output*strain_max])/stress_max
  output = tf.concat([strain_output, stress_output], axis = -1)
  # output = tf.concat([strain_output, strain_output], axis = -1) 
  print(output.shape)
  output = tf.expand_dims(output, axis = 1)
  
  model = Model(inputs = [input1, input2, input3], outputs = output)
  if weights_path:
    model.load_weights(weights_path)
  return model


# In[10]: Define vm_stress for 5 different loads and normalize it by the maximum value.
print('compute average strain')
average_strain = {}
if 0:
  average_strain['0'] = np.zeros((2000,2000,12))
else:
  for load in range(num_load):
    average_xx = np.empty((2000-2*win, 2000-2*win, 4))    
    average_yy = np.empty((2000-2*win, 2000-2*win, 4))
    average_xy = np.empty((2000-2*win, 2000-2*win, 4))
    Out_img_xx = Out[load,1,:].reshape((2000,2000)).transpose()
    Out_img_yy = Out[load,2,:].reshape((2000,2000)).transpose()
    Out_img_xy = Out[load,6,:].reshape((2000,2000)).transpose()
    for i in range(2000-2*win):
        for j in range(2000-2*win):
            center = [i+win, j+win]
            if j == 0:
              average_xx[i,j,0] = np.mean(Out_img_xx[(center[0]-win), center[1]-win : center[1]+win+1])
              average_xx[i,j,1] = np.mean(Out_img_xx[(center[0]+win), center[1]-win :  center[1]+win+1])
              
              average_yy[i,j,0] = np.mean(Out_img_yy[(center[0]-win), center[1]-win : center[1]+win+1])
              average_yy[i,j,1] = np.mean(Out_img_yy[(center[0]+win), center[1]-win :  center[1]+win+1])
              
              average_xy[i,j,0] = np.mean(Out_img_xy[(center[0]-win), center[1]-win : center[1]+win+1])
              average_xy[i,j,1] = np.mean(Out_img_xy[(center[0]+win), center[1]-win :  center[1]+win+1])      
            else:
              average_xx[i,j,0] = average_xx[i,j-1,0] + (Out_img_xx[(center[0]-win), center[1]+win] - Out_img_xx[(center[0]-win), center[1]-win-1])/(2*win+1)
              average_xx[i,j,1] = average_xx[i,j-1,1] + (Out_img_xx[(center[0]+win), center[1]+win] - Out_img_xx[(center[0]+win), center[1]-win-1])/(2*win+1)
              
              average_yy[i,j,0] = average_yy[i,j-1,0] + (Out_img_yy[(center[0]-win), center[1]+win] - Out_img_yy[(center[0]-win), center[1]-win-1])/(2*win+1)
              average_yy[i,j,1] = average_yy[i,j-1,1] + (Out_img_yy[(center[0]+win), center[1]+win] - Out_img_yy[(center[0]+win), center[1]-win-1])/(2*win+1)
              average_xy[i,j,0] = average_xy[i,j-1,0] + (Out_img_xy[(center[0]-win), center[1]+win] - Out_img_xy[(center[0]-win), center[1]-win-1])/(2*win+1)
              average_xy[i,j,1] = average_xy[i,j-1,1] + (Out_img_xy[(center[0]+win), center[1]+win] - Out_img_xy[(center[0]+win), center[1]-win-1])/(2*win+1)
    
            if i == 0:
              average_xx[i,j,2] = np.mean(Out_img_xx[(center[0]-win) : (center[0]+win+1),center[1]-win])
              average_xx[i,j,3] = np.mean(Out_img_xx[(center[0]-win) : (center[0]+win+1),center[1]+win])
              average_yy[i,j,2] = np.mean(Out_img_yy[(center[0]-win) : (center[0]+win+1),center[1]-win])
              average_yy[i,j,3] = np.mean(Out_img_yy[(center[0]-win) : (center[0]+win+1),center[1]+win])
              average_xy[i,j,2] = np.mean(Out_img_xy[(center[0]-win) : (center[0]+win+1),center[1]-win])
              average_xy[i,j,3] = np.mean(Out_img_xy[(center[0]-win) : (center[0]+win+1),center[1]+win])      
            else:
              average_xx[i,j,2] = average_xx[i-1,j,2] + (Out_img_xx[center[0]+win,center[1]-win] - Out_img_xx[center[0]-win-1,center[1]-win])/(2*win+1)
              average_xx[i,j,3] = average_xx[i-1,j,3] + (Out_img_xx[center[0]+win,center[1]+win] - Out_img_xx[center[0]-win-1,center[1]+win])/(2*win+1)
              average_yy[i,j,2] = average_yy[i-1,j,2] + (Out_img_yy[center[0]+win,center[1]-win] - Out_img_yy[center[0]-win-1,center[1]-win])/(2*win+1)
              average_yy[i,j,3] = average_yy[i-1,j,3] + (Out_img_yy[center[0]+win,center[1]+win] - Out_img_yy[center[0]-win-1,center[1]+win])/(2*win+1)
              average_xy[i,j,2] = average_xy[i-1,j,2] + (Out_img_xy[center[0]+win,center[1]-win] - Out_img_xy[center[0]-win-1,center[1]-win])/(2*win+1)
              average_xy[i,j,3] = average_xy[i-1,j,3] + (Out_img_xy[center[0]+win,center[1]+win] - Out_img_xy[center[0]-win-1,center[1]+win])/(2*win+1)
              
    
    average_strain[str(load)] = np.concatenate((average_xx, average_yy, average_xy), axis = -1)

# In[10]: Define vm_stress for 5 different loads and normalize it by the maximum value.


stress_max = np.max(Out[num_load-1, 7:13,:])
strain_max = np.max(Out[num_load-1, 1:7,:])

vm_stress = {}
vm_stress_mat = {}
strain_stress_vec = {}
stress_vec = {}
stress_mean = {}
# Nomalize by the loading condition
load_cond = ee[0,:]
for load in range(num_load):
  stress = Out[load, 7:13,:]/stress_max # Stress
  strain = Out[load, 1:7,:]/strain_max # Strain
  # stress = Out[load, 7:13,:] 
  # strain = Out[load, 1:7,:]
  strain_stress_vec[str(load)] = np.concatenate((strain,stress),axis=0)
  stress_vec[str(load)] = stress
  stress_mean[str(load)] = np.mean(stress, axis = 1)
  vm_stress[str(load)] = np.sqrt(stress[0,:]**2 - stress[0,:]*stress[1,:] +\
     stress[1,:]**2 + 3*stress[2,:]**2)
  # vm_stress_mat[str(load)] = (vm_stress[str(load)].reshape(im.shape).T)*mask
  # vm_stress_2d = vm_stress[str(load)].reshape((2000,2000)).transpose()

vm_max = np.max(vm_stress[str(num_load - 1)])
for load in range(num_load):
  vm_stress[str(load)] /= vm_max

#%%
del Out
print('Start DataGenerator')

BS = 256
EPOCHS = 20
stride = 1
num_patch_1d=1
training_generator = DataGeneratorPhase6(im, average_strain, strain_stress_vec, win, data_type = 'training', num_patch_1d=num_patch_1d, stride=stride, batch_size = BS, num_load = num_load)
valid_generator = DataGeneratorPhase6(im, average_strain, strain_stress_vec, win, data_type = 'valid', num_patch_1d=num_patch_1d, stride=stride, batch_size = BS, num_load = num_load, shuffle=False)

#%%

model = ExpandConv(num_patch_1d=num_patch_1d, im_input = im, ew_length = ew_length, stride=stride, BATCHSIZE = BS, weights_path = "./checkpoints/silica_constitutive_BS256.h5")
adam = Adam(learning_rate=1e-3, decay= 1e-3 / 200)
model.compile(optimizer=adam, loss=tf.keras.losses.Huber(), metrics=[coeff_determination],)

# In[ ]:

from tensorflow.keras.callbacks import ModelCheckpoint
# checkpoint
filepath="./checkpoints/silica_constitutive_BS256.h5"
# 有一次提升, 则覆盖一次.
checkpoint = ModelCheckpoint(filepath,monitor='loss', verbose=1, save_best_only=True,save_weights_only = True, mode='auto')
callbacks_list = [checkpoint]
history = model.fit(training_generator,
    validation_data = valid_generator,
    steps_per_epoch=training_generator.__len__(),
    validation_steps=valid_generator.__len__(),
    verbose = 1,
    callbacks = callbacks_list,
    epochs=EPOCHS,)

# %%
