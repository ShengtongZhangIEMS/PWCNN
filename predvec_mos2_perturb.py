# Iteratively predict the stress and strain on the New MS;
# The model has 6 outputs (strain, stress)
#%%
import argparse
import math
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio
# import seaborn as sns
from sklearn.linear_model import LinearRegression
from tensorflow.keras import regularizers

from tensorflow.keras.models import load_model

# In[4]:

def load_image(path):
    im = sio.loadmat(path)
    # plt.imshow(im)
    # plt.axis('off')
    return im




# In[5]:


def load_mat(path):
    mat_contents = sio.loadmat(path)
    return mat_contents


# In[6]:


import os
import time
import tensorflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (AveragePooling2D, Concatenate, Conv1D, Conv2D, Dense, Activation,
                          Flatten, Input, Lambda, MaxPooling2D, GlobalAveragePooling2D, Permute,
                          RepeatVector, Reshape, ZeroPadding2D, BatchNormalization)
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import linalg as LA
from tensorflow.keras.utils import Sequence

print(tf.test.gpu_device_name())

#%%
im = load_image('./simulation/mos2_perturb.mat')
im = im['mesh3D']
f = h5py.File('./simulation/mos2_Out.mat', 'r')
arrays = {}
# plt.imshow(im_perturb[1097-100:1097+100,1098-100:1098+100])
for k, v in f.items():
    arrays[k] = np.array(v)
Out = arrays['Out']
ee = arrays['ee']
ss = arrays['ss']

#%%
def Elas3D(E,v):
    # Return the 3D elastic tensor in Voigt notation for given young's modulus and poisson ratio
    C = np.array([[1-v, v, v, 0, 0, 0], [v, 1-v, v, 0, 0, 0], [v, v, 1-v, 0, 0, 0], [0, 0, 0, 1/2-v, 0, 0], [0, 0, 0, 0, 1/2-v, 0], [0, 0, 0, 0, 0, 1/2-v]])
    C = E/(1+v)/(1-2*v)*C
    return C

E = 38000; v = 0.2
C = Elas3D(E, v)

# In[8]: generate 10 moving window to perturb
win = 100
num_load = 10
ew_length = 9
stride = 1
BS = 200
num_patch_1d = 1

stress_max, strain_max = 253.31279436907013, 0.10431032663597739

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true, axis = 0) ) )
    return 1 - SS_res/(SS_tot + K.epsilon())
  
dependencies = {
    'coeff_determination': coeff_determination
}
model_mat = load_model('./save_data/model_mat_constitutive_irregular.h5', custom_objects=dependencies)  
for layer in model_mat.layers:
  layer.trainable = False
  
class ComputeStress(tensorflow.keras.layers.Layer):
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

def ExpandConv(num_patch_1d, im_input, win = 100, ew_length=9, stride=1,BATCHSIZE=1, num_load=10,weights_path=None):
  # ew_length: expanding window length
  inputs = {}  
  # preprocess
  row, col = im_input.shape
  inputs['input1'] = tf.cast(im_input.reshape(1,row,col,1),dtype='float')
  # inputs['input1'] = Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu", name = 'FEATURE_ST')(inputs['input1'])
  # inputs['input1'] = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", name = 'FEATURE_ST')(inputs['input1'])
  for i in range(1,6):
    inputs['input'+str(i+1)] = AveragePooling2D(pool_size=(2,2),strides=(2,2), padding='same')(inputs['input'+str(i)])
  padding = ew_length//2+1 - (2000//(2**5) - (2000-win)//(2**5))
  st_padding = ew_length//2 - win//(2**5)
  inputs['input6'] = ZeroPadding2D(padding=((st_padding,padding),(st_padding,padding)))(inputs['input6'])


  n_input = 12 # The number of numerical features.
  num_patch_2d = num_patch_1d**2
  input1 = tf.keras.Input(shape = (2,),batch_size = BATCHSIZE, dtype = 'int32') # (begin position at the first levels)
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
    x = BatchNormalization(trainable=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters,kernel_size=(3,3),padding="same", name = 'Conv_2_Scale'+str(i))(x)
    x = BatchNormalization(trainable=False)(x)
    feature['feature_'+str(i+1)] = Activation('relu')(x)

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

  # r1 = tf.squeeze(im_feature, axis = 1)
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

model = ExpandConv(num_patch_1d=1, im_input = im, ew_length = ew_length, stride=stride, BATCHSIZE = BS,\
  weights_path = "./checkpoints/irregular_constitutive_BS256.h5")
print('Finish Importing the Model \n')


#%%
# num_load = 10

def WindowBoundaryCondition(s_field, win = win):
  average_strain = {}
  average_xx = np.empty((2000-2*win, 2000-2*win, 4))    
  average_yy = np.empty((2000-2*win, 2000-2*win, 4))
  average_xy = np.empty((2000-2*win, 2000-2*win, 4))
  Out_img_xx = s_field[:,0].reshape((2000,2000)).transpose()
  Out_img_yy = s_field[:,1].reshape((2000,2000)).transpose()
  Out_img_xy = s_field[:,2].reshape((2000,2000)).transpose()
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
            
  average_strain = np.concatenate((average_xx, average_yy, average_xy), axis = -1)
  return average_strain

average_strain = {}
for load in range(num_load):
    average_strain[str(load)] = WindowBoundaryCondition(Out[load,[1,2,6],:].T)

#%%
matrix = np.empty(im.shape)
mask = np.empty(im.shape)
for i in range(im.shape[0]):
  for j in range(im.shape[1]):
    if im[i,j] == 1:
      matrix[i,j] = np.nan
      mask[i,j] = 0
    else:
      matrix[i,j] = 1
      mask[i,j] = 1

vm_stress = {}
vm_stress_mat = {}
strain_stress_vec = {}
# vm_max = 200.16943120127345
for load in range(num_load):
  # stress = Out[:, [1,2,6], load] # Strain
  stress = Out[load, 7:13,:]/stress_max # Strain
  strain = Out[load, 1:7,:]/strain_max # Strain
  strain_stress_vec[str(load)] = np.concatenate((strain,stress),axis=0)
  vm_stress[str(load)] = np.sqrt(stress[0,:]**2 - stress[0,:]*stress[1,:] +\
    stress[1,:]**2 + 3*stress[2,:]**2)
  vm_stress_mat[str(load)] = (vm_stress[str(load)].reshape(im.shape).T)*mask

#%%

class PredGenerator(Sequence):
  """Generates data for Keras
  Sequence based data generator. Suitable for building data generator for training and prediction.
  """
  def __init__(self, im, average_num_feature, s_vec, window_size, stride, load, num_patch_1d = 1,\
    batch_size=32, ew_length = 9, shuffle=True):
    self.batch_size = batch_size
    self.im = im.astype(np.float32)
    self.s_vec = s_vec
    self.stride = stride
    self.load = load
    self.num_patch_1d = num_patch_1d
    self.ew_length = ew_length
    self.average_num_feature = average_num_feature
    
    self.width = im.shape[0]
    self.height = im.shape[1]
    self.win = window_size
    self.num_width = 1800 // (self.num_patch_1d*self.stride)
    self.num_height = 1800 // (self.num_patch_1d*self.stride)
    # self.indices = np.arange(self.num_width*self.num_height).tolist()

    self.indices = [[i*self.num_patch_1d*self.stride + k,j*self.num_patch_1d*self.stride+l] for k in range(self.stride) for l in range(self.stride) \
      for i in range(self.num_width) for j in range(self.num_height)] 
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    return len(self.indices) // self.batch_size

  def __getitem__(self, index):
    index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
    batch = [self.indices[k] for k in index]
    X, y = self.__get_data__(batch)
    return X, y

  def on_epoch_end(self):
    self.index = np.arange(len(self.indices))

  def __get_data__(self, batch):

    # Initialization
    n_input = 12
    
    X1 = np.zeros((self.batch_size, 2))
    X2 = np.zeros((self.batch_size, 1, n_input))
    X3 = np.zeros((self.batch_size, 1, 1))
    y = np.zeros((self.batch_size, 12), dtype=float)
    
    for i, id in enumerate(batch):
      row = id[0]
      col = id[1]
      X1[i,] = np.array([row+self.win, col+self.win])
      X2[i,0,] = self.average_num_feature[row, col,:]
      X3[i,0,] = self.im[row+self.win, col+self.win]
      y[i,] = self.s_vec[:,(row+self.win)+2000*(col+self.win),]
      # for j in range(3):
      #   X2[i,j] = np.mean(stress[(center[0]-self.win):(center[0]+self.win+1), center[1]-self.win, j])
      #   X2[i,j+3] = np.mean(stress[(center[0]-self.win):(center[0]+self.win+1), center[1]+self.win+1, j])
      #   X2[i,j+6] = np.mean(stress[(center[0]-self.win), (center[1]-self.win):(center[1]+self.win+1), j])
      #   X2[i,j+6] = np.mean(stress[(center[0]+self.win), (center[1]-self.win):(center[1]+self.win+1), j])
      # y[i] = np.mean(self.vm_stress_2d[(center[0]-self.win):(center[0]+self.win+1),(center[1]-self.win):(center[1]+self.win+1)])
    
    return [X1,X2,X3], y

#%%

def VMStress(stress):
    vm_stress = np.sqrt(stress[:,0]**2 - stress[:,0]*stress[:,1] +\
        stress[:,1]**2 + 3*stress[:,2]**2)
    return vm_stress
#%%
# Predict under true average strain in the New MS
matrix_idx = np.nonzero(mask[100:1900,100:1900])
matrix_num = np.count_nonzero(mask[100:1900,100:1900])
particle_idx = np.nonzero(1-mask[100:1900,100:1900])
particle_num = 1800**2 - matrix_num


SSE_mat = np.zeros((num_load,6))
SST_mat = np.zeros((num_load,6))
SSE_par = np.zeros((num_load,6))
SST_par = np.zeros((num_load,6))

y_pred_2d = np.zeros((num_load, 1800,1800, 6))
y_pred_vec_mat = np.zeros((num_load, matrix_num, 6))
y_pred_vec_par = np.zeros((num_load, particle_num, 6))
y_true_vec_mat = np.zeros((num_load, matrix_num, 6))
y_true_vec_par = np.zeros((num_load, particle_num, 6))


output_file = './outputs/outputs_strain_stress_mos2_perturb.txt'
# output_file = 'tmp.txt'
with open(output_file, 'w') as f:
  f.write('Start to predict using true average strain \n')
       
#%%

def PredWindowBoundaryCondition(s_field, win = win):
  average_strain = {}
  average_xx = np.empty((2000-2*win, 2000-2*win, 4))    
  average_yy = np.empty((2000-2*win, 2000-2*win, 4))
  average_xy = np.empty((2000-2*win, 2000-2*win, 4))
  Out_img_xx = s_field[:,0].reshape((2000,2000))
  Out_img_yy = s_field[:,1].reshape((2000,2000))
  Out_img_xy = s_field[:,2].reshape((2000,2000))
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
            
  average_strain = np.concatenate((average_xx, average_yy, average_xy), axis = -1)
  return average_strain

def stress_gradient(stress):  
  stress_xx = stress[..., 0]
  stress_yy = stress[..., 1]
  stress_xy = stress[..., 2]
  
  eqx = - np.mean(stress_xx[0,]) + np.mean(stress_xx[-1,]) + np.mean(stress_xy[:,-1])-np.mean(stress_xy[:,0])
  eqy = np.mean(stress_yy[:,-1]) - np.mean(stress_yy[:,0]) + np.mean(stress_xy[-1,])-np.mean(stress_xy[0,])
  
  # eqx = - np.mean(stress_xx[:,0]) + np.mean(stress_xx[:,-1]) + np.mean(stress_xy[-1,])-np.mean(stress_xy[0,])
  # eqy = np.mean(stress_yy[-1,]) - np.mean(stress_yy[0,]) + np.mean(stress_xy[:,-1])-np.mean(stress_xy[:,0])
  return np.abs(eqx + eqy)

def equilibrium(stress, window = 200):  
  n = stress.shape[0] // window
  window_err = []
  for i in range(n):
    for j in range(n):
      stress_win = stress[i*window:(i+1)*window, j*window:(j+1)*window,]
      window_err.append(stress_gradient(stress_win))
      
  return np.mean(window_err)

#%%

num_iter = 5

# Check the local convergenec curve
pos = [] # position for the pixel to monitor
ref_line = im[1000:1200,1150]


err = np.zeros((num_iter, 6))

y_true_2d_iter = np.zeros((num_load, 1800, 1800, 12))
y_pred_2d_iter = np.zeros((num_load, 1800, 1800, 12))
rel_err = np.zeros((num_load, num_iter, 2)) ### SAE/SAT for strain and stress
equ_err = np.zeros((num_load, num_iter)) ### equivalent equation error
equilibrium_err = np.zeros((num_load, num_iter))
fft_err = np.zeros((num_load, num_iter)) ### SAE/SAT for strain and stress
no_adjustment_rel_err = np.zeros((num_load, num_iter,)) ### difference between predicted strain (before physical adjustment) and loading condition
# y_true_vec_mat = np.zeros((num_load, matrix_num,3))
# y_true_vec_par = np.zeros((num_load, particle_num,3))


pred_mean = np.zeros((num_iter, 12))
with open(output_file, 'a') as f:
  f.write('\n\nStart the iteratively prediction on the new MS \n\n')
# for load in [8]:
for load in range(num_load):
  y_true = np.empty((1800**2,12))
  for row in range(1800):
    for col in range(1800):
      i = row*1800+col
      y_true[i,] = strain_stress_vec[str(load)][:,(row+win)+2000*(col+win),]

  # y_true_vm_stress_mat = VMStress(y_true).reshape((2000,2000))[matrix_idx]
  # y_true_vm_stress_par = VMStress(y_true).reshape((2000,2000))[particle_idx]
  y_true[:,:6] *= strain_max
  y_true[:,6:] *= stress_max
  y_true_vec = y_true.reshape((1800,1800,12))
  y_true_2d_iter[load,] = y_true_vec
  ########################################
  ### Compute the initial average strain using particle length ratio for [0,1] only.
  avg_strain= np.empty((1800,1800,12))

  # for i in range(2):
  #   reg = LinearRegression().fit(ratio[...,i].flatten().reshape(-1,1),average_s train[str(load)][...,i].flatten())
  #   avg_strain[...,i] = reg.predict(ratio_perturb[...,i].reshape(-1,1)).reshape(2000,2000)

  bd_strain = np.array([ee[0,load],0, 0,]) # Use loading strain to initial the average strain

  # avg_strain[...,2:4] = bd_strain[0]
  avg_strain[...,:4] = bd_strain[0]
  avg_strain[...,4:8] = bd_strain[1]
  avg_strain[...,8:12] = bd_strain[2]

  y_pred_vec = np.zeros((1800,1800,12))

  # s_field_new = np.repeat(np.array(bd_strain).reshape((1,-1)), 2000**2, axis = 0)

  for itr in range(num_iter):
    with open(output_file, 'a') as f:
      f.write('\n load %d iteration number %d \n' % (load,itr))

    pred_generator = PredGenerator(im, avg_strain, strain_stress_vec[str(load)],\
      win, load = load, num_patch_1d=1, stride=1, batch_size = BS,  shuffle = False)

    begin = time.time()
    # y_pred = model.predict_on_batch(X)
    print('starts to predict on iteration %d' % itr)

    y_pred = model.predict(pred_generator, verbose = 0, workers = 5, use_multiprocessing = True)
    y_pred = y_pred[:,0,:]
    end = time.time()
    t = end - begin
    # print('the prediction time is %.3E' % t)

    with open(output_file, 'a') as f:
      f.write('\n the prediction time is %f for load %d iteration %d \n' % (t, load, itr))
    print(('\n the prediction time is %f for load %d iteration %d' % (t, load, itr)))

    y_pred[:,:6] *= strain_max
    y_pred[:,6:] *= stress_max
    
    # model_mat = load_model('./save_data/model_mat_constitutive.h5')    
    
    y_pred_vec = y_pred.reshape((1800,1800,12))
    
    
    no_adjustment_rel_err[load, itr] = np.mean(np.abs(y_pred_vec[...,0] - bd_strain[0]))

    # update the whole stress and strain fields
    if itr == 0:
      mean_strain = np.mean(np.mean(y_pred_vec[..., [0,1,5]], axis = 0), axis = 0)
      s_field_2d = np.zeros((2000,2000,3))
      s_field_2d[...,0] = bd_strain[0]
      for i in range(1,3):
        s_field_2d[...,i] = mean_strain[i]

    # update the strain field
    # for col in range(1800):
    #   for row in range(1800):
    #     s_field_2d[row+win, col+win, :] =  y_pred[row*1800+col,:3]
    #   # Multiplication adjustment for predicted strain. np.mean(strain_xx) = boundary_strain
    #   coef = ee[0,load] / np.mean(s_field_2d[:,col+win,0])
    #   s_field_2d[:, col+win, :] *= coef
    # s_field_new = s_field_2d.reshape((-1,3))
    

    s_field_2d[win:2000-win, win:2000-win, :] =  y_pred_vec[...,[0,1,5]]
    # Multiplication adjustment for predicted strain. np.mean(strain_xx) = boundary_strain
    coef = ee[0,load] / np.mean(y_pred_vec[...,0])
    s_field_2d[win:2000-win, win:2000-win, :] *= coef
    s_field_new = s_field_2d.reshape((-1,3))
    avg_strain = PredWindowBoundaryCondition(s_field = s_field_new)

    # y_pred_vec[..., 0] *= coef
    y_pred_vec[..., :6] *= coef
    if itr == 2:
      y_pred_2d_iter[load,] = y_pred_vec # Saved for plot
        
    m = y_pred_vec.mean(axis = 0).mean(axis = 0)
    pred_mean[itr,] = m
    
    ### Compute the relative absolute error
    rel_err[load, itr,0] = np.sum(np.abs(y_true_vec[...,:6] - y_pred_vec[...,:6]))/np.sum(np.abs(y_true_vec[...,:6]))
    rel_err[load, itr,1] = np.sum(np.abs(y_true_vec[...,6:12] - y_pred_vec[...,6:12]))/np.sum(np.abs(y_true_vec[...,6:12]))
    
    # compute the equilibrium equation error
    equ_err[load, itr] = stress_gradient(y_pred_vec[...,[6,7,11]])
    equilibrium_err[load, itr] = equilibrium(y_pred_vec[...,[6,7,11]], window = 200)

    
    with open(output_file, 'a') as f:
      
      f.write('The y_pred_vec mean is \n ')
      for element in m:
        f.write('%.3E' % element)
        f.write(', ')
      f.write('\n\n')

      f.write('The relative absolute error for strain is %.3E \n' % rel_err[load, itr,0])
      f.write('The relative absolute error for stress is %.3E \n' % rel_err[load, itr,1])

# Save prediction data
np.savez('./save_data/pred_mos2_constitutive_strain_xx.npz', true = y_true_2d_iter, pred = y_pred_2d_iter,\
  rel_err = rel_err, equ_err = equ_err, equilibrium_err = equilibrium_err)
    
#%%
data = np.load('./save_data/pred_mos2_constitutive_strain_xx.npz')
equilibrium_err = data['equilibrium_err']
for i in range(0, 10, 2):
    plt.plot(range(1,6), equilibrium_err[i, :], '-o', label = 'load = %.3f' % ee[0, i])
    # plt.plot(range(1,6), equ_err[i, :], '-o', label = 'load = %.3f' % ee[0, i])
    plt.title('equilibrium equation error', fontsize = 20)
    plt.xticks(range(1,6), fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('# iteration', fontsize = 20)
    plt.ylabel('error', fontsize = 20)
    plt.legend(fontsize = 15)
    
# %%
true = data['true']
pred = data['pred']
rel_strain = np.sum(np.abs(true[...,:6] - pred[...,:6]))/np.sum(np.abs(true[...,:6]))
rel_stress = np.sum(np.abs(true[...,6:] - pred[...,6:]))/np.sum(np.abs(true[...,6:]))
print(rel_strain, rel_stress)
# %%
