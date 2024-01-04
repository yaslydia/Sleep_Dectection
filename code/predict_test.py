import os
import keras

from keras.layers import concatenate

from sklearn.metrics import cohen_kappa_score

import scipy.io 
import math
import random
from keras import optimizers
import numpy as np
import tensorflow as tf
import scipy.io as spio
from sklearn.metrics import f1_score, accuracy_score
np.random.seed(0)  

from keras.preprocessing import sequence
from keras import utils
from keras.models import Sequential
from keras.layers import Layer,Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import  GRU, Bidirectional
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D
from keras.callbacks import History 
from keras.models import Model


from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

from myModel import build_model

from os import listdir
from os.path import isfile, join

import sys
sys.path.append("../..")
from loadData import  *
from utils import *


data_dir = '/home/cxyycl/scratch/Microsleep-code/data/files/'
f_set = '/home/cxyycl/scratch/Microsleep-code/data/file_sets.mat'
out_dir = '/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/pred/'

files_val = []
files_test = []
files_train = []
	
	
mat = spio.loadmat(f_set)
	
	
tmp =  mat['files_val']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_val.extend(file)
tmp =  mat['files_test']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_test.extend(file)

batch_size = 200
n_ep = 4
fs  = 200;
# half_size of the sliding window in samples
w_len = 8*fs;
data_dim = w_len*2
half_prec = 0.5
prec = 1
n_cl = 4

print("=====================")
print("Reading dataset to predict:")
(  data_val, targets_val, N_samples_val) = load_data(data_dir,files_val, w_len)

(  data_test, targets_test, N_samples_test) = load_data(data_dir,files_test, w_len)



ordering = 'channels_last';
keras.backend.set_image_data_format(ordering)

sample_list_val = []
for i in range(len(targets_val)):
	sample_list_val.append([])
	for j in range(len(targets_val[i][0])):
		mid = j*prec
		# we add the padding size
		mid += w_len
		wnd_begin = mid-w_len
		wnd_end = mid+w_len-1
		sample_list_val[i].append([i,j,wnd_begin, wnd_end, 0 ])
		
sample_list_val2 = []
for i in range(len(targets_val)):
	sample_list_val2.append([])
	for j in range(len(targets_val[i][1])):
		mid = j*prec
		# we add the padding size
		mid += w_len
		wnd_begin = mid-w_len
		wnd_end = mid+w_len-1
		sample_list_val2[i].append([i,j,wnd_begin, wnd_end, 1 ])
		

		
sample_list_test = []
for i in range(len(targets_test)):
	sample_list_test.append([])
	for j in range(len(targets_test[i][0])):
		mid = j*prec
		# we add the padding size
		mid += w_len
		wnd_begin = mid-w_len
		wnd_end = mid+w_len-1
		sample_list_test[i].append([i,j,wnd_begin, wnd_end, 0 ])
		
sample_list_test2 = []
for i in range(len(targets_test)):
	sample_list_test2.append([])
	for j in range(len(targets_test[i][1])):
		mid = j*prec
		# we add the padding size
		mid += w_len
		wnd_begin = mid-w_len
		wnd_end = mid+w_len-1
		sample_list_test2[i].append([i,j,wnd_begin, wnd_end, 1 ])


n_channels = 3

def my_generator(data_train, targets_train, sample_list, shuffle = True):
	if shuffle:
		random.shuffle(sample_list)
	while True:
		for batch in batch_generator(sample_list, batch_size):
			batch_data1 = []
			batch_data2 = []
			batch_targets = []
			for sample in batch:
				[f, s, b, e, c] = sample
				sample_label = targets_train[f][c][s]
				sample_x1 = data_train[f][c][b:e+1]
				sample_x2 = data_train[f][2][b:e+1]
				sample_x = np.concatenate( ( sample_x1, sample_x2 ), axis = 2 )
				batch_data1.append(sample_x)
				batch_targets.append(sample_label)
			batch_data1 = np.stack(batch_data1, axis=0)
			batch_targets = np.array(batch_targets)
			batch_targets = to_categorical(batch_targets, n_cl)
			batch_data1 = (batch_data1 )/100 #- np.mean(batch_data, axis=1)
			batch_data1 = np.clip(batch_data1, -1, 1)
			yield [ batch_data1 ], batch_targets

			
def val_data_to_batch(data, targets):
	batch_data = []
	batch_targets = []
	for j in range(len(targets)):
		mid = j*prec
		# we add the padding size
		mid += w_len
		wnd_begin = mid-w_len
		wnd_end = mid+w_len-1
		b = wnd_begin
		e = wnd_end
		sample_label = targets[j]
		sample_x = data[b:e+1]
		batch_data.append(sample_x)
		batch_targets.append(sample_label)
	batch_data = np.stack(batch_data, axis=0)
	batch_targets = to_categorical(batch_targets, n_cl)
	batch_data = (batch_data )/100 #- np.mean(batch_data, axis=1)
	np.clip(batch_data, -1, 1)
	return batch_data, batch_targets		
			


[cnn_eeg, model] = build_model(data_dim, n_channels, n_cl)


Nadam = optimizers.Nadam( )
model.compile(optimizer=Nadam,  loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode=None)






model.load_weights('/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/model.h5')

y_ = []
y = []

O2y_ = []
O2y = []


y_p = []

O2y_p = []

f_list = files_test
for j in range(0,len(f_list)):
	f = f_list[j]
	generator_test = my_generator(data_test, targets_test, sample_list_test[j], shuffle = False)
	scores = model.evaluate_generator( generator_test, int(math.ceil((len(sample_list_test[j],)+0.0)/batch_size)), workers=1)
	generator_test = my_generator(data_test, targets_test, sample_list_test[j], shuffle = False)
	y_pred = model.predict_generator( generator_test, int(math.ceil((len(sample_list_test[j],)+0.0)/batch_size)), workers=1)
	print(y_pred.shape)
	
	y_ = np.argmax(y_pred, axis=1).flatten() 
	y_p = scores
	y = targets_test[j][0]
	
	
	generator_test = my_generator(data_test, targets_test, sample_list_test2[j], shuffle = False)
	scores2 = model.evaluate_generator( generator_test, int(math.ceil((len(sample_list_test2[j],)+0.0)/batch_size)), workers=1)
	generator_test = my_generator(data_test, targets_test, sample_list_test2[j], shuffle = False)
	y_pred2 = model.predict_generator( generator_test, int(math.ceil((len(sample_list_test2[j],)+0.0)/batch_size)), workers=1)
	
	
	O2y_ = np.argmax(y_pred, axis=1).flatten() 
	O2y_p = scores
	O2y =  targets_test[j][0]
	
	scipy.io.savemat( out_dir+'/test/'+f+'.mat', mdict={ 'y_p':y_p,  'y_': y_, 'y':y, 'O2y_p':O2y_p,  'O2y_': O2y_, 'O2y':O2y })
