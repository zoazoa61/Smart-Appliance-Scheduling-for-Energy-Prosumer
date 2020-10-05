# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:51:59 2020

@author: user
"""
#CS
import matplotlib as mpl
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import sympy
from sympy import Symbol, solve

import pmdarima as pm
import sklearn
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# from keras import batch_normalization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from IPython.display import Image
# visualize in 3D plot
from pylab import rcParams
from bitstring import BitArray, ConstBitStream
import pandas as pd
import tensorflow as tf1
import tensorflow.compat.v1 as tf
from IPython.display import SVG
import pandas as pd
from sklearn import model_selection
# import hyparams as hp
# from tensorflow.keras.utils.vis_utils import model_to_dot
import time
import sys
import random
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.constraints import max_norm
import os
from keras import backend as K

from keras.losses import mean_absolute_error
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_percentage_error
#ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import inspect
print(inspect.getsource(sklearn.metrics.mean_squared_error))
# define loss function myself

def RAE(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.abs( (y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None) )    
    return K.mean(diff, axis=-1)

def average(x):
    mean_x = sum(x)/len(x)    
    return(mean_x)

def step_function(x_data): 
    return [0 if x<=0 else 1 for x in x_data ]

def Loss_Plotting(x, x_hat):
    #CS loss
    loss_mse = sklearn.metrics.mean_squared_error(x, x_hat)
    loss_mae = sklearn.metrics.mean_absolute_error(x, x_hat)

    """ Plot """
    plt.plot(x, '--')
    plt.plot(x_hat, '--')

    plt.show()
    return(loss_mse, loss_mae)


"""ARIMA"""
def Opt_ARIMA(np_data,p,d,q):
# def Opt_ARIMA(np_data):
    
    series = pd.DataFrame(np_data)

    # series.plot()
    # plot_acf(series)
    # plot_pacf(series)
    # plt.show()

    # diff_1=series.diff(periods=1).iloc[1:] #1st differation
    # diff_1.plot()
    # plot_acf(diff_1)
    # plot_pacf(diff_1)
    # plt.show()
    # fit stepwise auto-ARIMA
    "optimazation"
    # arima_opt = pm.auto_arima(series, start_p=1, start_q=1,
    #                               max_p=3, max_q=3, m=12,
    #                               start_P=0, seasonal=True,
    #                               d=1, D=1, trace=True,
    #                               error_action='ignore',  # don't want to know if an order does not work
    #                               suppress_warnings=True,  # don't want convergence warnings
    #                               stepwise=True)  # set to stepwise
    # opt_params = arima_opt.to_dict()
    # opt_params1 = opt_params['order']
    
    # model = ARIMA(series.values, order=opt_params1) #(opt_params[0],opt_params[1],opt_params[2])
    model = ARIMA(series.values, order=(p,d,q)) #(opt_params[0],opt_params[1],opt_params[2])

    model_fit = model.fit(trend='nc',full_output=True, disp=1)
    print(model_fit.summary())

    #prediction
    series_prediction = model_fit.predict(typ ='levels')
    # series_prediction = model_fit.predict(typ ='linear')
    plt.show()
    # plt.plot(original_data_x,'+')
    plt.plot(series)
    plt.plot(series_prediction)
    plt.show()
    
    return(series_prediction)

def outliers_iqr(data): #Tukey fences method
    q1, q3 = np.percentile(data, [20,80])    
    iqr = q3-q1
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    return(np.where((data>upper_bound)|(data<lower_bound)))

def smooth(data):
    degree = 40
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed 

tf.disable_v2_behavior() 

"""Parameters"""
#matrix elements
# rows = 3600
# cols = 291
rows = 256
cols = 130
#ARIMA
p1, d1, q1 = [1,1,1]
p1, d1, q1 = [2,1,2]

#ML
length_x = rows
# length_latent = 3000
length_latent = 50

epochs_no = 3000
batch_size_no = 1
learning_rate = 0.001


""" Load data """
# os.chdir('house_1')

data_x = pd.read_table('aggreagated_day1.csv', header = None)
data_x = data_x.values
data_x = data_x[range(1,len(data_x)),0]
data_x = np.float32(data_x)

data_y = pd.read_table('aggreagated_day2.csv', header = None)
data_y = data_y.values
data_y = data_y[range(1,len(data_x)),0]
data_y = np.float32(data_y)



#delete outlier
call_outlier_x = outliers_iqr(data_x)
outlier_index_x = call_outlier_x[0] 
for i in range(len(outlier_index_x)):
    outlier_x = outlier_index_x[i]
    # data_x.values[outlier_x] = data_x.values[outlier_x-1] # use previous value
    data_x[outlier_x] = data_x[outlier_x-1] # use previous value

#delete outlier
call_outlier_y = outliers_iqr(data_y)
outlier_index_y = call_outlier_y[0] 
for i in range(len(outlier_index_y)):
    outlier_y = outlier_index_y[i]
    data_y[outlier_y] = data_y[outlier_y-1] # use previous value


# data_x = data_x[0:1048320]
original_data_x = data_x.reshape(-1,1)
original_data_y = data_y.reshape(-1,1)



""" Normalization"""
#x
max_v = np.array(data_x.max())
min_v = np.array(data_x.min())

data_x = data_x.reshape(-1,1)
nomal_data_x = (data_x.copy()-min_v)/(max_v-min_v)
data_x1 = nomal_data_x

#y
max_v_y = np.array(data_y.max())
min_v_y = np.array(data_y.min())

data_y = data_y.reshape(-1,1)
nomal_data_y = (data_y.copy()-min_v)/(max_v-min_v)
data_y1 = nomal_data_y

""" Preprocessing """
features = int(len(data_x)/cols)-1
matrix_data_x = np.zeros([rows,features])
for i in range(features):
    data_row = data_x1[cols*i : rows+cols*i]  
    matrix_data_x[:,i] = data_row[:,0]

data_test_x = matrix_data_x.T #training
x_train, x_test = model_selection.train_test_split(data_test_x, test_size = 0.2, random_state = 0)

features_y = int(len(data_y)/cols)-1
matrix_data_y = np.zeros([rows,features])
for i in range(features_y):
    data_row = data_y1[cols*i : rows+cols*i]  
    matrix_data_y[:,i] = data_row[:,0]

data_test_y = matrix_data_y.T #training
y_train, y_test = model_selection.train_test_split(data_test_y, test_size = 0.8, random_state = 0)


"""AE Modeling"""
start_time = time.time()
input_empty = Input(shape=length_x) # X
encoder1 = Dense(length_latent, activation=tf.keras.layers.PReLU())(input_empty) # Gn
decoder1 = Dense(length_x, activation=tf.keras.layers.PReLU())(encoder1) # Gn

# this model maps an input to its reconstruction
early_stopping = EarlyStopping(monitor='val_loss', mode='min', baseline=0.001, verbose=1, patience=10)#patience=10
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
adams = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)
sgds = tf.keras.optimizers.SGD(decay=1e-6, momentum=0.9, nesterov=True)
adagrad = tf.keras.optimizers.Adagrad(lr=learning_rate, epsilon=1e-6)

autoencoder = Model(inputs=input_empty, outputs=decoder1)
autoencoder.compile(optimizer=adagrad, loss='mean_absolute_error') # RAE 'mean_absolute_error'
# autoencoder.compile(optimizer=adagrad, loss=RAE) # RAE 'mean_absolute_error'
autoencoder.summary()
history = autoencoder.fit(x_train, x_train,
                          epochs=epochs_no,
                          batch_size=batch_size_no,
                          shuffle=True, 
                          validation_data=(x_test, x_test)) #   validation_data=(x_test, x_test)

print("fitting")

"""AE Training"""
# create encoder model
encoder_layer1 = autoencoder.layers[1]
encoder_output1 = encoder_layer1(input_empty)
encoder = Model(inputs=input_empty, outputs=encoder_output1)
encoder.compile(optimizer=adagrad, loss='mean_absolute_error')
encoder.fit(x_train)

# create decoder model
encoded_input = Input(shape=length_latent) # shape=(Gn)
decoder_layer1 = autoencoder.layers[-1]
decoder_output1 = decoder_layer1(encoded_input)
decoder = Model(inputs=encoded_input, outputs=decoder_output1)

end_time = time.time()
print(end_time-start_time)
    
    
"""AE Prediction"""
latent_vector_x = encoder.predict(data_test_x)

#check correlation for latent vectors
latent_data_x = pd.DataFrame(latent_vector_x.flatten())
# tmp_latent = pd.concat([latent_data_x, latent_data_y, latent_data_z], axis=1)
tmp_latent = latent_data_x

# corr_latent = tmp_latent.corr()
# corr_latent = corr_latent.values[0,1]

reconstructed_data_x = decoder.predict(latent_vector_x)


""" Postprocessing """
pred_data1 = reconstructed_data_x.T
post_pred_data1 = np.zeros(data_x.shape)
post_pred_data1[range(0,cols),0] = pred_data1[range(0,cols),0]

for i in range(1, pred_data1.shape[1]):
    data_tmp = pred_data1[:,i]
    post_pred_data1[range(cols*i, cols*i+rows),0] = data_tmp

post_pred_data1 = post_pred_data1.reshape(-1,1)


""" Denormalization """

denomal_post_pred_data1 = post_pred_data1[:,0]*(max_v-min_v)+min_v

post_pred_data1 = denomal_post_pred_data1
loss_mse_fin, loss_mae_fin = Loss_Plotting(data_x, post_pred_data1) #[range(1,len(data_x))]
# loss_mse_fin, loss_mae_fin = Loss_Plotting(a, b) #[range(1,len(data_x))]


post_pred_data1 = post_pred_data1.reshape(-1,1)
loss_RAE = RAE(data_x, post_pred_data1)
residual = pd.DataFrame(data_x-post_pred_data1)
# residual.to_csv("residue.csv")
plt.plot(residual, '--')

data_original = pd.DataFrame(data_x)
data_reconstruction = pd.DataFrame(post_pred_data1)
# data_original.to_csv("original.csv")
# data_reconstruction.to_csv("reconstruction.csv")

concate_r = pd.concat([residual, pd.DataFrame(post_pred_data1)], axis=1)
corr_r = concate_r.corr()
""" Loss plot """

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.scatter(data_original.values, residual.values)







