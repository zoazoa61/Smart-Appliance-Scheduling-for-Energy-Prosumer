import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


    
def kemans_cluster(X, num_cluster):
    km = KMeans(n_clusters=num_cluster, init='k-means++',random_state = 0)
    km.fit(X)
    y_km = km.fit_predict(X)
    y_pred = km.labels_
#    viz_img(y_pred)
    return(y_km, y_pred)

def elbow(X, num, w_len, feature_name):
    sse = []
    for i in range(1, num):
        km = KMeans(n_clusters=i, init='k-means++',random_state = 0)
        km.fit(X)
        y_km = km.fit_predict(X)
        sse.append(km.inertia_)
        
    plt.title("Feature: "+feature_name+" "+str(w_len))    
    plt.plot(range(1, num), sse, marker='o')
    plt.xlabel('# of cluster')
    plt.ylabel('SSE') # Sum square error
    plt.show()
    return([y_km, sse[3], w_len])  

from sklearn.metrics import silhouette_score, silhouette_samples
def Silhouette(X, num_cluster):
    silhouette_avg_list = []
    for i in range(2, num_cluster):
        km = KMeans(n_clusters=i, init='k-means++',random_state = 0)
        km.fit(X)
        y_km = km.fit_predict(X)
   
        cluster_labels = np.unique(y_km)
        n_cluster = cluster_labels.shape[0]
        silhouette_vals = silhouette_samples(X, y_km)
        y_ax_lower, y_ax_upper = 0,0
        yticks = []

        silhouette_avg = np.mean(silhouette_vals)
        silhouette_avg_list.append(silhouette_avg)
        
    plt.plot(range(2, num_cluster), silhouette_avg_list, marker='o')
    plt.xlabel('# of cluster')
    plt.ylabel('Silhouette_avg') # Sum square error
    plt.show()
    
    return(silhouette_avg_list)
    
    
os.chdir('쏘영\preprocessing')

crt_file = 'AA8동.csv'
data = pd.read_csv(crt_file,  engine='python')
del data['Unnamed: 0'] 
data_value = data['0.1'].values


data_m = np.zeros([24, int(len(data_value)/24)])

for i in range(data_m.shape[1]):
    data_m[:,i] = data_value[i*24: (i+1)*24]

""" PCA """
import sklearn
from sklearn import decomposition


import time
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf1
import tensorflow.compat.v1 as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.constraints import max_norm
from keras import backend as K
from keras.losses import mean_absolute_error
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_percentage_error
from sklearn import model_selection
tf.disable_v2_behavior() 

#ML
rows = data_m.shape[1]
length_x = rows
# length_latent = 3000
length_latent = 22
epochs_no = 500
batch_size_no = 1
learning_rate = 0.001

#Load data
data_test_x = data_m
x_train, x_test = model_selection.train_test_split(data_test_x, test_size = 0.2, random_state = 0)

start_time = time.time()
input_empty = Input(shape=length_x) # X
encoder1 = Dense(length_latent, activation=tf.keras.layers.ReLU())(input_empty) # Gn
decoder1 = Dense(length_x, activation=tf.keras.layers.ReLU())(encoder1) # Gn

# this model maps an input to its reconstruction
early_stopping = EarlyStopping(monitor='val_loss', mode='min', baseline=0.001, verbose=1, patience=10)#patience=10
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
adams = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)
sgds = tf.keras.optimizers.SGD(decay=1e-6, momentum=0.9, nesterov=True)
adagrad = tf.keras.optimizers.Adagrad(lr=learning_rate, epsilon=1e-6)

autoencoder = Model(inputs=input_empty, outputs=decoder1)
autoencoder.compile(optimizer=adams, loss='mean_absolute_error') # RAE 'mean_absolute_error'
# autoencoder.compile(optimizer=adagrad, loss=RAE) # RAE 'mean_absolute_error'
autoencoder.summary()
history = autoencoder.fit(x_train, x_train,
                          epochs=epochs_no,
                          batch_size=batch_size_no,
                          shuffle=True, 
                          validation_data=(x_test, x_test)) #   validation_data=(x_test, x_test)

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

latent_vector_x = encoder.predict(data_test_x)
#check correlation for latent vectors
latent_data_x = pd.DataFrame(latent_vector_x)
reconstructed_data_x = decoder.predict(latent_vector_x)

latent_data_x.to_csv('autoencoder_feature.csv')

for i in range(latent_data_x.shape[1]):
    plt.plot(latent_data_x.values[:,i])
plt.show()






