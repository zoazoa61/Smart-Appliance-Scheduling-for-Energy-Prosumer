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

#Call AE features
crt_file_AE = 'autoencoder_feature.csv'
data_AE = pd.read_csv(crt_file_AE,  engine='python')
del data_AE['Unnamed: 0'] 

#Call Original files
crt_file = 'AA8동.csv'
data = pd.read_csv(crt_file,  engine='python')
del data['Unnamed: 0'] 
data_value = data['0.1'].values


""" Normalization """
def Normalized(data_value):
    data_value_max = data_value.max()
    data_value_min = data_value.min()
    normalized_data_value = (data_value-data_value_min)/(data_value_max-data_value_min)
    return normalized_data_value

data_value = Normalized(data_value)


data_m = np.zeros([24, int(len(data_value)/24)])
for i in range(data_m.shape[1]):
    data_m[:,i] = data_value[i*24: (i+1)*24]
data_m = pd.DataFrame(data_m)

data_concat = pd.concat([data_m, data_AE], axis=1)
data_concat = data_concat.T

""" Clustering"""
clustering_no = data_AE.shape[1]
clustering_no = 22
#elbow(data_concat, clustering_no, 1, "AE") #poly
#eval_Silhouette = Silhouette(data_concat, clustering_no)

cluster_fin, cluster_pred = kemans_cluster(data_concat, clustering_no)
cluster_fin = cluster_fin[0:data_m.shape[1]].reshape(1,-1)


from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, XGBModel
from xgboost import plot_importance, plot_tree

def rmsle(y, y_pred): 
    return np.sqrt(mean_squared_error(y, y_pred))

#xgb_modelfit(xgb1, X_train, y_train, x_test, y_test)
def xgb_modelfit(alg, train_data_features, train_labels, test_data_features, test_labels, useTrainCV=True, cv_folds=5):

    if useTrainCV:
        params=alg.get_xgb_params()
        xgb_param=dict([(key,[params[key]]) for key in params])

        boost = xgb.sklearn.XGBClassifier()
        cvresult = GridSearchCV(boost,xgb_param,cv=cv_folds)
        cvresult.fit(X,y)
        alg=cvresult.best_estimator_

    #Fit the algorithm on the data
    alg.fit(train_data_features, train_labels)

    #Predict training set:
    dtrain_predictions = alg.predict(train_data_features)
    dtrain_predprob = alg.predict_proba(train_data_features)[:,1]
    
    dtest_predictions = alg.predict(test_data_features)
    dtest_predprob = alg.predict_proba(test_data_features)[:,1]


    #Print model report:
    print("\nModel Report")
    print("Accuracy_train : %.4g" % sklearn.metrics.accuracy_score(train_labels, dtrain_predictions))
    print("RMSE_train : %.4g" % rmsle(train_labels, dtrain_predictions))
    print("Accuracy_test : %.4g" % sklearn.metrics.accuracy_score(test_labels, dtest_predictions))
    print("RMSE_test : %.4g" % rmsle(test_labels, dtest_predictions))

    return(dtrain_predictions, dtest_predictions)

X = data_m.values.T
y = cluster_fin.T
#y = y.astype(float).reshape(-1)
y = y.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

xgb1 = xgb.XGBClassifier(learning_rate =0.01,
                         n_estimators=300,
                         max_depth=3, 
                         min_child_weight=1, 
                         gamma=0, 
                         subsample=0.8, 
                         colsample_bytree=0.8, 
                         objective='multi:softmax', 
                         nthread=4, 
                         scale_pos_weight=1, 
                         seed=27)    

y_train_hat, y_test_hat = xgb_modelfit(xgb1, X_train, y_train, X_test, y_test)

X_test_hat = np.zeros([X_test.shape[0], X_test.shape[1]])

for i in range(len(y_test_hat)):
    X_test_hat[i,:] = data_AE.values[:,y_test[i]] 

#X_test_hat =  Normalized(X_test_hat)


""" Plotting """
for i in range(X_test.shape[0]):
    plt.plot(X_test[i,:])
    plt.plot(X_test_hat[i,:])
    plt.show()
#
for i in range(X_test.shape[0]):
#    plt.plot(X_test_hat[i,:])
    plt.plot(X_test[i,:])
plt.show()



plt.plot(X_test.flatten())
plt.plot(X_test_hat.flatten())
plt.show()








