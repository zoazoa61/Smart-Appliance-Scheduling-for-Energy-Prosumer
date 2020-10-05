import xgboost as xgb
import sklearn
import numpy as np
from sklearn.model_selection import GridSearchCV

def modelfit(alg, train_data_features, train_labels,useTrainCV=True, cv_folds=5):

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

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % sklearn.metrics.accuracy_score(train_labels, dtrain_predictions))

xgb1 = xgb.sklearn.XGBClassifier(
 learning_rate =0.1,
 n_estimators=50,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective='multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)    


X=np.random.normal(size=(200,30))
y=np.random.randint(0,5,200)

modelfit(xgb1, X, y)