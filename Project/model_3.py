from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pre_processing
# from pre_processing import *
import pandas as pd
import pickle
import numpy as np
import time

def SVM(cleaned_data) :

    Y = cleaned_data["PriceRate"]
    X = cleaned_data.drop(columns = ["PriceRate"])
    X = pre_processing.feature_selection(cleaned_data,X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=None, shuffle=True)

    SVcmodel = svm.SVC(kernel='poly',degree=2, C=10)
    start = time.time()
    SVcmodel.fit(x_train, y_train)
    end = time.time()
    pickle.dump(SVcmodel, open('SVC.sav', 'wb'))
    y_pred = SVcmodel.predict(x_test)

    return pre_processing.displayMetrics("Model 3 : Support vector machine",y_test, y_pred,start,end)