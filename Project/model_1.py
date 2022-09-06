import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import os
import time

import pre_processing


def Logistic_Reg(df) :

    X = df.drop(columns=['PriceRate'])
    Y = df['PriceRate']

    X = pre_processing.feature_selection(df,X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0,shuffle=True)

    logisticRegr = LogisticRegression()
    # FIT Train Data
    start = time.time()
    logisticRegr.fit(x_train, y_train)
    end = time.time()
    pickle.dump(logisticRegr,open('LogisticReg.sav', 'wb'))
    # Predict Test Data
    y_pred = logisticRegr.predict(x_test)

    return pre_processing.displayMetrics("Model 1 : Logistic regression",y_test, y_pred,start,end)
