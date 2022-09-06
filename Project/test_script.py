import pickle
import os
import pandas as pd

import model_1
import model_2
import model_3
import pre_processing
import time

# data = pd.read_csv('House_Data_Classification.csv')
# cleaned_data = pre_processing.pre_processingAll(data)

def run_test_script(cleaned_data):
    if(not os.path.exists('LogisticReg.sav')):
        model_1.Logistic_Reg(cleaned_data)
    else :
        start = time.time()
        lm = pickle.load(open('LogisticReg.sav','rb'))
        y_pred = lm.predict(pre_processing.feature_selection(cleaned_data,cleaned_data.drop(columns=['PriceRate'])))
        end = time.time()
        Lm_test_time = end-start
        pre_processing.displayMetrics("Model 1 : Logistic regression",cleaned_data["PriceRate"], y_pred,start,end)

    if (not os.path.exists('DecTree.sav')):
        model_2.Dec_tree(cleaned_data)
    else :
        start = time.time()
        lm = pickle.load(open('DecTree.sav', 'rb'))
        y_pred = lm.predict(cleaned_data.drop(columns=['PriceRate']))
        end = time.time()
        DecTree_test_time = end - start
        pre_processing.displayMetrics("Model 2 : decision tree", cleaned_data["PriceRate"], y_pred,start,end)

    if (not os.path.exists('SVC.sav')):
        model_3.SVM(cleaned_data)
    else :
        start = time.time()
        svm = pickle.load(open('SVC.sav', 'rb'))
        y_pred = svm.predict(pre_processing.feature_selection(cleaned_data, cleaned_data.drop(columns=['PriceRate'])))
        end = time.time()
        SVM_test_time = end - start
        pre_processing.displayMetrics("Model 3 : Support vector machine", cleaned_data["PriceRate"], y_pred,start,end)

    return Lm_test_time,SVM_test_time,DecTree_test_time