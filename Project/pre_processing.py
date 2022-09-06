import pandas as pd
import numpy as np
import ast
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
import time

#Drop cols
def drop_almost_null(data):
    dropped_cols = data.loc[:,data.isin([' ',np.nan,'NULL']).mean() > 0.4].columns
    data = data.loc[:, data.isin([' ',np.nan,'NULL']).mean() < 0.4]
    return dropped_cols,data
def FillNanInNumericColumns(data):
    df = data.copy()
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col].fillna(df[col].mean(), inplace=True)
    return df

# Encodeing
def EncodeStringColumns(df):
    temp_list = list()
    for col in df.columns:
        if df[col].dtype == 'object':
            temp_list.append(col)
    return Feature_Encoder(df, temp_list)
def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def PredictNullStrings(df, features):
    """
    INPUTS :
        df : encoded dataframe contain nulls string columns, numeric columns have no nulls
        features :  dataframe of features names (INT OR FLOAT OR STRING) with no nulls
   Return :
        complete df with predicted nulls
       """

    for col in df.columns:
        if df[col].isna().sum().sum():

            X_train = (df[features.columns]).loc[~df[col].isnull() == True]
            X_unknown = (df[features.columns]).loc[df[col].isna()]  # select rows where the current column value is null

            Y_train = (df[col]).loc[~df[col].isnull()]  # select rows where the current column value isn't null
            Y_train = pd.DataFrame(Y_train)
            Y_train = Feature_Encoder(Y_train, Y_train.columns)
            frames = [X_train, Y_train]

            not_null_df = pd.concat([X_train, Y_train], axis=1, join='inner')

            X_train = pd.DataFrame(X_train)
            X_unknown = pd.DataFrame(X_unknown)
            # Features That Correlation is higher than average
            corr = not_null_df.corr()

            c_top_features = corr.index[abs(corr[col]) > corr.values.mean()]
            c_top_features = c_top_features.delete(-1)

            # if len(c_top_features):
            X_train = X_train.loc[:, c_top_features]
            X_unknown = X_unknown.loc[:, c_top_features]
            prediction = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train).predict(X_unknown)
            (df[col]).loc[~df[col].isna()] = Y_train[col].to_numpy()
            (df[col]).loc[df[col].isna()] = prediction
            df[col] = df[col].astype(int)
    return df

def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X

def split_dict(data):
    f1 = []
    f2 = []
    f3 = []
    for row in data['MiscFeature2']:
        #     print(row)
        splt = ast.literal_eval(row)
        for key in splt.keys():
            if (key == 'f1'):
                f1.append(splt[key])
            if (key == 'f2'):
                f2.append(splt[key])
            if (key == 'f3'):
                f3.append(splt[key])
    data['f1'] = pd.DataFrame(f1)
    data['f2'] = pd.DataFrame(f2)
    data['f3'] = pd.DataFrame(f3)
    data.drop(columns=['MiscFeature2'], inplace=True)
    return data

def feature_selection(data,X):
    corr = data.corr()
    c_top_features = corr.index[abs(corr['PriceRate']) > 0.5]
    c_top_features = c_top_features.delete(-1)
    X = X.loc[:, c_top_features]
    return X

def pre_processingAll(data):
    dropped_cols , data = drop_almost_null(data)
    data = FillNanInNumericColumns(data)
    data = split_dict(data)
    data = EncodeStringColumns(data)
    features = data[data.columns[~data.isnull().any()]]
    data = PredictNullStrings(data, features)

    # Normalized_data = featureScaling(data, 0, 1)
    # print(Normalized_data.columns)
    return data
def displayMetrics(model_name,y_test, y_pred,start_time,end_time) :
    print(model_name)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    print("\tMSE : ", MSE)

    # r_2_score
    r_2_score = metrics.r2_score(y_test, y_pred)
    print("\tr_2_score : ", r_2_score)

    # Accuracy
    accuracy = np.mean(y_test == y_pred)
    print("\tAccuracy : ", accuracy)

    #time
    print("\tTime : ", end_time-start_time)
    return r_2_score,accuracy,(end_time-start_time)
