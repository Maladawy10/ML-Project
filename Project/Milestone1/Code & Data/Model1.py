#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
import time

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
# def Null_feature_encoder(df):
#     df_temp = df[cols].astype("str").apply(LabelEncoder().fit_transform)
#     df_temp = df_temp.where(~df.isna(), df)
#
#     df.drop(cols,axis=1,inplace=True)
#     df = pd.concat([df,df_temp],axis=1)
#     return df
def FillNanInNumericColumns(df):
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col].fillna(df[col].mean(), inplace=True)
    return df
def PredictNullStrings(df,features):
    """
    INPUTS :
        df : encoded dataframe contain nulls
        X :  list of features names (INT OR FLOAT)
   Return :
        complete df with predicted nulls
       """

    for col in df.columns:
        if df[col].isna().sum().sum():
          
            X_train = (df[features.columns]).loc[~df[col].isnull()==True]
            X_unknown = (df[features.columns]).loc[ df[col].isna() ] # select rows where the current column value is null

            Y_train = (df[col]).loc[ ~df[col].isnull() ] # select rows where the current column value isn't null
            Y_train = pd.DataFrame(Y_train)
            Y_train = Feature_Encoder(Y_train,Y_train.columns)
            frames=[X_train,Y_train]

            not_null_df=pd.concat([X_train,Y_train],axis=1,join='inner')
            
            X_train = pd.DataFrame(X_train)
            X_unknown = pd.DataFrame(X_unknown)
            # Features That Correlation is higher than average
            corr = not_null_df.corr()
        
            c_top_features = corr.index[abs(corr[col]) > corr.values.mean()]
            c_top_features = c_top_features.delete(-1)

            if len(c_top_features) :
                X_train = X_train.loc[:,c_top_features]
                X_unknown = X_unknown.loc[:,c_top_features]
                prediction = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train).predict(X_unknown)
                (df[col]).loc[~df[col].isna()] = Y_train[col].to_numpy()
                (df[col]).loc[df[col].isna()] = prediction
                df[col]=df[col].astype(int)             
    return df
def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


data = pd.read_csv('House_Data.csv')
data = data.loc[:, data.isin([' ',np.nan,'NULL']).mean() < 0.4]
data = FillNanInNumericColumns(data)
features = data[data.columns[~data.isnull().any()]] 
updated_data = PredictNullStrings(data,features)
updated_data = EncodeStringColumns(updated_data)


corr = updated_data.corr()
top_features = corr.index[abs(corr['SalePrice']) > 0.5]
Y=updated_data['SalePrice']


plt.subplots(figsize=(12, 8))
top_corr = updated_data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

top_features = top_features.delete(-1)
X = updated_data[top_features]
X = featureScaling(X,0,1)

cls = linear_model.LinearRegression()
start = time.time()
cls.fit(X,Y)
end = time.time()
prediction= cls.predict(X)
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), prediction))
print('Training time',"%.10f"%float(end-start))




