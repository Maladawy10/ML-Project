#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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
def Null_feature_encoder(df):
    df_temp = df[cols].astype("str").apply(LabelEncoder().fit_transform)
    df_temp = df_temp.where(~df.isna(), df)

    df.drop(cols,axis=1,inplace=True)
    df = pd.concat([df,df_temp],axis=1)
    return df
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

top_features = top_features.delete(-1)
X = updated_data[top_features]
X = featureScaling(X,0,1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

rmses = []
degrees = np.arange(1, 6)
min_rmse, min_deg = 1e10, 0
st=time.time()
for deg in degrees:

    # Train features
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly_train = poly_features.fit_transform(x_train)

    # Linear regression
    poly_reg = linear_model.LinearRegression()
    poly_reg.fit(x_poly_train, y_train)

    # Compare with test updated_data
    x_poly_test = poly_features.fit_transform(x_test)
    poly_predict = poly_reg.predict(x_poly_test)
    poly_mse = metrics.mean_squared_error(y_test, poly_predict)
    poly_rmse = np.sqrt(poly_mse)
    rmses.append(poly_rmse)
    # Cross-validation of degree
    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg
en=time.time()

print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))
print('Polynomial Training time',en-st)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(degrees, rmses)
ax.set_yscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('RMSE')
plt.show()


# In[ ]:




