import pre_processing
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import time

def Dec_tree (df) :
    X = df.drop(columns=['PriceRate'])
    Y = df['PriceRate']

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),algorithm="SAMME",n_estimators=200)
    start = time.time()
    bdt.fit(x_train,y_train)
    end = time.time()
    pickle.dump(bdt, open('DecTree.sav', 'wb'))

    y_pred = bdt.predict(x_test)

    return pre_processing.displayMetrics("Model 2 : decision tree",y_test, y_pred,start,end)