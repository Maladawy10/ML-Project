import pre_processing
import pandas as pd
import model_1
import model_2
import model_3
import test_script
import matplotlib.pyplot as plt

data = pd.read_csv('House_Data_Classification.csv')
cleaned_data = pre_processing.pre_processingAll(data)

print("Logistic Regression : ")
_, Lm_acc, Lm_train_time = model_1.Logistic_Reg(cleaned_data)

print("De Tree: ")
_,Dec_acc,Dec_train_time=model_2.Dec_tree(cleaned_data)

print("SVM: ")
_,SVM_acc,SVM_train_time= model_3.SVM(cleaned_data)

print('-----------------------------------------')
print('pickle models')
Lm_test_time,SVM_test_time,Dec_test_time = test_script.run_test_script(cleaned_data)
print('-----------------------------------------')

print("Logistic Reg test time : ",Lm_test_time)
print("SVM test time : ",SVM_test_time)
print("decision tree test time : ",Dec_test_time)
fig = plt.figure(figsize=(6,5))
plt.bar(['Lm_test_time','SVM_test_time','Dec_test_time'],[Lm_test_time,SVM_test_time,Dec_test_time])
plt.xlabel('Models')
plt.ylabel('Test Time')
plt.show()

fig = plt.figure(figsize=(5,5))
plt.bar(['Lm_train_time','SVM_train_time','Dec_train_time'],[Lm_train_time,SVM_train_time,Dec_train_time])
plt.xlabel('Models')
plt.ylabel('Train Time')
plt.show()

fig = plt.figure(figsize=(5,5))
plt.bar(['Lm_Acc','SVM_Acc','Dec_Acc'],[Lm_acc,SVM_acc,Dec_acc])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()