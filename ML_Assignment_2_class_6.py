import pandas as pd
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df=pd.read_csv('C:\\Study_mat\\run_or_walk.csv',sep=',',header='infer')
col_to_del=['date','time','username']
for i,val in enumerate(col_to_del):
    df.drop(val,inplace=True,axis=1)
#print('dataframe without non features col\n\n: ',df)


x_var=['wrist','acceleration_x','acceleration_y','acceleration_z','gyro_x','gyro_y','gyro_z']
y_var=['activity']
train,test=train_test_split(df,train_size=0.75,random_state=85)
train_x=train[x_var]
train_y=train[y_var]
test_x=test[x_var]
test_y=test[y_var]
nb=GaussianNB()
nb.fit(train_x,train_y)
y=nb.predict(test_x)
print("Accuracy score overall: ",accuracy_score(y_true=test_y,y_pred=y))
decode_activity={0:'walk',1:'run'}
df['activity_label']=df['activity'].map(decode_activity)
set_decode_act=set(df.activity_label.tolist())
print(set_decode_act)
clf_report=classification_report(y_pred=y,y_true=test_y,target_names=set_decode_act)
print(clf_report)
segment_val_list=[['acceleration_x','acceleration_y','acceleration_z'],['gyro_x','gyro_y','gyro_z']]
for i,val in enumerate(segment_val_list):
    x_var=val
    train_x=train[x_var]
    train_y=train[y_var]
    test_x=test[x_var]
    test_y=test[y_var]
    nb=GaussianNB()
    nb.fit(train_x,train_y)
    y=nb.predict(test_x)
    print("Accuracy score %s  : "%str(val),accuracy_score(y_true=test_y,y_pred=y))
    clf_report=classification_report(y_pred=y,y_true=test_y,target_names=set_decode_act)
    print("Classification Report with %s \n\n:"%str(val),clf_report)
#gyro giving very poor prediction - so accelarations are strong predictors
