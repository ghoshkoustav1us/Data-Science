import pandas as pd
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv('C:\\Study_mat\\voice-classification.csv',sep=',',header='infer')
lbl=LabelEncoder()
df.label=lbl.fit_transform(df.label)
print(df)
train,test=train_test_split(df,train_size=0.85,random_state=5)
train_x=train.iloc[::,0:20]
train_y=train['label']
test_x=test.iloc[::,0:20]
test_y=test['label']
sv=SVC(kernel='linear',gamma=10,C=2)
sv.fit(train_x,train_y)
y=sv.predict(test_x)
print("Model accuracy score with SVM: " ,accuracy_score(y_true=test_y,y_pred=y))
nb=GaussianNB()
nb.fit(train_x,train_y)
y=nb.predict(test_x)
y_df=pd.DataFrame(y)
tot=len(y_df.index)
test_y_df=pd.DataFrame(test_y)
test_y_df.reset_index(drop=True,inplace=True)
comb=pd.concat([test_y_df,y_df],axis=1,ignore_index=False)
comb_corr_pred=comb[comb.label==comb[0]]
ac=len(comb_corr_pred.index)
print("manual accuracy check : ",(ac*100)/tot)
print(len(test_y_df.index))
print ("Model accuracy score with Naive Bayes : ",accuracy_score(y_true=test_y,y_pred=y))
cnf=confusion_matrix(y_pred=y,y_true=test_y)
print(cnf)
tn, fp, fn, tp = cnf.ravel()
print(tn, fp, fn, tp )
print("confusion matrix precision : ",tp/(tp+fp))
print("confusion matrix Recall : ",tp/(tp+fn))
print("confusion matrix Accuracy",(tp+tn)/(tp+tn+fp+fn))



