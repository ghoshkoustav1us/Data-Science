import pandas as pd
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv('C:\\Study_mat\\college.csv',sep=',',header='infer')
lbl=LabelEncoder()
df.Private=lbl.fit_transform(df['Private'])
features=['Apps','Accept','Enroll','Top10perc','Top25perc','F.Undergrad','P.Undergrad','Outstate','Room.Board','Books','Personal','PhD','Terminal','S.F.Ratio','perc.alumni','Expend','Grad.Rate']
target=['Private']


sv=SVC(kernel='rbf',gamma=50,C=2)
train,test=train_test_split(df,train_size=0.8,random_state=5)
train_x=train[features]
train_y=train[target]
test_x=test[features]
test_y=test[target]
sv.fit(train_x,np.ravel(train_y))
y=sv.predict(test_x).ravel()
print("Model accuracy score with SVM without scaling: " ,accuracy_score(y_true=test_y,y_pred=y))

sc=StandardScaler()
df_arr=sc.fit_transform(X=df.iloc[::,1:18])
df1=pd.DataFrame(df_arr,columns=features)
df2=pd.concat([df1,df['Private']],ignore_index=False,axis=1)
sum1=df_arr.mean(axis=0)#due to integer data mean is very very tends to zero but not exactly zero
print('sum is :',sum1)
#sv=SVC(kernel='rbf',gamma=50,C=2)
sv=SVC(kernel='rbf',gamma='auto_deprecated',C=1.0)
train,test=train_test_split(df2,train_size=0.8,random_state=5)
train_x=train[features]
train_y=train[target]
test_x=test[features]
test_y=test[target]
sv.fit(train_x,np.ravel(train_y))
y=sv.predict(test_x).ravel()
print("Model accuracy score with SVM with scaling: " ,accuracy_score(y_true=test_y,y_pred=y))
parameters_list={'kernel':('linear', 'rbf','poly'), 'C':[1, 10],'gamma':[1,50]}
generic_svc=SVC(gamma='scale')
gr=GridSearchCV(generic_svc,cv=6,iid=False,param_grid=parameters_list)
print(gr)
gr.fit(train_x,np.ravel(train_y))
be=gr.best_estimator_
y=be.predict(test_x)
print("Model accuracy score with SVM with scaling: " ,accuracy_score(y_true=test_y,y_pred=y))
