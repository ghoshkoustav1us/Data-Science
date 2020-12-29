import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split

df1=pd.read_csv('C:\\Study_mat\\loan_borowwer_data.csv',sep=',',header='infer')
print(df1.head(5))
le=LabelEncoder()
df1['purpose']=le.fit_transform(df1['purpose'])
print(df1['purpose'])
sc=StandardScaler()
sc.fit(X=df1.loc[::,['int.rate','installment','log.annual.inc','dti','fico','days.with.cr.line','	revol.bal','revol.util']])
print(df1.head(10))
lr=LogisticRegression()
features=['int.rate','installment','log.annual.inc','dti','fico','days.with.cr.line','revol.bal','revol.util','inq.last.6mths','delinq.2yrs','pub.rec']
dependent=['not.fully.paid']
train,test=train_test_split(df1,train_size=0.85)

train_x=train[features]
train_y=train[dependent]
test_x=test[features]
test_y=test[dependent]

lr.fit(X=train_x,y=train_y)
y=lr.predict(X=test_x)
print("RMSE : ",sqrt(mean_squared_error(y_pred=y,y_true=test_y)))
print("Accuracy Score : ",accuracy_score(y_true=test_y,y_pred=y))

sns.pairplot(data=df1,x_vars=['int.rate','installment','log.annual.inc','dti','fico','days.with.cr.line','revol.bal','revol.util'],y_vars=dependent,kind='reg')
plt.show()
