import pandas as pd
from math import sqrt
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.decomposition import PCA
from sklearn import metrics

df1=pd.read_csv('C:\\Study_mat\\voice.csv',sep=',',header='infer')
print(df1.head(5))
#dict1={'male':0,'female':1}
#df1.label=df1.label.map(dict1)
#print(df1.iloc[2500:2575:1,-1])
df1.iloc[::,-1]=LabelEncoder().fit_transform(df1.loc[::,'label'])
#df2=OneHotEncoder(categorical_features=[-1]).fit_transform(df1).toarray()
#print(type(df2))
#print(df2)
df2=df1.loc[::,['kurt','sp.ent','sfm','mode','centroid','dfrange','label']]
print(df2.head(5))
train,test=ttsplit(df2,train_size=0.8)
print(train.shape)
print(test.shape)
features=['kurt','sp.ent','sfm','mode','centroid','dfrange']
dependent=['label']
train_x=train[features]
train_y=train[dependent]
test_x=test[features]
test_y=test[dependent]
##############################
logistic=LogisticRegression(solver='sag')
logistic.fit(X=train_x,y=train_y);
print('r2 val is :' ,logistic.score(X=train_x,y=train_y));
y=logistic.predict(test_x);
print('intercept is : ',logistic.intercept_);
print('Accuracy score: ',metrics.accuracy_score(y_pred=y,y_true=test_y))
print("RMSE: ",sqrt(metrics.mean_squared_error(y_true=test_y,y_pred=y)))

#RMSE val is more than 0.5 and R2 is less than 0.7 -- so model has flaw
#removing multi co-linearity
vdf=df2.corr()
sns.set_palette('colorblind')
sns.heatmap(data=vdf,cbar=True,linewidths=0.5,annot=True,cmap='vlag')
plt.show()
#clearly from heatmap
#1) sfm is highly co-linear with sp.ent
#2) sfm is highly co-linear with centroid
#therefore keeping only one independent  from co-linear pair of features we remove sfm
print(df2.head(10))
df2.drop('sfm',inplace=True,axis=1)
features.remove('sfm')
print(features)
train1,test1=ttsplit(df2,train_size=0.8)
train_x=train1[features]
train_y=train1[dependent]
test_x=test1[features]
test_y=test1[dependent]

logistic_improved=LogisticRegression()
logistic_improved.fit(train_x,train_y)
print("r2 val is :",logistic_improved.score(X=train_x,y=train_y))
y1=logistic_improved.predict(test_x)
print("Imporved accuracy score : ",metrics.accuracy_score(y_true=test_y,y_pred=y1))
print("RMSE : ",sqrt(metrics.mean_squared_error(y_pred=y1,y_true=test_y)));
#removing multi-colinearity decreased RMSE and incresed R2
