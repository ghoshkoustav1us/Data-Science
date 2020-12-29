import pandas as pd
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
bunch1=load_digits()
plt.matshow(bunch1.images[0])
plt.show()
conc=np.c_[bunch1['data'],bunch1['target']]
df=pd.DataFrame(conc)
df.to_csv('C:\\Study_mat\\digits.csv',sep=',',header=True)
features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
target=[64]

train,test=train_test_split(df,random_state=85,train_size=0.8)
train_x=train[features]
train_y=train[target]
test_x=test[features]
test_y=test[target]
Lr=LogisticRegression()
Lr.fit(train_x,train_y)
y=Lr.predict(test_x)
print(accuracy_score(y_pred=y,y_true=test_y))
sc=MinMaxScaler(feature_range=[0,1])
rescaled=sc.fit_transform(df.iloc[::,0:63])
rescaled_features_df=pd.DataFrame(data=rescaled)
pc_red=PCA()
pc_red.fit(rescaled_features_df)
plt.plot(np.cumsum(pc_red.explained_variance_ratio_))
plt.show()

#from plot we can see 95% of variance we can get from 35 features
pc_red1=PCA(n_components=35,svd_solver='full')
pc_data=pc_red1.fit_transform(rescaled_features_df)
pc_data_df=pd.DataFrame(data=pc_data)
final_df=pd.concat([pc_data_df,df.iloc[::,64]],axis=1)
train,test=train_test_split(final_df,train_size=0.8,random_state=85)
train_x=train.iloc[::,0:35]
train_y=train.iloc[::,35]
test_x=test.iloc[::,0:35]
test_y=test.iloc[::,35]
Lr.fit(train_x,train_y)
y=Lr.predict(test_x)
print(accuracy_score(y_true=test_y,y_pred=y))
#2% accuracy score incresed with PCA of 35 features -- reducing over fitting
actual_y_ser=pd.Series(test_y)
actual_y_ser.reset_index(drop=True,inplace=True)
y_ser=pd.Series(y)

cf=confusion_matrix(y_pred=y_ser,y_true=actual_y_ser)
print(cf)

compared_df=pd.concat([y_ser,actual_y_ser],axis=1)
#print(compared_df.columns)
compared_df["predicted_val"],compared_df["Actual_val"]=compared_df.iloc[::,0],compared_df.iloc[::,1]
compared_df.drop([0,64],axis=1,inplace=True)

compared_df1=compared_df[compared_df['predicted_val']!=compared_df['Actual_val']]
fig,ax=plt.subplots(ncols=1,nrows=2,sharex=True)
sns.barplot(data=compared_df1,x=compared_df1.index.values,y='Actual_val',ax=ax[0])
sns.barplot(data=compared_df1,x=compared_df1.index.values,y='predicted_val',ax=ax[1])
plt.show()
