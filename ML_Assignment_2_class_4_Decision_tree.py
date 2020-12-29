import pandas as pd
from math import sqrt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import pca
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv('C:\\Study_mat\\horse.csv',sep=',',header='infer')
print('Number of Row in Df: ',len(df.index))
print('Number of col in Df: ',len(df.columns))
categorical_features=['surgery','age','peripheral_pulse','mucous_membrane','capillary_refill_time',
'pain','peristalsis','abdominal_distention','nasogastric_tube','rectal_exam_feces','abdomen',
'abdomo_appearance','outcome','surgical_lesion','nasogastric_reflux'];
dependant=['cp_data'];
if (df.isnull().values.any()) or (df.isna().values.any()):
    print("we have nulls in dataframe")
    df.fillna('unk',inplace=True,axis=1)
    im=SimpleImputer(strategy='most_frequent')
    im.fit_transform(df.loc[::,categorical_features])
else :
    pass
print (df.loc[0:10:1,'rectal_temp'])

for i,val  in enumerate(categorical_features):
    colname=str(val)
    df.loc[::,colname]=LabelEncoder().fit_transform(df.loc[::,colname])

df.loc[::,'cp_data']=LabelEncoder().fit_transform(df.loc[::,'cp_data'])

df2=df.loc[::,['surgery','age','peripheral_pulse','mucous_membrane','capillary_refill_time',
'pain','peristalsis','abdominal_distention','nasogastric_tube','rectal_exam_feces','abdomen',
'abdomo_appearance','outcome','surgical_lesion','nasogastric_reflux','cp_data']]

train,test=train_test_split(df2,train_size=0.75,random_state=5)
train_x=train[categorical_features]
train_y=train[dependant]
test_x=test[categorical_features]
test_y=test[dependant]
DT=DecisionTreeClassifier()
DT.fit(train_x,train_y);
y=DT.predict(test_x)
print('RMSE :',sqrt(mean_squared_error(y_pred=y,y_true=test_y)))
print('Accuracy score :',accuracy_score(y_true=test_y,y_pred=y))

RF=RandomForestClassifier(n_estimators=40,criterion='gini',max_features='log2')
RF.fit(train_x,train_y);
y=RF.predict(test_x)
print('RMSE :',sqrt(mean_squared_error(y_pred=y,y_true=test_y)))
print('Accuracy score :',accuracy_score(y_true=test_y,y_pred=y))

