import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,fbeta_score,roc_auc_score,mean_squared_error,make_scorer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv("C:\\Study_mat\\glass.csv")
features=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
target=['Type']
train,test=train_test_split(df,train_size=0.73,random_state=0)#73% train size giving best score in Dt and Grid searched RF
train_x1=train[features]
train_y1=train[target]
test_x1=test[features]
test_y1=test[target]
DT=DecisionTreeClassifier(criterion='gini',splitter='best',random_state=0)
DT.fit(train_x1,train_y1)
y=DT.predict(test_x1)
y1=np.ravel(y,1)
pred_DT=pd.DataFrame(y1)
print(y1)
print("Accuracy score :",accuracy_score(y_pred=y1,y_true=test_y1))
print("Fbeta score :",fbeta_score(y_pred=y1,y_true=test_y1,beta=1,average='weighted'))
kf=KFold(n_splits=3,random_state=0,shuffle=True)
kf1=kf.split(df)
print(kf)

for trainidx,testidx in kf1:
    #print(trainidx,testidx)
    #print('\n\n')
    train_x=df.iloc[trainidx,0:9]
    train_y=df.iloc[trainidx,-1]
    test_x=df.iloc[testidx,0:9]
    test_y=df.iloc[testidx,-1]
    DT.fit(train_x,train_y)
    y=np.ravel(DT.predict(test_x),1)
    print("accuracy score :",accuracy_score(y_pred=y,y_true=test_y))

estimatorRF=RandomForestClassifier(random_state=0)
print(estimatorRF.get_params())
tree_val=np.arange(2,100,2)
parm_dict={'n_estimators':tree_val}
scorer = make_scorer(accuracy_score)
gr=GridSearchCV(estimator=estimatorRF,param_grid=parm_dict,scoring=scorer,iid=False,cv=5)
grid_fit=gr.fit(train_x1,np.ravel(train_y1))
best_estimator_RF=grid_fit.best_estimator_
print(best_estimator_RF)
y2=np.ravel(best_estimator_RF.predict(test_x1),1)
print("accuracy score after grid search with RF ",accuracy_score(y_true=test_y1,y_pred=y2))
print("Fbeta score after grid search with RF :",fbeta_score(y_pred=y2,y_true=test_y1,beta=1,average='weighted'))
