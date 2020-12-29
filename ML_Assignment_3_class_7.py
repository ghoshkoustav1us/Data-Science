from math import sqrt
import numpy as np
import seaborn as sns
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import cross_validate,GridSearchCV,train_test_split
df=pd.read_csv("C:\\Study_mat\\zoo.csv")
print(df.info())
print(df.head(5))
profile1 =pp.ProfileReport(df,title='Zoo data Profiling Report')
profile1.to_file(output_file="C:\\Study_mat\\zoo-data-profile.html")

#no real co relation from html output except milk vs eggs ( negetive co-relation ) and hair vs eggs ( negative co-relation -- they are perfectly expected
l=df['class_type'].unique()
print("Number of Unique class",len(l))
print(df['class_type'].value_counts())
sns.countplot(data=df,y='class_type',orient='v',palette="twilight")
plt.savefig("C:\\Study_mat\\count_by_class.pdf")
plt.show()
ag=AgglomerativeClustering(n_clusters=7, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='average', pooling_func='deprecated',
distance_threshold=None)
features=['hair','feathers','milk','eggs','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
ag.fit_predict(df.loc[::,features])
df['class_predicted']=ag.labels_
print(df)
print(ag.get_params())
print("RMSE :",sqrt(mean_squared_error(y_true=df['class_type'],y_pred=df['class_predicted'])))
print("ACC %:",accuracy_score(y_true=df['class_type'],y_pred=df['class_predicted'])*100)
#RMSE is 2.06 with linkage='average' , this is minimum RMSE possible
df2=df[df['class_type']==df['class_predicted']]
print(len(df2.index))
#train,test=train_test_split(df,train_size=0.8,random_state=5)
#train_x=train.iloc[::,1:17]
#train_y=train['class_type']
#test_x=test.iloc[::,1:17]
#test_y=test['class_type']
parm_dict={'affinity': ['euclidean','l1','l2','cosine','manhattan'], 'compute_full_tree': ['auto'], 'linkage': ["ward", "complete", "average", "single"], 'n_clusters':np.arange(1,10,1)}
#ag_1=AgglomerativeClustering()
#gr=GridSearchCV(estimator=ag_1,n_jobs=3,param_grid=parm_dict,iid=False,cv=2,scoring='f1_samples')
#print(gr)
#gr.fit(X=train_x,y=train_y)
#be=gr.best_estimator_
