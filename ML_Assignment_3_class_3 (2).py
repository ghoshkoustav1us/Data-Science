import pandas as pd
from scipy import stats
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import pca
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('C:\\Study_mat\\FyntraCustomerData.csv',sep=',',header='infer')

jg=sns.JointGrid(data=df,x='Time_on_Website',y='Yearly_Amount_Spent')
jg=jg.plot_joint(sns.regplot,color='red')
jg=jg.plot_marginals(sns.distplot,color='Blue')
jg=jg.annotate(stats.pearsonr)
plt.show()
jg=sns.JointGrid(data=df,x='Time_on_App',y='Yearly_Amount_Spent')
jg=jg.plot_joint(sns.regplot,color='red')
jg=jg.plot_marginals(sns.distplot,color='Blue')
jg=jg.annotate(stats.pearsonr)
plt.show()
#Time on app is more co-related as we see from pearson value
sns.pairplot(data=df,hue='Avatar',x_vars=['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership'],y_vars=['Yearly_Amount_Spent'],kind='scatter')
plt.show()
df1=df.corr()
sns.heatmap(data=df1,cmap='vlag',linewidths=0.5,annot=True)
plt.show()
sns.lmplot(data=df,x='Length_of_Membership',y='Yearly_Amount_Spent',scatter=True)
plt.show()
#hence proved Length of Membership is very highly co -related to Yearly amount spent( co-relation=0.81)
sns.residplot(data=df,x='Length_of_Membership',y='Yearly_Amount_Spent')
plt.show()
df_model=df.loc[::,['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership','Yearly_Amount_Spent']]
train,test=train_test_split(df_model,train_size=0.75,random_state=85)
train_x=train[['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership']]
train_y=train['Yearly_Amount_Spent']
test_x=test[['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership']]
test_y=test['Yearly_Amount_Spent']
lr=LinearRegression()
lr.fit(train_x,train_y)
y=lr.predict(test_x)
print("RMSE :",sqrt(mean_squared_error(y_true=test_y,y_pred=y)))
#RMSE value printed above it is 10 for 125 sample , so quite good
pred=pd.DataFrame(data=y,columns=['Yearly_Amount_Spent_pred'])
actual=pd.DataFrame(data=test_y)
actual.reset_index(drop=True,inplace=True)
print(len(actual.index.values),len(pred.index.values))
combined=pd.concat([actual,pred],join='inner',axis=1)
sns.scatterplot(x='Yearly_Amount_Spent',y='Yearly_Amount_Spent_pred',data=combined)
plt.show()
combined['difference']=combined['Yearly_Amount_Spent']-combined['Yearly_Amount_Spent_pred']
print(len(combined [combined['difference'] <=2.0]))
#out of 125 sample 68 has less 2 value difference between predicted and Actual-- so yes matching pretty much
