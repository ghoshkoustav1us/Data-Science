import pandas as pd
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,completeness_score
# Create a function
def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components
bunch1=load_digits()
plt.matshow(bunch1.images[0])
plt.show()
conc=np.c_[bunch1['data'],bunch1['target']]
df=pd.DataFrame(conc)
sns.heatmap(data=df.corr())
plt.show()
#feature 9 and 54 are co-linear so removing one of them
df.drop(54,axis=1,inplace=True)
print(df)
features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63]
target=[64]
Lr=LogisticRegression()
sc=MinMaxScaler(feature_range=[0,1])
rescaled=sc.fit_transform(df)
rescaled_features_df=pd.DataFrame(data=rescaled)
pc_red=PCA()
pc_red.fit(rescaled_features_df)
plt.plot(np.cumsum(pc_red.explained_variance_ratio_))
plt.show()
pca_var_ratios = pc_red.explained_variance_ratio_
number_of_comp=select_n_components(pca_var_ratios, 0.95)
print(number_of_comp)
#from plot we can see 95% of variance we can get from 35 features
pc_red1=PCA(n_components=30,svd_solver='full')
pc_data=pc_red1.fit_transform(rescaled_features_df)
pc_data_df=pd.DataFrame(data=pc_data)
print(pc_data_df)
final_df=pd.concat([pc_data_df,df.loc[::,64]],axis=1)
train,test=train_test_split(final_df,train_size=0.8,random_state=85)
train_x=train.iloc[::,0:30]
train_y=train.iloc[::,30]
test_x=test.iloc[::,0:30]
test_y=test.iloc[::,30]
Lr.fit(train_x,train_y)
y=Lr.predict(test_x)

print("Accuracy Score on PCA:",accuracy_score(y_true=test_y,y_pred=y))

actual_y_ser=pd.Series(test_y)
actual_y_ser.reset_index(drop=True,inplace=True)
y_ser=pd.Series(y)
print(df)
sns.heatmap(data=df.corr(),cmap='vlag_r',linewidths=0.5)
#more co-linearity ,removing 55
df.drop(55,axis=1,inplace=True)
features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,56,57,58,59,60,61,62,63]
target=[64]
plt.show()
train,test=train_test_split(df,train_size=0.8)
train_x=train[features]
train_y=train[target]
test_x=test[features]
test_y=test[target]
ld=LinearDiscriminantAnalysis(n_components=None)
transformed_X_trained=ld.fit(train_x,train_y)
lda_var_ratios = ld.explained_variance_ratio_
print(lda_var_ratios)
number_of_comp=select_n_components(lda_var_ratios, 0.95)
print(number_of_comp)
df.to_csv('C:\\Study_mat\\digit_resut.csv',sep=',',header='infer')
ld=LinearDiscriminantAnalysis(n_components=number_of_comp,solver='svd')
transformed_X_trained=ld.fit_transform(train_x,train_y)
transformed_X_test=ld.transform(test_x)

Lr.fit(transformed_X_trained,train_y)
y=Lr.predict(transformed_X_test)
print("Accuracy Score on LDA:",accuracy_score(y_true=y,y_pred=y))

#LDA scores are higher than PCA score
