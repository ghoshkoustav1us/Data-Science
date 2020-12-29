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

df=pd.read_csv('C:\\Study_mat\\breast-cancer-data.csv',sep=',',header='infer')

features=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
'symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
'concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
target=['diagnosis']

Lbl=LabelEncoder()
df.diagnosis=Lbl.fit_transform(df['diagnosis'])

train,test=train_test_split(df,train_size=0.8,random_state=85)
train_x=train[features]
train_y=train[target]
test_x=test[features]
test_y=test[target]
Lr=LogisticRegression(penalty='l2',solver='lbfgs')
Lr.fit(train_x,train_y)
y=np.ravel(Lr.predict(test_x))
print("accuracy score before PCA : ",accuracy_score(y_true=np.ravel(test_y),y_pred=y))

sc=MinMaxScaler(feature_range=[0,1])
rescaled=sc.fit_transform(df[features])
rescaled_features_df=pd.DataFrame(data=rescaled)
pc_red=PCA()
pc_red.fit(rescaled_features_df)
plt.plot(np.cumsum(pc_red.explained_variance_ratio_))
plt.show()
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
PCA_var_ratios = pc_red.explained_variance_ratio_
number_of_comp=select_n_components(PCA_var_ratios, 0.95)
print(number_of_comp)
#plot shows 10 features are enough to get 97% explained variance ratio.so we for 15 component PCA

pc_red1=PCA(n_components=10,svd_solver='full')
pc_data=pc_red1.fit_transform(rescaled_features_df)
pc_data_df=pd.DataFrame(data=pc_data)
final_df=pd.concat([pc_data_df,df.loc[::,'diagnosis']],axis=1)

train,test=train_test_split(final_df,train_size=0.8,random_state=85)
train_x=train.iloc[:,0:14]
train_y=train[target]
test_x=test.iloc[::,0:14]
test_y=test[target]

Lr.fit(train_x,train_y)
y=Lr.predict(test_x)
print("accuracy score after PCA: ",accuracy_score(y_true=test_y,y_pred=y))
