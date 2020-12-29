import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
df=pd.read_csv("C:\\Study_mat\\driver-data.csv")
print(df.head())
profile1 = df.profile_report(title='Driver data Profiling Report')

profile1.to_file(output_file="C:\\Study_mat\\driver-data-profile.html")

#no real co relation from html output

print(df.columns)
print(df.index)

df1=df.iloc[::,[1,2]]
print('df-0 -shape',df1.shape)
print(df1.head())
dist = []
K = range(1,15)
print('\n\n THE df1 shape',df1.shape[0])
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df1)
    print('\n\n\n ',kmeanModel.cluster_centers_)
    dist.append(sum(np.min(cdist(df1, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df1.shape[0])
print(dist)
# Plot the elbow
plt.plot(K, dist, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#it is evident from elbow plot that number of clusters should be 2 or max 4
km=KMeans(n_clusters=4,init='k-means++')
km.fit(df1)
print(km.labels_)
print(len(km.labels_))

print("\n\n",km.cluster_centers_)
df1['cluster_number']=km.labels_
print(df1.head())
sns.lmplot(data=df1,hue='cluster_number',y='mean_over_speed_perc',x='mean_dist_day',fit_reg=False)
plt.savefig('C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\k_means_on_driver_data.pdf')
plt.show()
