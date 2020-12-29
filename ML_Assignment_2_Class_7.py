import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import PIL
from pandas.plotting import  table


p1="C:\\Study_mat\\dogs.jpeg"
img = PIL.Image.open(p1).convert("L")
imgarr = np.array(img)

print("array of image's dimension" ,imgarr.shape)


df=pd.DataFrame(imgarr)
print(len(df.index.values),len(df.columns.values))
print("dataframe's dimension",df.shape)
km=KMeans(n_clusters=3,init='k-means++')
km.fit(df)

print("cluster labels",km.labels_)

print("cluster centers ",km.cluster_centers_)

df['cluster']=km.labels_

print("\n\n The count of smaple by clusters ",df['cluster'].value_counts())

sns.countplot(data=df,x='cluster',orient='v',palette='twilight_r')
plt.savefig("C:\\Study_mat\\clusters_count.pdf")
plt.show()
#ax = plt.subplot(111, frame_on=False)
#ax.xaxis.set_visible(False)
#ax.yaxis.set_visible(False)

#table(ax, df1)

#plt.savefig('C:\\Study_mat\\mytable.png')

img = PIL.Image.fromarray(imgarr)
img.save("C:\\Study_mat\\output.png")
