# Databricks notebook source
# MAGIC %md ###Loading all necessary packages

# COMMAND ----------

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn import decomposition, ensemble
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas,  numpy,  string
import pandas as pd
import os

# COMMAND ----------

# MAGIC %md ###Consolidating the data 

# COMMAND ----------

data_folder = "../dataset/bundle_archive/BBC News Summary/BBC News Summary"
folders = ["business","entertainment","politics","sport","tech"]

os.chdir(data_folder)

x = []
y = []

for i in folders:
    files = os.listdir(i)
    for text_file in files:
        file_path = i + "/" +text_file
        print "reading file:", file_path
        with open(file_path) as f:
            data = f.readlines()
        data = ' '.join(data)
        x.append(data)
        y.append(i)
        
data = {'news': x, 'type': y}       
df = pd.DataFrame(data)
print 'appending csv flie ...'
df.to_csv('../input/bbc-consolidated_text.csv', encoding='utf-8',index=False)

# COMMAND ----------

# MAGIC %md ###Reading  the data

# COMMAND ----------

trainDF = pd.read_csv('../input/bbc-consolidated_text.csv',encoding='utf-8') 

# COMMAND ----------

trainDF.shape

# COMMAND ----------

trainDF['category'].unique()

# COMMAND ----------

trainDF['category'].value_counts()

# COMMAND ----------

sns.set(rc={'figure.figsize':(10,10)})
sns.countplot(trainDF['category'])

# COMMAND ----------

# MAGIC %md ###split the dataset into train and test

# COMMAND ----------

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['category'])

train_labels = train_y
valid_labels = valid_y
# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# COMMAND ----------

# MAGIC %md ###Apply count vectorizer

# COMMAND ----------

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# COMMAND ----------

# MAGIC %md Applying PCA of component=2 and Plotting trained features for count vectorizer

# COMMAND ----------

pca = PCA(n_components=2).fit(xtrain_count.toarray())
data2D = pca.transform(xtrain_count.toarray())
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(data2D[:,0], data2D[:,1],
hue=train_labels.tolist(),size=train_labels.tolist(),palette="husl")

# COMMAND ----------

# MAGIC %md Applying PCA of component=2 and Plotting tested features for count vectorizer

# COMMAND ----------

pca = PCA(n_components=2).fit(xvalid_count.toarray())
data2D = pca.transform(xvalid_count.toarray())
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(data2D[:,0], data2D[:,1],
hue=valid_labels.tolist(),size=valid_labels.tolist(),palette="husl")

# COMMAND ----------

# MAGIC %md ###Apply TF_IDF

# COMMAND ----------

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# COMMAND ----------

# MAGIC %md Applying PCA of component=2 and Plotting trained features for TF_IDF

# COMMAND ----------

pca = PCA(n_components=2).fit(xtrain_tfidf.toarray())
data2D = pca.transform(xtrain_tfidf.toarray())
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(data2D[:,0], data2D[:,1],
hue=train_labels.tolist(),size=train_labels.tolist(),palette="husl")

# COMMAND ----------

# MAGIC %md Applying PCA of component=2 and Plotting tested features for TF_IDF

# COMMAND ----------

pca = PCA(n_components=2).fit(xvalid_tfidf.toarray())
data2D = pca.transform(xvalid_tfidf.toarray())
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(data2D[:,0], data2D[:,1],
hue=valid_labels.tolist(),size=valid_labels.tolist(),palette="husl")

# COMMAND ----------

# MAGIC %md ###N-Gram TF IDF

# COMMAND ----------

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


# COMMAND ----------

# MAGIC %md Applying PCA of component=2 and Plotting trained features for n -gram TF_IDF

# COMMAND ----------

pca = PCA(n_components=2).fit(xtrain_tfidf_ngram.toarray())
data2D = pca.transform(xtrain_tfidf_ngram.toarray())
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(data2D[:,0], data2D[:,1],
hue=train_labels.tolist(),size=train_labels.tolist(),palette="husl")

# COMMAND ----------

# MAGIC %md Applying PCA of component=2 and Plotting tested  features for n -gram TF_IDF

# COMMAND ----------

pca = PCA(n_components=2).fit(xvalid_tfidf_ngram.toarray())
data2D = pca.transform(xvalid_tfidf_ngram.toarray())
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(data2D[:,0], data2D[:,1],
hue=valid_labels.tolist(),size=valid_labels.tolist(),palette="husl")

# COMMAND ----------

# MAGIC %md ###Function for Model run

# COMMAND ----------

def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    return metrics.accuracy_score(predictions, valid_y),metrics.silhouette_score(predictions, valid_y, metric='euclidean')
  #for euclidian distance we use 'euclidian' which tells better about inter and intra cluster distance

# COMMAND ----------

# MAGIC %md Calling Multinomial Naive Bayes on all algos above

# COMMAND ----------

# Naive Bayes on Count Vectors
accuracy,silhouette = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB accuracy for Count Vectors: ", accuracy)
print("NB silhoutte for  Count Vectors: ", silhouette)

# Naive Bayes on Word Level TF IDF Vectors
accuracy,silhouette = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB accuracy for  WordLevel TF-IDF: ", accuracy)
print("NB silhoutte for  WordLevel TF-IDF: ", silhouette)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy,silhouette = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB silhoutte for  Ngram TFIDF: ", accuracy)
print("NB silhoutte for  Ngram TFIDF: ", silhouette)

# COMMAND ----------

# MAGIC %md Calling Random forest  on all algos above

# COMMAND ----------

# RF on Count Vectors
accuracy,silhouette = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_count, train_y, xvalid_count)
print("RF accuracy for Count Vectors: ", accuracy)
print("RF silhoutte for  Count Vectors: ", silhouette)

# RF on Word Level TF IDF Vectors
accuracy,silhouette = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_tfidf, train_y, xvalid_tfidf)
print("RF accuracy for  WordLevel TF-IDF: ", accuracy)
print("RF silhoutte for  WordLevel TF-IDF: ", silhouette)

# RF on Ngram Level TF IDF Vectors
accuracy,silhouette = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("RF silhoutte for  Ngram TFIDF: ", accuracy)
print("RF silhoutte for  Ngram TFIDF: ", silhouette)

# COMMAND ----------

# MAGIC %md Calling Kmeans on all algos above

# COMMAND ----------

# Kmeans on Count Vectors
accuracy,silhouette = train_model(KMeans(n_clusters=5, random_state=1), xtrain_count, train_y, xvalid_count)
print("KM accuracy for Count Vectors: ", accuracy)
print("KM silhoutte for  Count Vectors: ", silhouette)

# Kmeans on Word Level TF IDF Vectors
accuracy,silhouette = train_model(KMeans(n_clusters=5, random_state=1), xtrain_tfidf, train_y, xvalid_tfidf)
print("KM accuracy for  WordLevel TF-IDF: ", accuracy)
print("KM silhoutte for  WordLevel TF-IDF: ", silhouette)

# Kmeans on Ngram Level TF IDF Vectors
accuracy,silhouette = train_model(KMeans(n_clusters=5, random_state=1), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("KM silhoutte for  Ngram TFIDF: ", accuracy)
print("KM silhoutte for  Ngram TFIDF: ", silhouette)

# COMMAND ----------

#1)What does Silhouette Coefficient tell us? - It tells us how coherent a sample in same cluster and how disimilar it is in different cluster

#2.Which algorithm you chose and why? - I chose kmeans( lightweight efficient clustering) , RF ( best service when we have multiple level in classification ) and MultinomialNB ( efficient in text data)
#4.Which vectorization techniqueis the best and why?- TF IDF is best as it removes common words in every document seeing they are trivial and not relevant for analysis 
# TF IDF also comes with lot of parameters to be tuned

