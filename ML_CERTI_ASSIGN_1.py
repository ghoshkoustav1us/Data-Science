import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, fbeta_score, roc_curve, auc, roc_auc_score,make_scorer
import matplotlib.patches as mpatches
import matplotlib.pyplot as pl
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Load the online news dataset
data = pd.read_csv("C:\\Study_mat\\SPARK+ML\\ML_CERT_ASSIGN1\\OnlineNewsPopularity.csv")
display(data.head())
popularity_raw = data[data.keys()[-1]]
popularity_raw.describe()
# Encode the label by threshold 1400

label_encoder = preprocessing.LabelEncoder()
popular_label = pd.Series(label_encoder.fit_transform(popularity_raw>=1400))

features_raw = data.drop(['url',data.keys()[1],data.keys()[-1]], axis=1)
display(features_raw.head())

columns_day = features_raw.columns.values[29:36]
unpop=data[data[' shares']<1400]
pop=data[data[' shares']>=1400]
unpop_day = unpop[columns_day].sum().values
pop_day = pop[columns_day].sum().values


fig = pl.figure(figsize = (13,5))
pl.title("Count of popular/unpopular news over different day of week", fontsize = 16)
pl.bar(np.arange(len(columns_day)), pop_day, width = 0.3, align="center", color = 'purple', label = "popular")
pl.bar(np.arange(len(columns_day)) - 0.3, unpop_day, width = 0.3, align = "center", color = 'black', label = "unpopular")
pl.xticks(np.arange(len(columns_day)), columns_day)
pl.ylabel("Count", fontsize = 12)
pl.xlabel("Days of week", fontsize = 12)

pl.legend(loc = 'upper right')
pl.tight_layout()
pl.savefig("C:\\Study_mat\\SPARK+ML\\ML_CERT_ASSIGN1\\days.pdf")
pl.show()

columns_chan=features_raw.columns.values[11:17]
unpop_chan = unpop[columns_chan].sum().values
pop_chan = pop[columns_chan].sum().values
fig = pl.figure(figsize = (13,5))
pl.title("Count of popular/unpopular news over different article category", fontsize = 16)
pl.bar(np.arange(len(columns_chan)), pop_chan, width = 0.3, align="center", color = 'g',           label = "popular")
pl.bar(np.arange(len(columns_chan)) - 0.3, unpop_chan, width = 0.3, align = "center", color = 'y',           label = "unpopular")
pl.xticks(np.arange(len(columns_chan)), columns_chan)

pl.ylabel("Count", fontsize = 12)
pl.xlabel("Different category", fontsize = 12)

pl.legend(loc = 'upper center')
pl.tight_layout()
pl.savefig("C:\\Study_mat\\SPARK+ML\\ML_CERT_ASSIGN1\\chan.pdf")
pl.show()

scaler = MinMaxScaler()
numerical = [' n_tokens_title', ' n_tokens_content', ' num_hrefs', ' num_self_hrefs', ' num_imgs',' num_videos',\
            ' average_token_length',' num_keywords',' self_reference_min_shares',' self_reference_max_shares',\
             ' self_reference_avg_sharess']
features_raw[numerical] = scaler.fit_transform(data[numerical])
display(features_raw.head(n = 1))

pca = PCA(n_components=2).fit(features_raw)
reduced_features = pca.transform(features_raw)
reduced_features = pd.DataFrame(reduced_features, columns = ['Dimension 1', 'Dimension 2'])
reduced_features_pop = reduced_features[data[' shares']>=1400]
reduced_features_unpop = reduced_features[data[' shares']<1400]

fig, ax = pl.subplots(figsize = (10,10))
# Scatterplot of the reduced data
ax.scatter(x=reduced_features_pop.loc[:, 'Dimension 1'], y=reduced_features_pop.loc[:, 'Dimension 2'],           c='black',alpha=0.7)
ax.scatter(x=reduced_features_unpop.loc[:, 'Dimension 1'], y=reduced_features_unpop.loc[:, 'Dimension 2'],          c='r', alpha=0.5)
ax.set_xlabel("Dimension 1", fontsize=14)
ax.set_ylabel("Dimension 2", fontsize=14)
ax.set_title("PCA on 2 dimensions.", fontsize=16);
pl.show()
pl.savefig("C:\\Study_mat\\SPARK+ML\\ML_CERT_ASSIGN1\\pca2.jpg")


pca = PCA(n_components=3).fit(features_raw)
reduced_features = pca.transform(features_raw)
reduced_features = pd.DataFrame(reduced_features, columns = ['Dimension 1', 'Dimension 2','Dimension 3'])
reduced_features_pop = reduced_features[data[' shares']>=1400]
reduced_features_unpop = reduced_features[data[' shares']<1400]

estimator_DT= DecisionTreeClassifier(random_state=0,criterion='entropy',splitter='best')
print(estimator_DT.get_params().keys())
selector_DT= RFECV(estimator_DT, step=1, cv=5)
selector_DT = selector_DT.fit(features_raw, popular_label)
print("DT Ranking is: ",selector_DT.ranking_)

estimator = AdaBoostClassifier(random_state=0)
print(estimator.get_params().keys())
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(features_raw, popular_label)
print("ADA ranking is : ",selector.ranking_)

estimator_LR = LogisticRegression(random_state=0,solver='liblinear')
print(estimator_LR.get_params().keys())
selector_LR = RFECV(estimator_LR, step=1, cv=5)
selector_LR = selector_LR.fit(features_raw, popular_label)
print("LR rankning is : ",selector_LR.ranking_)

estimator_RF = RandomForestClassifier(random_state=0,n_estimators=10)
print(estimator_RF.get_params().keys())
selector_RF = RFECV(estimator_RF, step=1, cv=5)
selector_RF = selector_RF.fit(features_raw, popular_label)
print("RF ranking is : ",selector_RF.ranking_)

pl.figure()
pl.xlabel("Number of features selected in DT ")
pl.ylabel("Cross validation score (nb of correct classifications)")
pl.plot(range(1, len(selector_DT.grid_scores_) + 1), selector_DT.grid_scores_)
pl.savefig('C:\\Study_mat\\SPARK+ML\\ML_CERT_ASSIGN1\\RFE_DT.pdf')
pl.show()

print ("\nfeature selected in DT :\n",features_raw.columns.values[selector_DT.ranking_==1])
features_DT = features_raw[features_raw.columns.values[selector_DT.ranking_==1]]

pl.figure()
pl.xlabel("Number of features selected in ADA")
pl.ylabel("Cross validation score (nb of correct classifications)")
pl.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
pl.savefig('C:\\Study_mat\\SPARK+ML\\ML_CERT_ASSIGN1\\RFE_ADA.pdf')
pl.show()

print ("\nfeature selected in ADA :\n",features_raw.columns.values[selector.ranking_==1])
features_ADA = features_raw[features_raw.columns.values[selector.ranking_==1]]

pl.figure()
pl.xlabel("Number of features selected in LR")
pl.ylabel("Cross validation score (nb of correct classifications)")
pl.plot(range(1, len(selector_LR.grid_scores_) + 1), selector_LR.grid_scores_)
pl.savefig('C:\\Study_mat\\SPARK+ML\\ML_CERT_ASSIGN1\\RFE_LR.pdf')
pl.show()


print ("\nfeature selected in LR :\n",features_raw.columns.values[selector_LR.ranking_==1])
features_LR = features_raw[features_raw.columns.values[selector_LR.ranking_==1]]

pl.figure()
pl.xlabel("Number of features selected in RF")
pl.ylabel("Cross validation score (nb of correct classifications)")
pl.plot(range(1, len(selector_RF.grid_scores_) + 1), selector_RF.grid_scores_)
pl.savefig('C:\\Study_mat\\SPARK+ML\\ML_CERT_ASSIGN1\\RFE_RF.pdf')
pl.show()

print ("\nfeature selected in RF :\n",features_raw.columns.values[selector_RF.ranking_!=1])
features_RF = features_raw[features_raw.columns.values[selector_RF.ranking_==1]]

X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(features_DT, popular_label, test_size = 0.1, random_state = 0)

X_train_ADA, X_test_ADA, y_train_ADA, y_test_ADA = train_test_split(features_ADA, popular_label, test_size = 0.1, random_state = 0)

X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(features_LR, popular_label, test_size = 0.1, random_state = 0)

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(features_RF, popular_label, test_size = 0.1, random_state = 0)

print("No of samples in Training set: ",X_train_ADA.shape[0])
print("No of samples in test set: ",X_test_ADA.shape[0])

#########################################
#############################

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    results = {}
    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    results['train_time'] = end-start
    # Get predictions on the first 4000 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:4000])
    end = time() # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end-start

    # Compute accuracy on the first 4000 training samples
    results['acc_train'] = accuracy_score(y_train[:4000],predictions_train)

    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
      # Compute F-score on the the first 4000 training samples
    results['f_train'] = fbeta_score(y_train[:4000],predictions_train,beta=1)

    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=1)

    # Compute AUC on the the first 4000 training samples
    results['auc_train'] = roc_auc_score(y_train[:4000],predictions_train)

    # Compute AUC on the test set
    results['auc_test'] = roc_auc_score(y_test,predictions_test)

    # Success
    print ("trained on samples.:",learner.__class__.__name__, sample_size)
    print("\n")
    print (" with accuracy, , F1  and AUC as ",(learner.__class__.__name__,results['acc_test'],results['f_test'], results['auc_test']))
    print("\n")
    print("\n")
    # Return the results
    return results



# Initialize the diff supervised  models
clf_z=DecisionTreeClassifier(random_state=0)
clf_A = AdaBoostClassifier(random_state=0)
clf_B = LogisticRegression(random_state=0,C=1.0)
clf_C = RandomForestClassifier(random_state=0)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(X_train_ADA.shape[0]*0.01)
samples_10 = int(X_train_ADA.shape[0]*0.1)
samples_100 = X_train_ADA.shape[0]

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C,clf_z]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        if clf == clf_A:
            results[clf_name][i] = train_predict(clf, samples, X_train_ADA, y_train_ADA, X_test_ADA, y_test_ADA)
        elif clf == clf_B:
            results[clf_name][i] = train_predict(clf, samples, X_train_LR, y_train_LR, X_test_LR, y_test_LR)
        elif clf==clf_C:
            results[clf_name][i] = train_predict(clf, samples, X_train_RF, y_train_RF, X_test_RF, y_test_RF)
        else:
            results[clf_name][i] = train_predict(clf, samples, X_train_DT, y_train_DT, X_test_DT, y_test_DT)



def gridsearch(clf,parameters,X_train, y_train, X_test, y_test):
    scorer = make_scorer(roc_auc_score)
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
    best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
    print (clf.__class__.__name__)
    print ("Unoptimized model\n------")
    print ("Accuracy score on testing data: ",accuracy_score(y_test, predictions))
    print ("F-score on testing data: ",fbeta_score(y_test, predictions,beta=1))
    print ("AUC on testing data: ",roc_auc_score(y_test, predictions))
    print ("\nOptimized Model\n------")
    print ("Final accuracy score on the testing data: ",accuracy_score(y_test, best_predictions))
    print ("Final F-score on the testing data: ",fbeta_score(y_test, best_predictions, beta=1))
    print ("Final AUC on the testing data: ",roc_auc_score(y_test, best_predictions))

    print (best_clf)


parameters_RF = {"n_estimators": [10,20,50,100,250,500]}
parameters_LR = {"penalty": ['l1','l2'],"C": [0.1,0.5,1.0,2.0,2.5,5]}
parameters_ADA = {"n_estimators": [100,200,300,400],
              "learning_rate": [0.1,0.5,1]}
parameters_DT = {"criterion": ["entropy"],"splitter":["best"]}
############
gridsearch(clf_A,parameters_ADA,X_train_ADA, y_train_ADA, X_test_ADA, y_test_ADA)
###########
gridsearch(clf_C,parameters_RF,X_train_RF, y_train_RF, X_test_RF, y_test_RF)
###########
gridsearch(clf_B,parameters_LR,X_train_LR, y_train_LR, X_test_LR, y_test_LR)
#############
gridsearch(clf_z,parameters_DT,X_train_DT, y_train_DT, X_test_DT, y_test_DT)

###################################################################Decision tree is poorest accuracy so removing it from next course of model fitting
###########Collecting the data for tuned hyper params
samples_1 = int(X_train_ADA.shape[0]*0.01)
samples_10 = int(X_train_ADA.shape[0]*0.1)
samples_100 = X_train_ADA.shape[0]
# Run the classifier with refined hyperparameters
clf_A = AdaBoostClassifier(random_state=0,learning_rate=0.5,n_estimators=300)
clf_B = LogisticRegression(random_state=0, C=0.5,penalty='l1')
clf_C = RandomForestClassifier(random_state=0, n_estimators=500)
clf_z=DecisionTreeClassifier(criterion='entropy',min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0,random_state=0, splitter='best')
# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C,clf_z]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        if clf == clf_A:
            results[clf_name][i] = train_predict(clf, samples, X_train_ADA, y_train_ADA, X_test_ADA, y_test_ADA)
        elif clf == clf_B:
            results[clf_name][i] = train_predict(clf, samples, X_train_LR, y_train_LR, X_test_LR, y_test_LR)
        elif clf==clf_C :
            results[clf_name][i] = train_predict(clf, samples, X_train_RF, y_train_RF, X_test_RF, y_test_RF)
        else:
            results[clf_name][i] = train_predict(clf, samples, X_train_DT, y_train_DT, X_test_DT, y_test_DT)

clf_A = AdaBoostClassifier(random_state=0,learning_rate=0.5,n_estimators=300)
clf_B = LogisticRegression(random_state=0, C=0.5)
clf_C = RandomForestClassifier(random_state=0, n_estimators=50)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C,clf_z]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        if clf == clf_A:
            results[clf_name][i] =             train_predict(clf, samples, X_train_ADA, y_train_ADA, X_test_ADA, y_test_ADA)
        elif clf == clf_B:
            results[clf_name][i] =             train_predict(clf, samples, X_train_LR, y_train_LR, X_test_LR, y_test_LR)
        elif clf==clf_C:
            results[clf_name][i] =             train_predict(clf, samples, X_train_RF, y_train_RF, X_test_RF, y_test_RF)
        else:
            results[clf_name][i] =             train_predict(clf, samples, X_train_DT, y_train_DT, X_test_DT, y_test_DT)


##Hence Random forest is giving  best accuracy ,F BETA  [harmonic mean of Precision (tp/tp+fp) and recall (tp/tp+fn)],and AUROC score
##so using RF will fecth more accuracy and seperability in model
