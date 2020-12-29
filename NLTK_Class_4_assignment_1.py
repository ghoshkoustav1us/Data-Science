import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
from collections import defaultdict


def makecorpus(str_list):
    corpus_set=set()
    for i in str_list:
      tw_tokenizer=nltk.tokenize.TweetTokenizer ()
      corpus_set=corpus_set.union(set(tw_tokenizer.tokenize(i)))
    return corpus_set


S1="India won the match of match"
S2="England won the cricket match"
S3= " Australia won the final kapil match"
corpus_set=makecorpus([S1,S2,S3])
print(corpus_set)

def presence_absence(str_list,corpus_set):
    tw_tokenizer=nltk.tokenize.TweetTokenizer()
    data_list=[]
    for str_piece in str_list:
        presence_absence_dict=defaultdict(int)
        tokens=tw_tokenizer.tokenize(str_piece)
        for token in corpus_set :
            if token in tokens:
                presence_absence_dict[token]=1
            else:
                presence_absence_dict[token]=0
        data_list.append(presence_absence_dict)
    return data_list

data_list=presence_absence([S1,S2,S3],corpus_set)
print(data_list)
pdf=pd.DataFrame(data_list)
print(pdf)
print('\n\n')


def CountVectorization(str_list):
    cvz=CountVectorizer()
    cvz_container=cvz.fit_transform(str_list)
    colnames=cvz.get_feature_names()
    cvz_pdf=pd.DataFrame(cvz_container.toarray(),columns=colnames)
    return cvz_pdf

cvz_pdf=CountVectorization([S1,S2,S3])
print("Count vecotr\n",cvz_pdf)
print('\n\n')

def TF_IDFVectorization(str_list):
    TF_IDF_z=TfidfVectorizer(lowercase=True,stop_words='english')
    TF_IDF_z_container=TF_IDF_z.fit_transform(str_list)
    colnames=TF_IDF_z.get_feature_names()
    TF_IDF_z_pdf=pd.DataFrame(TF_IDF_z_container.toarray(),columns=colnames)
    return TF_IDF_z_pdf

TF_IDF_z_pdf=TF_IDFVectorization([S1,S2,S3])
print("TF IDF:\n",TF_IDF_z_pdf)


