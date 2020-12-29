import nltk
import re
import pandas as pd
from pprint import pprint
import preprocess
import vectorization
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.parse.generate import generate
from nltk.stem import WordNetLemmatizer,PorterStemmer

path='C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\nltk_datasets\\698_m4_datasets_v1.0\\'

pdf=pd.read_csv(path+'Wine.csv',header='infer',sep=',',encoding='utf-8')
pdf_description=pdf[['country','description']]
#
refined_list_of_desc=[]
refine_val=''

for indx,row in pdf_description.iterrows():
    print(row[1],'\n')
    refine_val=preprocess.refine(row[1])
    refined_list_of_desc.append(refine_val)

pdf_description_refined=pd.DataFrame(refined_list_of_desc,columns=['refined_desc'])
print(pdf_description_refined.head(10))

TF_IDF_z_pdf=vectorization.TF_IDFVectorization(refined_list_of_desc)

TF_IDF_z_pdf.to_csv(path_or_buf=path+'TF_IDF_REFINED_WINE_DESCRIPTION.csv')
