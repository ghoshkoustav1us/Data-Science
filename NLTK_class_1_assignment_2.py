from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
path='C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\'
# problem 1
import nltk
print('\n')
# problem 2
import os
files=os.listdir(nltk.data.find('corpora'))
print(files)
print('\n')

#problem 3 and 4
print(nltk.corpus.twitter_samples.fileids())
print('\n')
#problem 5
#already done as I installed 'all' from nltk downloader

#problem 6
print(nltk.corpus.gutenberg.fileids())
print('\n')
#problem 7
macbeth_text=nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')
print(macbeth_text)
print('\n')

#problem 8
with open(path+'Macbeth.txt','w') as fp:
    fp.write(' '.join(macbeth_text))

#problem 9
article_dict=defaultdict(int)
macbeth_article_removed_text=list(macbeth_text).copy()
for j in macbeth_text:
    if str(j).lower() in ('a','an','the'):
        macbeth_article_removed_text.remove(j)
        article_dict[str(j).lower()]+=1
print(article_dict)
with open(path+'Macbeth-ArticlesRemoved.txt','w') as fp:
    fp.write(' '.join(macbeth_article_removed_text))

#problem 10
mcbeth_raw_text=nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')
print(type(mcbeth_raw_text),'\n',mcbeth_raw_text)

punc_removed_text=re.sub(r'[\]\[\.\,\’:;#\(\)]','',mcbeth_raw_text)

with open(path+'Macbeth-punctuationsRemoved.txt','w') as fp:
    fp.write(punc_removed_text)

# problem 11
#already done as I installed 'all' from nltk downloader

#problem 12
corp_names_=list(nltk.corpus.names.words('female.txt'))+list(nltk.corpus.names.words('male.txt'))
corp_names_total_words=len(corp_names_)
print(corp_names_)
print(corp_names_total_words,'\n')

# problem 13
list_each_alphabet=defaultdict(list)
freqdict_each_alphabet=defaultdict(int)
for i in corp_names_:
    freqdict_each_alphabet[i[0]]+=1
    list_each_alphabet[i[0]].append(i)

print("Alphabet wise count :\n")
for k in  freqdict_each_alphabet.items():
    print(k)
    print('\n')

albhabet_wise_name_count_df=pd.DataFrame(list(freqdict_each_alphabet.items()),columns=['Alphabet','count'])

#problem 14,15,16
sns.barplot(x='Alphabet',y='count',data=albhabet_wise_name_count_df)
plt.xlabel('Starting Alphabet of Names in Names corpora ')
plt.ylabel('Correspoing names counts which starts with the Alphabet')
plt.title("Alphabet vs Count for Names corpora")
plt.savefig(path+"FrequencyOfNamesOfEachAlphabet.png")
plt.show()
