import nltk
from nltk.corpus import stopwords
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer,PorterStemmer

path='C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\'
with  open(path+'nltk_class_1_assignment_1_input.txt','r+')  as fp:
    x=fp.read();

def lexical_diversity(text):
    return len(set(nltk.word_tokenize(text,'english'))) / len(nltk.word_tokenize(text,'english'))

lex_diversity=lexical_diversity(x)
print('Lexical Diversity of the text input is :',lex_diversity)

def tokenize(x):
    least_common_words=[]
    derived_tokens=nltk.word_tokenize(x,'english')
    word_cnt_pairs=nltk.probability.FreqDist(derived_tokens)
    cnt_word_pairs=[(val,key) for (key,val ) in word_cnt_pairs.items()]
    least_common_words_list=[j[1] if len(j[1])>1 else 'NA' for j in sorted(cnt_word_pairs)]
    for i in least_common_words_list:
        if i!='NA':
            least_common_words.append(i)
        if len(least_common_words)==5:
            break
    return word_cnt_pairs,derived_tokens,least_common_words

word_cnt_pairs,derived_tokens,least_common_words=tokenize(x)

print("Tokens generated:",[str(j) for j in derived_tokens])
print("Frequency of tokens generated:",[x for x in word_cnt_pairs.items()])
print("5 least common words ",least_common_words)

def removestopwords(x):
    non_stopwords=[]
    stopwords_present=[]
    stopwords_set=set(stopwords.words('english'))
    for i in nltk.word_tokenize(x):
        if i not in stopwords_set:
            non_stopwords.append(i)
        else:
            stopwords_present.append(i)
    filtered_text=(' ').join(non_stopwords)
    stopwords_word_freq_list=[i for i in nltk.probability.FreqDist(stopwords_present).items()]
    stopwords_df=pd.DataFrame(stopwords_word_freq_list,columns=['stopword','occurance'])
    return filtered_text,stopwords_word_freq_list,stopwords_df

filtered_text,stopwords_word_freq_list,stopwords_df=removestopwords(x)
print('\n Original text is :',x)
print('\n Filtered text is:',filtered_text)
print("Stopword frequency : ",stopwords_word_freq_list)

sns.barplot(data=stopwords_df.sort_values(by=['occurance'],axis=0,ascending=False),x='stopword',y='occurance')
plt.show()

def lemmatize_and_stemmed(x):
    lemma_stem_word_touple_list=[]
    tokens=nltk.wordpunct_tokenize(x)
    lema_maker=WordNetLemmatizer()
    stem_maker=PorterStemmer()
    for i in tokens:
        lemma_stem_word_touple_list.append((i,lema_maker.lemmatize(i),stem_maker.stem(i)))
    stem_pattern_df=pd.DataFrame(lemma_stem_word_touple_list,columns=['Original Word','Lemmatized Form','Stemmed Form'])
    stem_pattern_df.to_csv(path+'nltk_class_1_assignment_1_stem_pattern.csv')
    return stem_pattern_df

print(lemmatize_and_stemmed(x))


