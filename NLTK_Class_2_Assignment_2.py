import nltk
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import docx
from collections import defaultdict
path='C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\'
ip_path='brexit.txt'

#document is not opening in m MS WORD so followed a slightly different approach
# ip_doc=docx.Document(path+ip_path)
# fp=open('brexit.txt','w+')
# for i in ip_doc.paragraphs:
#     para_str=str(i.text)
#     print('\n')
#     print(para_str)
#     fp.write(para_str)
#     sentences=nltk.tokenize.sent_tokenize(para_str,language='english')
#     word_punc=nltk.tokenize.wordpunct_tokenize(para_str)
#     print(sentences)
#     print(word_punc)
#     print(bl_line)
#     print('\n')

with open(ip_path,'r+') as fp:
    textstr=fp.read()

def GetNGrams(textstr,n):
    sentences=nltk.tokenize.word_tokenize(textstr,language='english')
    ngram_op=[i for i in nltk.ngrams(sentences,n=n)]
    return ngram_op

bigram_op=GetNGrams(textstr,2)
print(bigram_op)
trigram_op=GetNGrams(textstr,3)
quad_gram_op=GetNGrams(textstr,4)


def tag_Count(textstr,pos_tag,pos):
    tag_cnt_dict=defaultdict(int)
    tags=nltk.pos_tag(nltk.tokenize.word_tokenize(textstr))
    for word,tag in tags:
        if tag in pos_tag:
            tag_cnt_dict[tag]+=1
    total_count=(np.sum([i[1] for i in list(tag_cnt_dict.items())]),pos)
    return tag_cnt_dict,total_count

tag_cnt_noun_dict,total_noun_count=tag_Count(textstr,['NNP','NNS','NNPS','NN'],'Noun')
print('The different noun tags present in string has following count distribution:\n',[i for i in list(tag_cnt_noun_dict.items())],"\nThe total noun count :" ,total_noun_count,'\n')
tag_cnt_pronoun_dict,total_pronoun_count=tag_Count(textstr,['PRP','PRP$'],'Pronoun')
print('The different pronoun tags present in string has following count distribution:\n',[i for i in list(tag_cnt_pronoun_dict.items())],"\nThe total pronoun count :" ,total_pronoun_count,'\n')
tag_cnt_verb_dict,total_verb_count=tag_Count(textstr,['VB','VBG','VBN','VBD','VBP','VBZ'],'Verb')
print('The different verb tags present in string has following count distribution:\n',[i for i in list(tag_cnt_verb_dict.items())],"\nThe total noun count :" ,total_verb_count,'\n')
tag_cnt_adj_dict,total_adj_count=tag_Count(textstr,['JJ','JJR','JJS'],'Adjective')
print('The different adjective tags present in string has following count distribution:\n',[i for i in list(tag_cnt_adj_dict.items())],"\nThe total adjective count :" ,total_adj_count,'\n')
tag_cnt_adv_dict,total_adv_count=tag_Count(textstr,['RB','RBR'],'Adverb')
print('The different adverb tags present in string has following count distribution:\n',[i for i in list(tag_cnt_adv_dict.items())],"\nThe total adverb count :" ,total_adv_count,'\n')

pie_list=[total_noun_count,total_pronoun_count,total_verb_count,total_adj_count,total_adv_count]

pie_df=pd.DataFrame(pie_list,columns=['count','pos'],index=[i[1] for i in pie_list])
print('\n The df input for pie chart is :\n',pie_df)

pie_df.plot.pie(y='count')
plt.show()

print('\n\n')

def most_freq_noun(textstr,pos_tag,pos):
    tag_cnt_dict=defaultdict(int)
    tags=nltk.pos_tag(nltk.tokenize.word_tokenize(textstr))
    for (word,tag) in (tags) :
        if tag in pos_tag:
            tag_cnt_dict[word]+=1
    noun_tag_cnt_df=pd.DataFrame(list(tag_cnt_dict.items()),columns=['words','cnt'])
    noun_tag_cnt_df=noun_tag_cnt_df.sort_values(by=['cnt'],axis=0,ascending=False)
    return noun_tag_cnt_df.iloc[0,:]

most_noun_tag_cnt_df=most_freq_noun(textstr,['NNP','NNS','NNPS','NN'],'Noun')
print("Most common noun in the brexit text:",most_noun_tag_cnt_df)

def GetMostFrequentbiGrams(textstr,n=2):
    bigram_cnt_dict=defaultdict(int)
    sentences=nltk.tokenize.word_tokenize(textstr,language='english')
    ngram_op=[i for i in nltk.ngrams(sentences,n=n)]
    for i in ngram_op:
        bigram_cnt_dict[i]+=1
    bigram_cnt_df=pd.DataFrame(list(bigram_cnt_dict.items()),columns=['bigrams','cnt'])
    bigram_cnt_df=bigram_cnt_df.sort_values(by=['cnt'],axis=0,ascending=False)
    return bigram_cnt_df.iloc[0,:]

most_common_bigram_cnt_df=GetMostFrequentbiGrams(textstr)
print("Most common bigrams in the brexit text:",most_common_bigram_cnt_df)

def chunk_count(chunk_tag):
    chunk_tag_cnt_dict=defaultdict(int)
    chunked_entity_count_wise_dict=defaultdict(int)
    chunk_list=[]
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.tokenize.word_tokenize(textstr))):
        if hasattr(chunk,'label'):
            chunk_list.append((chunk.label(), ' '.join(c[0] for c in chunk)))
    for i in chunk_list:
        print(i)
        if i[0]in chunk_tag:
            ner_tag=i[0]
            chunk_tag_cnt_dict[ner_tag]+=1
        if i[0]in chunk_tag:
            entity=i[1]
            chunked_entity_count_wise_dict[(entity,ner_tag)]+=1
    ner_chunk_count_wise_list=[i for i in chunk_tag_cnt_dict.items()]
    ner_chunk_count_wise_df=pd.DataFrame(ner_chunk_count_wise_list,columns=['ner_chunk','cnt'],index=[j[0] for j in ner_chunk_count_wise_list])

    chunked_entity_count_wise_list=[i for i in chunked_entity_count_wise_dict.items()]
    chunked_entity_count_wise_df=pd.DataFrame(chunked_entity_count_wise_list,columns=['Entity/NER','cnt'])
    chunked_entity_count_wise_df=chunked_entity_count_wise_df['Entity/NER'].apply(pd.Series).merge(chunked_entity_count_wise_df,left_index=True,right_index=True).drop('Entity/NER',axis=1)
    chunked_entity_count_wise_df=chunked_entity_count_wise_df.rename(columns={1:'NER',0:'ENTITY'})

    return chunk_tag_cnt_dict,chunked_entity_count_wise_dict,ner_chunk_count_wise_df,chunked_entity_count_wise_df

chunk_tag_cnt_dict,chunked_entity_count_wise_dict,ner_chunk_count_wise_df,chunked_entity_count_wise_df=chunk_count(['GPE','PERSON','ORGANIZATION'])
print(chunked_entity_count_wise_dict)
print('\n NER chunk count wise :\n')
print(ner_chunk_count_wise_df)

ner_chunk_count_wise_df.plot.pie(y='cnt')
plt.show()

print('\n')
GPE_df=chunked_entity_count_wise_df[chunked_entity_count_wise_df['NER']=='GPE']
PERSON_df=chunked_entity_count_wise_df[chunked_entity_count_wise_df['NER']=='PERSON']
ORG_df=chunked_entity_count_wise_df[chunked_entity_count_wise_df['NER']=='ORGANIZATION']

Entity_with_Max_GPE_ner_df=GPE_df.sort_values(by=['cnt'],axis=0,ascending=False).iloc[0,:]
Entity_with_Max_PER_ner_df=PERSON_df.sort_values(by=['cnt'],axis=0,ascending=False).iloc[0,:]
Entity_with_Max_ORG_ner_df=ORG_df.sort_values(by=['cnt'],axis=0,ascending=False).iloc[0,:]

print("Entity with most number of GPE NER:",Entity_with_Max_GPE_ner_df)
print('\n')
print("Entity with most number of PERSON NER:",Entity_with_Max_PER_ner_df)
print('\n')
print("Entity with most number of ORG NER:",Entity_with_Max_ORG_ner_df)
