import nltk
import re
import pandas as pd
import  matplotlib.pyplot as plt
import os
import math

path='C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\nltk_datasets\\698_m3_datasets_v1.0\\'

pdf=pd.read_csv(path+'Tweets.csv',header='infer',sep=',',encoding='utf-8')
pdf_text_sentiment_container=pdf.iloc[:,1:11:9]

print(pdf_text_sentiment_container)

tweet_handle_list=[]

grammer_pattern=r'@[a-zA-Z0-9]+'
reg_parser=nltk.RegexpTokenizer(grammer_pattern)

for indx,row in pdf_text_sentiment_container.iterrows():
    tweet_handle_val=reg_parser.tokenize(row[1])
    tweet_handle_list.append((' '.join(tweet_handle_val),row[0]))


pdf_refrences=pd.DataFrame(tweet_handle_list,columns=['refereces','sentiment'])
pdf_refrences.to_csv(path_or_buf=path+'References.txt',sep=',',header='infer')


def extract_phrases_in_file(df,col_pos,text_pos,sentiment,phrase,phrase_name):
    Tw_tokenizer=nltk.tokenize.TweetTokenizer()
    df_extraction=df.iloc[:,col_pos]
    df_extraction=df_extraction[df_extraction.iloc[:,0]==sentiment]
    framed_result=[]
    fp=open(path+'Review_'+phrase_name+'_'+sentiment+'.txt','w+',encoding='utf-8')
    for indx,row in df_extraction.iterrows():
        sent_tagged_tokens=nltk.pos_tag(Tw_tokenizer.tokenize(row[text_pos]))
        parser_container=nltk.RegexpParser(phrase)
        framed_result.append(parser_container.parse(sent_tagged_tokens))
    for indx,val  in enumerate(framed_result):
        for value in val:
            if re.search(phrase_name,str(value)):
                fp.write(str(value)+'\n')
    return framed_result

phrase_name="NOUN_PHRASES"
grammer_rule_NP=phrase_name+":{<DT>+(<NNP>|<NN>)+<JJ>}"
framed_result=extract_phrases_in_file(pdf,[1,10],1,'positive',grammer_rule_NP,phrase_name)

phrase_name="NOUN_PHRASES"
grammer_rule_NP=phrase_name+":{<DT>+(<NNP>|<NN>)+<JJ>}"
framed_result=extract_phrases_in_file(pdf,[1,10],1,'negative',grammer_rule_NP,phrase_name)

phrase_name="NOUN_PHRASES"
grammer_rule_NP=phrase_name+":{<DT>+(<NNP>|<NN>)+<JJ>}"
framed_result=extract_phrases_in_file(pdf,[1,10],1,'neutral',grammer_rule_NP,phrase_name)

phrase_name="VERB_PHRASES"
grammer_rule_VERB_PHRASES=phrase_name+":{(<VBZ>|<VBP>|<VBN>|<VBD>|<VB>|<VBG>)+<NN>}"
framed_result=extract_phrases_in_file(pdf,[1,10],1,'positive',grammer_rule_VERB_PHRASES,phrase_name)

phrase_name="VERB_PHRASES"
grammer_rule_VERB_PHRASES=phrase_name+":{(<VBZ>|<VBP>|<VBN>|<VBD>|<VB>|<VBG>)+<NN>}"
framed_result=extract_phrases_in_file(pdf,[1,10],1,'negative',grammer_rule_VERB_PHRASES,phrase_name)

phrase_name="VERB_PHRASES"
grammer_rule_VERB_PHRASES=phrase_name+":{(<VBZ>|<VBP>|<VBN>|<VBD>|<VB>|<VBG>)+<NN>}"
framed_result=extract_phrases_in_file(pdf,[1,10],1,'neutral',grammer_rule_VERB_PHRASES,phrase_name)


pdf_text_sentiment_count=pdf_text_sentiment_container.iloc[:,0].value_counts()
pdf_text_sentiment_count.plot.pie()
plt.show()


files = os.listdir(path)

file_cnt_list=[]
for text_file in files:
    if re.search(r'Review',str(text_file)):
        file_path = path+text_file
        num_lines=sum(1 for line in open(file_path,encoding='utf-8',mode='r+'))
        file_cnt_list.append((num_lines,text_file.replace('Review_','').replace('PHRASES_','').replace('.txt','')))

file_cnt_pdf=pd.DataFrame(file_cnt_list,columns=['cnt','file_name'])
file_cnt_pdf=file_cnt_pdf.set_index('file_name')
print(file_cnt_pdf)
file_cnt_pdf.plot.pie(y='cnt')
plt.show()
