import nltk
import re
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.parse.generate import generate
from nltk.stem import WordNetLemmatizer,PorterStemmer

path='C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\698_m3_datasets_v1.0\\'
with  open(path+'FIFAWorldCup2018.txt','r')  as fp:
    x=fp.read();


def GetNMostFrequentPos(text,N,pos_val):
    pos_val=pos_val[:2]
    tag_dict=defaultdict(list)
    tt=TweetTokenizer()
    words=tt.tokenize(text)
    tagged_words=nltk.pos_tag(words)
    cfd=nltk.probability.ConditionalFreqDist((t,w) for (w,t) in tagged_words  if t.startswith(pos_val))
    for t in cfd.conditions():
        tag_dict[t]=cfd[t].most_common(N)
    return tag_dict

noun_tag_dict=GetNMostFrequentPos(x,6,'NN')
verb_tag_dict=GetNMostFrequentPos(x,6,'VB')
prepostition_tag_dict=GetNMostFrequentPos(x,6,'IN')
determiner_which_tag_dict=GetNMostFrequentPos(x,6,'WDT')
determiner_which_tag_dict=GetNMostFrequentPos(x,6,'DT')

for i in noun_tag_dict.items():
    print(i)

print('\n\n')

for i in verb_tag_dict.items():
    print(i)

print('\n\n')

for i in prepostition_tag_dict.items():
    print(i)

print('\n\n')

for i in determiner_which_tag_dict.items():
    print(i)

print('\n\n')

def PrintSyntaxTree(text,grammer_pattern):
    framed_result=[]
    first_line= nltk.tokenize.sent_tokenize(text)[0]
    first_line_tokens=nltk.tokenize.word_tokenize(first_line)
    sent_tagged_tokens=nltk.pos_tag(first_line_tokens)
    framed_parser=nltk.RegexpParser(grammer_pattern)
    framed_result.append(framed_parser.parse(sent_tagged_tokens))
    print(framed_result)

phrase_name="DET_AND_NOUN"
grammer_rule_DET_AND_NOUN=phrase_name+":{<DT>+(<NNP>|<NN>|<NNS>|<NNPS>)}"
print(grammer_rule_DET_AND_NOUN)
framed_result=PrintSyntaxTree(x,grammer_rule_DET_AND_NOUN)


def TextAfterRemovingPatterns(text,pattern):
    Twitt_tokenizer=nltk.tokenize.TweetTokenizer()
    words=Twitt_tokenizer.tokenize(text)
    pattern=pattern
    print(words)
    finetuned_list=[re.sub(pattern,'',word)+' ' for word in words ]
    finetuned_text=''.join(finetuned_list)
    return finetuned_text

punctuation_eliminated_text=TextAfterRemovingPatterns(x,r'[\.\,;"_]')
digits_eliminated_text=TextAfterRemovingPatterns(x,r'[0-9]')
capitalized_eliminated_text=TextAfterRemovingPatterns(x,r'[A-Z]{2,}')

dummy_str='Koustav has an email id of koustav.ghosh@thermo.com and he is working there for 1 year' #As we dont have any email id in Fifa txt so for testing taking this dummy string
email_eliminated_text=TextAfterRemovingPatterns(dummy_str,r'([a-zA-Z0-9_\-\.])([a-zA-Z0-9_\-\.])+@([a-zA-Z0-9_\-\.])+\.[a-zA-Z]{2,5}')

print("Original text is:\n",x,'\n')
print("Punctuation eliminated text is:\n",punctuation_eliminated_text,'\n')
print("Digit eliminated text is:\n",digits_eliminated_text,'\n')
print("Capitalized eliminated text is:\n",capitalized_eliminated_text,'\n')
print("Email eliminated text is:\n",email_eliminated_text,'\n')


def chunking(text,grammer_pattern,phrase_name):
    words=nltk.tokenize.word_tokenize(text)
    framed_result=[]
    sent_tagged_tokens=nltk.pos_tag(words)
    framed_parser=nltk.RegexpParser(grammer_pattern)
    framed_result.append(framed_parser.parse(sent_tagged_tokens))
    for indx,val  in enumerate(framed_result):
        for value in val:
            if re.search(phrase_name,str(value)):
                print(value)
    return framed_result

phrase_name="PROP_NOUN_AND_VERB"
grammer_rule_PROP_NOUN_AND_VERB=phrase_name+":{<NNP>+(<VBZ>|<VBP>|<VBN>|<VBD>|<VB>|<VBG>)}"
print(grammer_rule_PROP_NOUN_AND_VERB)
framed_result=chunking(x,grammer_rule_PROP_NOUN_AND_VERB,phrase_name)
print('\n\n')


phrase_name="VERB_AND_ADJ"
grammer_rule_VERB_AND_ADJ=phrase_name+":{(<VBZ>|<VBP>|<VBN>|<VBD>|<VB>|<VBG>)+<JJ>}"
print(grammer_rule_VERB_AND_ADJ)
framed_result=chunking(x,grammer_rule_VERB_AND_ADJ,phrase_name)
print('\n\n')


phrase_name="DET_AND_NOUN"
grammer_rule_DET_AND_NOUN=phrase_name+":{<DT>+(<NNP>|<NN>|<NNS>|<NNPS>)}"
print(grammer_rule_DET_AND_NOUN)
framed_result=chunking(x,grammer_rule_DET_AND_NOUN,phrase_name)
print('\n\n')


phrase_name="VERB_AND_ADVERB"
grammer_rule_VERB_AND_ADVERB=phrase_name+":{(<VBZ>|<VBP>|<VBN>|<VBD>|<VB>|<VBG>)+<RB>}"
print(grammer_rule_VERB_AND_ADVERB)
framed_result=chunking(x,grammer_rule_VERB_AND_ADVERB,phrase_name)
print('\n\n')


phrase_name="DET_AND_NOUN_AND_ADJ"
grammer_rule_DET_AND_NOUN_AND_ADJ=phrase_name+":{<DT>*(<NNP>|<NN>|<NNS>|<NNPS>)+<JJ>}"
print(grammer_rule_DET_AND_NOUN_AND_ADJ)
framed_result=chunking(x,grammer_rule_DET_AND_NOUN_AND_ADJ,phrase_name)



CFG_rule=nltk.CFG.fromstring("""
    S -> NP VP
    VP -> V N
    VP ->V N PP
    PP ->P NP
    NP ->Det N
    Det -> "The"|"a"
    N ->"World"|"Cup"
    V ->"was"|"had"
    P ->"of"|"in"
    """)

print(CFG_rule.productions(),'\n\n')

for s in nltk.parse.generate.generate(grammar=CFG_rule):
    print(' '.join(s))

def CFG_parse_function(text):
    ofp=open(path+'CFG.txt','w')
    sent=nltk.tokenize.sent_tokenize(text)
    proper_noun_tag_list=['NNP','NNPS']
    noun_tag_list=['NN','NNS']
    verb_tag_list=['VBD','VB','VBN','VBP']
    det_tag_list=['DT']
    prp_tag_list=['IN']
    detholder=propernounholder=nounholder=verbholder=prpholder=''
    for j in sent:
        sent_tagged_words=nltk.pos_tag(nltk.tokenize.word_tokenize(j))
        for word,tag in sent_tagged_words:
            if tag in proper_noun_tag_list:
                propernounholder="\'"+word+"\'"
            elif tag in noun_tag_list:
                nounholder="\'"+word+"\'"
            elif tag in verb_tag_list:
                verbholder="\'"+word+"\'"
            elif tag in prp_tag_list:
                prpholder="\'"+word+"\'"
            elif tag in det_tag_list:
                detholder="\'"+word+"\'"
            else:
                pass
            CFG_rule_1=nltk.CFG.fromstring("""
        S -> NP VP
        VP -> V NN PP
        PP -> P NN
        NP -> Det NNP
        Det -> {}
        NNP ->{}
        NN -> {}
        V -> {}
        P -> {}
        """.format(detholder,propernounholder,nounholder,verbholder,prpholder))

        for s in nltk.parse.generate.generate(grammar=CFG_rule_1):
                op_str=' '.join(s)
                ofp.write(op_str+'\n')
    ofp.close()
    return True


CFG_parse_function(x)

