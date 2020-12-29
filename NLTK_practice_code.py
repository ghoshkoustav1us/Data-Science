import nltk
from sklearn.metrics import jaccard_score
import nltk.corpus
import os
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd_common=tag_fd.most_common(100)
for word,count in tag_fd_common:
    print(word,count)


print(os.listdir(nltk.data.find('corpora')))

print(nltk.corpus.gutenberg.fileids())
words_carroll_alice=nltk.corpus.gutenberg.words('carroll-alice.txt')
sentence_carroll_alice=nltk.corpus.gutenberg.raw('carroll-alice.txt')
print('\n',words_carroll_alice)

print('\n',sentence_carroll_alice)


s1=['koustav']
s2=['kuotsva']

print(jaccard_score(s1,s2),pos_label=0)
