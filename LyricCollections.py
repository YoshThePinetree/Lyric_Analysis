import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import word2vec
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import corpora
from gensim import models


path = 'C:\PythonIO\Input\lyrics\lyric_counterparts.txt'
titlepath = 'C:\PythonIO\Input\lyrics\Titles.txt'

# reading lyrics
f = open(path,encoding="utf-8")
lyrics = []
for line in f:
    print(line)
    if line != '\n':
        lyrics.append(line)
f.close()
n = len(lyrics)

# reading album and song titles
f = open(titlepath)
year = []
album = []
song = []
for line in f:
    itemList = line[:-1].split('\t')
    year.append(itemList[0])
    album.append(itemList[1])
    song.append(itemList[2])
f.close()

stopWords = stopwords.words('english')
stopWords.extend(['’', '...', '(', ')', '``', ',', ':', '"', 'ca', 'wo', '\'ll', '\'d', '\'ve', '\'s', 'n\'t', '\'re', '?', '!', 'us'])
lyrics_split = []
lyrics_split_series = []
for lyric in lyrics:
    text = nltk.word_tokenize(lyric.lower())
    for i in range(len(stopWords)):
        text = [s for s in text if s != stopWords[i]]
    for i in range(len(text)):
        lyrics_split_series.append(text[i])
    lyrics_split.append(text)

### Vectorization by word2vec
h = 200
model_word = word2vec.Word2Vec(lyrics_split, sg=0, size=h, alpha=0.025, min_count=5, window=5, iter=300, seed=0)     # word2vec training
print(model_word.wv['nothing'])
print('Similar words of: ', 'nothing')
print(model_word.wv.most_similar(positive=['nothing']))

### Word Analysis
v = len(model_word.wv.vocab)
vector_word = model_word.wv.syn0    # all word vectors (v * h)
vector_word_reduced = TSNE(n_components=2, random_state=0).fit_transform(vector_word)   # dimension reduction by t-SNE

words = model_word.wv.index2word    # extracted words list
words_picked = ['nothing', 'god', 'live', 'hope', 'different', 'blood', 'tear', 'love', 'heaven', 'home']   # words to be shown
for word in words_picked:
    print('Similar words of: ', word)
    print(model_word.wv.most_similar(positive=[word]))

plt.figure()
for (i, j, k) in zip(vector_word_reduced[:, 0], vector_word_reduced[:, 1], words):
    plt.plot(i, j, 'o', color='k')
    plt.annotate(k, xy=(i, j))
plt.show()


### Document Analysis
trainings = [TaggedDocument(words=data, tags=[i]) for i,data in zip(song, lyrics_split)]
model_doc = Doc2Vec(documents=trainings, dm=1, vector_size=h, alpha=0.025, window=5, min_count=5, epochs=300, seed=0, workers=6)
print(model_doc.docvecs.most_similar(['Love Me']))

vector_doc = model_doc.docvecs.doctag_syn0    # all word vectors (n * h)
vector_doc_reduced = TSNE(n_components=2, random_state=0).fit_transform(vector_doc)   # dimension reduction by t-SNE

markers = ['o', '*', '^', 'x', 'D', 'v', '+']
colors = ['k', 'r', 'c', 'g', 'b', 'm', 'k']
albumset = set(album)
albumset = list(albumset)
songarr = np.array(song)

plt.figure()
for i in range(len(albumset)):
    plt.title('All Albums')
    plotlist = []
    for j in range(n):
        if album[j] == albumset[i]:
            plotlist.append(j)
    plt.plot(vector_doc_reduced[plotlist,0], vector_doc_reduced[plotlist,1], marker=markers[i], color=colors[i], linestyle='None')
    for j in range(len(plotlist)):
        plt.annotate(songarr[plotlist[j]], xy=(vector_doc_reduced[plotlist[j],0], vector_doc_reduced[plotlist[j],1]), color=colors[i])

for i in range(len(albumset)):
    plt.figure()
    plt.xlim(-120, 120)
    plt.ylim(-120, 120)
    plt.title(albumset[i])
    plotlist = []
    for j in range(n):
        if album[j] == albumset[i]:
            plotlist.append(j)
    plt.plot(vector_doc_reduced[plotlist,0], vector_doc_reduced[plotlist,1], marker=markers[i], color=colors[i], linestyle='None')
    for j in range(len(plotlist)):
        plt.annotate(songarr[plotlist[j]], xy=(vector_doc_reduced[plotlist[j],0], vector_doc_reduced[plotlist[j],1]), color=colors[i])


### Finding the keywords in the albums by tf-idf
dictionary = corpora.Dictionary(lyrics_split)
corpus = list(map(dictionary.doc2bow, lyrics_split))
test_model = models.TfidfModel(corpus)
corpus_tfidf = test_model[corpus]
lyrics_tfidf = []
lyrics_corpus = []
for doc in corpus_tfidf:
    text_corpus = []
    text_tfidf = []
    for word in doc:
        text_corpus.append(dictionary[word[0]])
        text_tfidf.append(word[1])
    lyrics_corpus.append(text_corpus)
    lyrics_tfidf.append(text_tfidf)

nw = 3  # no. of words picked from the song
featuredwordlist = {}
for i in range(len(albumset)):
    songlist = []
    for j in range(n):
        if album[j] == albumset[i]:
            songlist.append(j)
    maxtfidf = {}
    for j in range(len(songlist)):
        lyrarr = np.array(lyrics_tfidf[songlist[j]])
        unsorted_max_indices = np.argpartition(-lyrarr, nw)[:nw]
        for k in range(nw):
            maxtfidf[lyrics_corpus[songlist[j]][unsorted_max_indices[k]]] = lyrics_tfidf[songlist[j]][unsorted_max_indices[k]]
    maxtfidf = sorted(maxtfidf.items(), key=lambda x: x[1], reverse=True)
    featuredwordlist[albumset[i]] = maxtfidf
    print('The Keywords of:', albumset[i])
    print(maxtfidf[:5])
