from gensim.models import Word2Vec
import jieba
import pandas as pd 
import numpy as np
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
#python -u submit_ensenble.py test_x.csv dict.txt.big 0430_word2vec_withoutB blending_test.csv rnn_05022044 rnn_05021909 rnn_05021957 rnn_05022118 rnn_05031540
data = pd.read_csv(sys.argv[1])
jieba.load_userdict(sys.argv[2])

# Dealing data
seg_list = []
for j in data['comment'].values:
	seg = jieba.lcut(j, cut_all = False)
	for w in seg:
		if w == ' ' or w[0] == 'B':
			seg.remove(w)
	seg_list.append(seg)
seg_list = np.array(seg_list)
model_w2v = Word2Vec.load(sys.argv[3])

embedding_matrix = np.zeros((len(model_w2v.wv.vocab.items()) + 1, model_w2v.vector_size))
#print(embedding_matrix.shape)
word2idx = {}

vocab_list = [(word, model_w2v.wv[word]) for word, _ in model_w2v.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
	word, vec = vocab
	embedding_matrix[i + 1] = vec
	word2idx[word] = i + 1

def text_to_index(corpus):
	new_corpus = []
	for doc in corpus:
		new_doc = []
		for word in doc:
			try:
				new_doc.append(word2idx[word])
			except:
				new_doc.append(0)
		new_corpus.append(new_doc)
	return np.array(new_corpus)

PADDING_LENGTH = 66
X = text_to_index(seg_list)
X = pad_sequences(X, maxlen=PADDING_LENGTH)
print('end pad_sequences')

pred = 0
for i in range(5, 10):
	model = load_model(sys.argv[i])
	pred += model.predict(X)
opt = np.argmax(pred, axis = 1)

f = open(sys.argv[4], 'w')
wt_str = 'id,label\n'
f.write(wt_str)
for i in range(opt.shape[0]):
	wt_str = str(i)+','+str(int(opt[i]))+'\n'
	f.write(wt_str)
print('ending!!!')