from gensim.models import Word2Vec
import jieba
import pandas as pd 
import numpy as np
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, CuDNNLSTM, Bidirectional, PReLU, CuDNNGRU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
# python -u rnn.py train_x.csv dict.txt.big 0430_word2vec_withoutB train_y.csv rnn_...

# Load data
data = pd.read_csv(sys.argv[1])
jieba.load_userdict(sys.argv[2])
Y = np.genfromtxt(sys.argv[4], delimiter = ',', encoding = 'big5')
Y = Y[1:, 1].reshape(-1, 1)
Y = np.hstack((Y == 0, Y == 1))

# Dealing with data
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

# Build model:
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
							output_dim=embedding_matrix.shape[1],
							weights=[embedding_matrix],
							trainable=True)

print('X, Y shape :', X.shape, Y.shape)
model = Sequential()
model.add(embedding_layer)
'''
model.add(Bidirectional(CuDNNLSTM(150, return_sequences = True)))
model.add(Bidirectional(CuDNNLSTM(150)))
'''
model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
model.add(Bidirectional(CuDNNGRU(128)))
model.add(Dropout(0.1))
model.add(Dense(units = 64))
model.add(PReLU(alpha_initializer = 'zeros'))
model.add(Dropout(0.5))
model.add(Dense(units = 32))
model.add(PReLU(alpha_initializer = 'zeros'))
model.add(Dropout(0.5))
model.add(Dense(units = 2, activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

opt_filepath = sys.argv[5]
checkpoint = ModelCheckpoint(opt_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0)
callbacks_list = [checkpoint, reduce_lr]

model.fit(X, Y, batch_size = 500, epochs = 10, validation_split=0.2, callbacks = callbacks_list, shuffle = True)

# Source: https://www.kaggle.com/jerrykuo7727/embedding-rnn-0-876