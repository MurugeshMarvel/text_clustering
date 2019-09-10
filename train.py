import numpy as np
import os
import torch.nn as nn
import model
from tqdm import tqdm
import torch.optim as optim
import torch as T
import utils
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

TRAIN_FILE = 'data/StackOverflow.txt'
EMB_FILE = 'GoogleNews-vectors-negative300.bin'
OUT_FILE = 'data/StackOverflow_gnd.txt'
EPOCHS_NUM = 10

with open(TRAIN_FILE, 'r') as f:
    data = [text.strip() for text in f]

with open(OUT_FILE) as f:
    target = f.readlines()
target = [int(label.rstrip('\n')) for label in target]

trn_data, test_data = train_test_split(data, test_size = 0.15)
tokenizer = Tokenizer(char_level= False)
tokenizer.fit_on_texts(data)
full_seq = tokenizer.texts_to_sequences(trn_data)
word_index = tokenizer.word_index
max_words = len(word_index)
print (f'Found {max_words} words in the dataset')

seq_lens = [len(s) for s in full_seq]
max_seq_len = max(seq_lens)

inp_data = pad_sequences(full_seq, maxlen= max_seq_len)

## Preparing Embedding Matrix
w2v = KeyedVectors.load_word2vec_format(EMB_FILE, binary= True)
Embedding_dim = 300
nb_words = min(max_words, len(word_index)) + 1
embedding_matrix = np.zeros((nb_words, Embedding_dim))

for word, i in word_index.items():
    if word in w2v.vocab:
        embedding_matrix[i] = w2v.word_vec(word)
    else:
        #print (word)
        pass
#target using Average Embeddings

y = {}
tfidf = tokenizer.sequences_to_matrix(full_seq, mode = 'tfidf')
denom = 1 + np.sum(tfidf, axis =1)[:, None]
normed_tfidf = tfidf/denom
average_embeddings = np.dot(normed_tfidf, embedding_matrix)
y['ae'] = average_embeddings
print (f"Shape of the Average Embeddings: {y['ae'].shape}")

B = utils.binarize(y["ae"])
target_dim = B.shape[1]

print (f'The Shape of Binarized Average Embeddings is {B.shape}')
print (f'The shape of the train_inp_data is {inp_data.shape}')

train_inp_data = model.TextData(inp_data, B)
train_dataloader = DataLoader(train_inp_data, shuffle = True, batch_size = 100)
MODEL = model.TextCluster(embedding_matrix, target_dim)

crit = nn.MSELoss()
optimizer = optim.Adam(MODEL.parameters(), lr = 1e-3, betas = [0.9, 0.999], eps = 1e-08)

for epoch in range(1, EPOCHS_NUM +  1):
    print (f'EPOCH-{epoch}')
    for batch in tqdm(train_dataloader):
        txt_inp, bout = batch
        txt_inp = txt_inp.type(T.LongTensor)
        bout = bout.type(T.FloatTensor)
        optimizer.zero_grad()
        pred_bout = MODEL(txt_inp)
        loss = crit(pred_bout, bout)
        loss.backward()
        optimizer.step()
    print(f"The loss for Epoch - {epoch} is {loss/ 100}")

test_full_seq = tokenizer.texts_to_sequences(test_data)
test_data = pad_sequences(test_full_seq, maxlen= max_seq_len)

test_inp_data = model.TestTextData(inp_data)
test_dataloader = DataLoader(test_inp_data, shuffle = True, batch_size = 100)

pred_out_list = []
for test_inp_text in test_dataloader:
    output = MODEL(test_inp_text.type())
    pred_out_list += list(output.detach().numpy())

y = target
true_labels = y
n_clusters = len(np.unique(y))
print("Number of classes: %d" % n_clusters)
km = KMeans(n_clusters=n_clusters, n_jobs=10)
result = dict()
V = normalize(np.array(pred_out_list), norm='l2')
km.fit(V)
pred = km.labels_
print(pred)
a = {'deep': utils.cluster_quality(true_labels, pred)}