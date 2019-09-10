import torch as T
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class TextData(Dataset):
    def __init__(self, text_data_seq, bin_ae):
        self.len = len(text_data_seq)
        self.data = text_data_seq
        self.out = bin_ae
    def __len__(self):
        return self.len
    def __getitem__(self, ind):
        return self.data[ind], self.out[ind]

class TestTextData(Dataset):
    def __init__(self, text_data_seq):
        self.len = len(text_data_seq)
        self.data = text_data_seq
    def __len__(self):
        return self.len
    def __getitem__(self, ind):
        return self.data[ind]


def create_emb_layer(weights_matrix, non_trainable = False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(int(num_embeddings), int(embedding_dim))
    weights_matrix = np.asarray(weights_matrix, dtype = np.float32)
    emb_layer.weight = nn.Parameter(T.FloatTensor(T.from_numpy(weights_matrix).to('cpu')))
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim

class TextCluster(nn.Module):
    def __init__(self, weights_matrix, target_dim):
        super(TextCluster, self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.lstm = nn.LSTM(300, 150, 2, batch_first=True, bidirectional=True)
        self.cnn = nn.Conv1d(34, 100, 5, padding = 2)
        self.pool = nn.AdaptiveAvgPool1d(150)
        self.fc1 = nn.Linear(100 * 150, target_dim)
    
    def forward(self, inp):
        h0 = T.zeros(2*2, inp.size(0), 150).to('cpu') # 2 for bidirection 
        c0 = T.zeros(2*2, inp.size(0), 150).to('cpu')
        out = self.embedding(inp)
        out, _ = self.lstm(out, (h0, c0))
        out = self.cnn(T.FloatTensor(out))
        out = self.pool(out)
        out = out.view(-1, 100 * 150)
        out = self.fc1(out)
        return out
