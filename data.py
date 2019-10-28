import os
import torch
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

class Corpus():
    def __init__(self, path='./rocstory_data/'):
        self.word_idx = {}
        self.idx_word = []
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.weight = self.get_weight()
        self.ntoken = len(self.word_idx)
        
    def tokenize(self, path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word not in self.word_idx:
                        self.word_idx[word] = len(self.word_idx)
                        self.idx_word.append(word)
                    data.append(self.word_idx[word])
        return data
    
    def get_weight(self):
        glove2word2vec('.vector_cache/glove.6B.300d.txt',".vector_cache/word2vec.txt")
        wvmodel = KeyedVectors.load_word2vec_format(".vector_cache/word2vec.txt")
        weight = torch.zeros(len(self.word_idx), 300)
        cnt = 0
        for idx, word in enumerate(self.idx_word):
            if word in wvmodel.vocab:
                weight[idx, :] = torch.from_numpy(wvmodel.get_vector(word))
            else:
                cnt += 1
        return weight
    
    def word2idx(self, word):
        return self.word_idx[word]
    
    def idx2word(self, idx):
        return self.idx_word[idx]