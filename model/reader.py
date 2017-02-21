# Some inspiration from https://github.com/carpedm20/lstm-char-cnn-tensorflow/
## ===========================================================================
from __future__ import division

import collections
import os
import sys
import time
import random
from itertools import islice

import numpy as np
import tensorflow as tf

import pickle
from nltk.util import ngrams
from tensorflow.python.platform import gfile

"""
TODO: Gestion des mots inconnus: si avant filtrage, pas pris en compte dans la distrib unigram ?
TODO: classe dataset version n-grams
"""

def _filter_vocab(vocab, threshold):
    return dict((k, v) for k, v in vocab.iteritems() if v >= threshold)

class dataset():
    def __init__(self,
                 dir_path,                 
                 data_file,
                 vocab_file,
                 reps,
                 max_seq_length,
                 max_word_length = None,
                 vocab_threshold = 1,
                 shuffle = True):

        self.path = os.path.join(dir_path, data_file)       
        self.vocab_data_path = os.path.join(dir_path, vocab_file + '.vocab.pkl')
        self.reps = reps
        self.outputRep = outputRep

        # If the vocabulary is not already saved:
        if not os.path.exists(self.vocab_data_path):
            self._file_to_vocab()
    
        with open(self.vocab_data_path, 'r') as _file:
            [ word_to_id, _, char_to_id, _, max_lengths] = pickle.load(_file)       
        
        #max_seq_length = min([max_lengths[0], max_seq_length])
        #max_word_length = min([max_lengths[1], max_word_length])
        word_to_id = _filter_vocab(word_to_id, vocab_threshold)
        char_to_id = _filter_vocab(char_to_id, vocab_threshold)

        if (reps[0] or reps[2]) and (reps[1] or reps[3]):
            self.train_word_tensor, self.train_char_tensor = self._file_to_data(word_to_id, char_to_id, max_seq_length, max_word_length)
        elif (reps[0] or reps[2]):
            self.train_word_tensor = self._file_to_data(word_to_id, char_to_id, max_seq_length, max_word_length)
        elif (reps[1] or reps[3]):
            self.train_char_tensor = self._file_to_data(word_to_id, char_to_id, max_seq_length, max_word_length)

        self.ids = range(self.tot)
        if shuffle:
            random.shuffle(self.ids)
        self.cpt = 0

    def _file_to_vocab(self):
        # Get words, find longuest sentence and longuest word
        data = []
        temp_max_seq_length = 0
        temp_max_word_length = 0      
        with open(self.train_path) as train_file:
            for line in train_file:
                seq = line.strip().split()
                temp_max_seq_length = max([temp_max_seq_length, len(seq)])
                temp_max_word_length = max([temp_max_seq_length] + [len(word) for word in seq])
                data.append(seq)

        # Get complete word and characters vocabulary
        word_counter = collections.Counter([word for seq in data for word in seq])
        word_pairs = sorted(word_counter.items(), key=lambda x: -x[1])
        words, w_counts = list(zip(*word_pairs))
        word_to_id = dict(zip(words, range(4, len(words)+4)))
        #id_to_word = dict(zip(range(4, len(words)), words))
        
        char_counter = collections.Counter([char for seq in data for word in seq for char in word])
        char_pairs = sorted(char_counter.items(), key=lambda x: -x[1])
        chars, c_counts = list(zip(*char_pairs))
        char_to_id = dict(zip(chars, range(4, len(chars)+4)))
        #id_to_char = dict(zip(range(4, len(chars)), chars))
        
        with open(self.vocab_data_path, 'w') as vocab_file:
            pickle.dump([word_to_id, w_counts, char_to_id, c_counts, np.array([temp_max_seq_length, temp_max_word_length, len(data)])], vocab_file)


    def _file_to_data(self, word_to_id, char_to_id, max_seq_length, max_word_length):
        # Create tensors and fill them with tokens
        if (self.reps[0] or self.reps[2]):
            word_train_tensor = np.ndarray(shape = (self.tot, max_seq_length + 1), dtype=int)
        if (self.reps[1] or self.reps[3]):
            char_train_tensor = np.zeros(shape = (self.tot, max_seq_length + 1, max_word_length), dtype=int)
        id_seq = 0
        with open(self.path) as train_file:
            for line in train_file:
                seq = line.strip().split()
                if (self.reps[0] or self.reps[2]):
                    word_train_tensor[id_seq] = np.ndarray(shape= (max_seq_length + 1,),
                                                           buffer= np.array([word_to_id.get(w, 1) for w in seq]+[0]*(max_seq_length + 1 - len(seq))),
                                                           dtype= int)
                if (self.reps[1] or self.reps[3]):
                    for id_word, word in enumerate(seq):
                        if id_word < max_seq_length:
                            char_train_tensor[id_seq, id_word] = np.ndarray(shape= (max_word_length,),
                                                                            buffer= np.array([char_to_id.get(c, 1) for c in word]+[0]*(max_word_length - len(word))),
                                                                            dtype= int)
                id_seq += 1
        self.tot = id_seq
        if (self.reps[1] or self.reps[3]) and (self.reps[2] or self.reps[0]):
            return word_train_tensor, char_train_tensor
        elif (self.reps[0] or self.reps[2]):
            return word_train_tensor
        elif (self.reps[1] or self.reps[3]):
            return char_train_tensor

    def sampler_seq(self, batch_size):
        while True:
            if (self.cpt+batch_size > self.tot):
                self.cpt=0
                random.shuffle(self.ids)
            out = []
            if self.reps[0]:
                x = self.train_word_tensor[self.ids[self.cpt:self.cpt+batch_size],:-1]
                out.append(x)
            if self.reps[1]:
                c = self.train_char_tensor[self.ids[self.cpt:self.cpt+batch_size],:-1,:]
                out.append(c)
            if self.reps[2]:
                y = self.train_word_tensor[self.ids[self.cpt:self.cpt+batch_size],1:]
                out.append(y)
            if self.reps[3]:
                yc = self.train_char_tensor[self.ids[self.cpt:self.cpt+batch_size],1:,:]
                out.append(yc)
            yield out

class datasetQ():
    def __init__(self,
                 dir_path,
                 data_file,
                 vocab_file,
                 reps,
                 max_seq_length = None,
                 context_length = None,
                 max_word_length = None,
                 vocab_threshold = 1):
        """
        Caution: this iterator will open the file and process lines on the fly before
        yielding them, (to be able to work with files too big to fit in memory
        which implies there is no data shuffling. Training data must be shuffled using:
        shuf --head-count=NB_SEQ train
        """
        self.path = os.path.join(dir_path, data_file)
        self.vocab_data_path = os.path.join(dir_path, vocab_file + '.vocab.pkl')
        self.reps = reps

        # If the vocabulary is not already saved:
        if not os.path.exists(self.vocab_data_path):
            self._file_to_vocab()

        with open(self.vocab_data_path, 'r') as _file:
            [ word_to_id, word_counts, char_to_id, char_counts, max_lengths] = pickle.load(_file)

        self.word_to_id = _filter_vocab(word_to_id, vocab_threshold)
        self.char_to_id = _filter_vocab(char_to_id, vocab_threshold)

        #self.max_seq_length = min([max_lengths[0], max_seq_length])
        #self.max_word_length = min([max_lengths[1], max_word_length])
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length
        self.tot = max_lengths[2]        

    def _file_to_vocab(self):
        # Get words, find longuest sentence and longuest word
        data = []
        temp_max_seq_length = 0
        temp_max_word_length = 0
        with open(self.path) as train_file:
            for line in train_file:
                seq = line.strip().split()
                temp_max_seq_length = max([temp_max_seq_length, len(seq)])
                temp_max_word_length = max([temp_max_seq_length] + [len(word) for word in seq])
                data.append(seq)

        # Get complete word and characters vocabulary
        word_counter = collections.Counter([word for seq in data for word in seq])
        word_pairs = sorted(word_counter.items(), key=lambda x: -x[1])
        words, w_counts = list(zip(*word_pairs))
        word_to_id = dict(zip(words, range(4, len(words)+4)))
        #id_to_word = dict(zip(range(4, len(words)), words))

        char_counter = collections.Counter([char for seq in data for word in seq for char in word])
        char_pairs = sorted(char_counter.items(), key=lambda x: -x[1])
        chars, c_counts = list(zip(*char_pairs))
        char_to_id = dict(zip(chars, range(4, len(chars)+4)))
        #id_to_char = dict(zip(range(4, len(chars)), chars))

        with open(self.vocab_data_path, 'w') as vocab_file:
            pickle.dump([word_to_id, w_counts, char_to_id, c_counts, np.array([temp_max_seq_length, temp_max_word_length, len(data)])], vocab_file)

    def sampler_ngrams(self, n, batch_size):
        with open(self.path) as _file:
            while True:
                to_be_read = list(islice(_file, batch_size))
                if len(to_be_read) < batch_size:
                    _file.seek(0)
                    to_be_read = list(islice(_file, batch_size))
                """
                TODO: one loop for word/char faster than two loops with list comprehensions ? 
                """
                n_grams = [ngrams(sent.strip().split(),n) for sent in to_be_read]
                if self.reps[0] or self.reps[2]:
                    word_train_tensor = np.array([[self.word_to_id.get(w, 1) for w in n_gram] for sent in n_grams for n_gram in sent], dtype = 'int32')
                if self.reps[1] or self.reps[3]:
                    char_train_tensor=np.array([[np.ndarray(shape=(self.max_word_length,),
                                                            buffer=np.array([self.char_to_id.get(c,1) for c in word]+[0]*(self.max_word_length - len(word))),
                                                            dtype=int) for word in n_gram] for sent in n_grams for n_gram in sent], dtype = 'int32')    
                out = []
                if self.reps[0]:
                    x = word_train_tensor[:,:-1]
                    out.append(x)
                if self.reps[1]:
                    c = char_train_tensor[:,:-1,:]
                    out.append(c)
                if self.reps[2]:
                    y = word_train_tensor[:,-1]
                    out.append(y)
                if self.reps[3]:
                    yc = char_train_tensor[:,-1,:]
                    out.append(yc)
                yield out

    def sampler_seq(self, batch_size):
        with open(self.path) as _file:
            while True:
                to_be_read = list(islice(_file, batch_size))
                if len(to_be_read) < batch_size:
                    _file.seek(0)
                    to_be_read = list(islice(_file, batch_size))
                if self.reps[0] or  self.reps[2]:
                    word_train_tensor = np.ndarray(shape = (batch_size, self.max_seq_length+1), dtype=int)
                if self.reps[1] or  self.reps[3]:
                    char_train_tensor = np.zeros(shape = (batch_size, self.max_seq_length+1, self.max_word_length), dtype=int)
                for id_seq, line in enumerate(to_be_read):
                    seq = line.strip().split()
                    if self.reps[0] or self.reps[2]:
                        word_train_tensor[id_seq]=np.ndarray(shape=(self.max_seq_length+1,),
                                                             buffer=np.array([self.word_to_id.get(w, 1) for w in seq]+[0]*(self.max_seq_length + 1 - len(seq))),
                                                             dtype=int)
                    if self.reps[1] or self.reps[3]:
                        for id_word, word in enumerate(seq):
                            if id_word < self.max_seq_length:
                                char_train_tensor[id_seq,id_word]=np.ndarray(shape=(self.max_word_length,),
                                                                             buffer=np.array([self.char_to_id.get(c,1) for c in word]+[0]*(self.max_word_length - len(word))),
                                                                             dtype=int)                                
                out = []
                if self.reps[0]:
                    x = word_train_tensor[:,:-1]
                    out.append(x)
                if self.reps[1]:
                    c = char_train_tensor[:,:-1,:]
                    out.append(c)
                if self.reps[2]:
                    y = word_train_tensor[:,1:]
                    out.append(y)
                if self.reps[3]:
                    yc = char_train_tensor[:,1:,:]
                    out.append(yc)
                yield out

#For test
"""
path = "../data"
train = "train"
vocab = "train"
d = datasetQ(path, train, vocab, [True, True, True, True], 30, None, 15)                                      
s = d.sampler_seq(1)
print(s.next())
"""



        

