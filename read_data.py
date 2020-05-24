from torch.utils.data import Dataset
import numpy as np
import torch as t
import os
from itertools import chain
import pickle


class ReadData:
    def __init__(self, filepath, window=5, unk='<UNK>', data_dir='./data'):
        self.filepath = filepath
        self.unk = unk
        self.window = window
        self.data_dir = data_dir

        self.sentences_count = 0
        self.word2idx, self.word_count = {}, {}
        self.idx2word = []

        self.data = []
        if os.path.exists(os.path.join(self.data_dir, 'word_count.dat')):
            self.word_count = pickle.load(open(os.path.join(self.data_dir, 'word_count.dat'), 'rb'))
            self.idx2word = pickle.load(open(os.path.join(self.data_dir, 'idx2word.dat'), 'rb'))
            self.word2idx = pickle.load(open(os.path.join(self.data_dir, 'word2idx.dat'), 'rb'))
            self.sentences_count = pickle.load(open(os.path.join(self.data_dir, 'sentences_count.dat'), 'rb'))
        else:
            self.build()

    def build(self, max_vocab=100000):
        print("building vocab...")
        self.word_count = {self.unk: 1}
        f = open(self.filepath, encoding="ISO-8859-1")
        lines = f.readlines()
        f.close()
        for line in lines:
            self.sentences_count += 1
            if not self.sentences_count % 10000:
                print("working on {}kth line".format(self.sentences_count // 1000), end='\r')
            line = line.strip()
            if not line:
                continue
            sent = line.split()
            for word in sent:
                self.word_count[word] = self.word_count.get(word, 0) + 1
        print("")
        self.idx2word = [self.unk] + sorted(self.word_count, key=self.word_count.get, reverse=True)[0:max_vocab-1]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

        pickle.dump(self.word_count, open(os.path.join(self.data_dir, 'word_count.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        pickle.dump(self.sentences_count, open(os.path.join(self.data_dir, 'sentences_count.dat'), 'wb'))
        print("build done")

    # def skipgram(self, sentence_idx):
    #     iwords = sentence_idx
    #     cat_sentence_idx = np.array([0]*self.window + sentence_idx + [0]*self.window)
    #     owords = np.zeros([len(sentence_idx), 2*self.window])
    #
    #     for i in range(len(sentence_idx)):
    #         owords[i,0:self.window]=cat_sentence_idx[i:i+self.window]
    #         owords[i, self.window:2*self.window] = cat_sentence_idx[i + self.window+1:i + 2*self.window+1]
    #     return iwords, owords
    def skipgram(self, sentence_idx):
        iwords = np.array(sentence_idx)
        len_sentence_idx = len(sentence_idx)
        sentence_idx = [0] * self.window + sentence_idx + [0] * self.window
        owords = np.array(list(map(lambda i: sentence_idx[i:i+self.window]+sentence_idx[i+self.window+1:i+2*self.window+1], range(len_sentence_idx))))

        return iwords, owords

    def convert(self):
        print("converting corpus...")
        # self.word_count = pickle.load(open(os.path.join(self.data_dir, 'word_count.dat'), 'rb'))
        # self.idx2word = pickle.load(open(os.path.join(self.data_dir, 'idx2word.dat'), 'rb'))
        # self.word2idx = pickle.load(open(os.path.join(self.data_dir, 'word2idx.dat'), 'rb'))

        # print("finished load file...")

        ## change all words not appeared in word2idx to 0
        self.all_word2idx = self.word2idx.copy()
        tmp_dict = {w: 0 for w in set.difference(set(self.word_count.keys()), set(self.word2idx.keys()))}
        self.all_word2idx.update(tmp_dict)

        self.token_count = sum(self.word_count.values())
        print("token_count: ", self.token_count)
        self.data = np.zeros([self.token_count, 1+2*self.window], dtype=np.int64)

        print("begin reading file...")

        step = 0
        token_count = 0
        f = open(self.filepath, encoding="ISO-8859-1")
        lines = f.readlines()
        f.close()
        print("finshed reading file...")
        for line in lines:
            step += 1
            if not step % 10000:
                print("working on {}kth line".format(step // 1000), end='\r')

            line = line.strip()
            if not line:
                continue

            line = line.split()
            sentence_idx = list(map(lambda word: self.all_word2idx[word], line))
            iwords, owords = self.skipgram(sentence_idx)
            self.data[token_count: token_count+len(sentence_idx), 0] = iwords
            self.data[token_count: token_count + len(sentence_idx), 1:] = owords
            token_count += len(sentence_idx)
        print("")
        np.save(os.path.join(self.data_dir, 'data.npy'), self.data)
        print("conversion done")


# class Copus(Dataset):
#     def __init__(self, read_data, window=5):
#         super(Copus, self).__init__()
#         self.read_data = read_data
#         self.window = window
#
#         self.input_file = open(self.read_data.filepath, encoding="ISO-8859-1")
#         self.all_word2idx = {w: 0 for w in self.read_data.word_count.keys()}
#         for w in self.read_data.vocab:
#             self.all_word2idx[w] = self.read_data.word2idx[w]
#
#         self.lines = self.input_file.readlines()
#
#     def __len__(self):
#         return self.read_data.sentences_count
#
#     # def __getitem__(self, index):
#     #     return self.data[index][0], self.data[index][1:]
#
#     def __getitem__(self, index):
#         line = self.lines[index].strip()
#         if len(line) > 1:
#             words = line.split()
#             if len(words) > 1:
#                 sentence_idx = list(map(lambda word: self.all_word2idx[word], words))
#                 return self.skipgram(sentence_idx)
#
#     def skipgram(self, sentence_idx):
#         iwords = sentence_idx
#         cat_sentence_idx = np.array([0]*self.window + sentence_idx + [0]*self.window)
#         owords = np.zeros([len(sentence_idx), 2*self.window])
#
#         for i in range(len(sentence_idx)):
#             owords[i,0:self.window]=cat_sentence_idx[i:i+self.window]
#             owords[i, self.window:2*self.window] = cat_sentence_idx[i + self.window+1:i + 2*self.window+1]
#
#         return iwords, owords
#
#     @staticmethod
#     def collate(batches):
#         all_u, all_v = zip(*batches)
#         all_u = t.LongTensor(list(chain(*all_u)))
#         all_v = t.LongTensor(list(chain(*all_v)))
#
#         return all_u, all_v



class Copus(Dataset):
    def __init__(self, data_dir):
        super(Copus, self).__init__()
        self.data = t.from_numpy(np.load(os.path.join(data_dir, 'data.npy'))).long()
        self.len_data = self.data.shape[0]

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.data[index, 0], self.data[index, 1:]


if __name__ == '__main__':
    read_data = ReadData("/ibex/scratch/mag0a/Github/data/aminer.txt")
    read_data.convert()