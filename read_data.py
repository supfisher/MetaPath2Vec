from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch as t
import os
from itertools import chain


class ReadData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = np.zeros([2000,2])

        self.sentences_count = 0
        self.token_count = 0

        self.word2id, self.id2word, self.word_frequency = {}, {}, {}

        self.readfile_txt()
        self.initTableDiscards()

    def load_data(self):
        """
        :param filepath: a path for a npy object, each line represents a sampled random walk
        """

        np_data = np.load(self.filepath)
        len_pairs = 0
        for line in np_data:
            line = line[line!=-1]
            for i, l in enumerate(line):
                context = line[max(i-self.window_size, 0): min(i+self.window_size, line.shape[-1])]
                if len_pairs + len(context) > self.data.shape[0]:
                    self.data = np.vstack((self.data, self.data))

                self.data[len_pairs:len_pairs + len(context), 0] = l
                self.data[len_pairs:len_pairs + len(context), 1] = context
                len_pairs += len(context)

        self.data = self.data[0:len_pairs]

    def readfile_txt(self, min_count=5):
        data_path = 'data.pkl'

        if os.path.exists(data_path):
            dict = t.load(data_path)
            self.word_frequency = dict['word_frequency']
            self.word2id = dict['word2id']
            self.id2word = dict['id2word']
            self.token_count = dict['token_count']
            self.sentences_count = dict['sentences_count']
            self.word_count = len(self.word2id)
        else:
            word_frequency = {}
            f = open(self.filepath, encoding="ISO-8859-1")
            lines = f.readlines()
            for line in lines:
                line = line.split()
                if len(line) > 1:
                    self.sentences_count += 1
                    for word in line:
                        if len(word) > 0:
                            self.token_count += 1
                            word_frequency[word] = word_frequency.get(word, 0) + 1
                            if self.token_count % 1000000 == 0:
                                print("Read " + str(int(self.token_count / 1000000)) + "M words.")

            wid = 0
            for w, c in word_frequency.items():
                if c < min_count:
                    continue
                self.word2id[w] = wid
                self.id2word[wid] = w
                self.word_frequency[wid] = c
                wid += 1

            self.word_count = len(self.word2id)
            print("Total embeddings: " + str(len(self.word2id)))

            t.save({'word_frequency': self.word_frequency, 'word2id': self.word2id,
                    'id2word': self.id2word, 'token_count': self.token_count,
                    'sentences_count': self.sentences_count, 'word_count': self.word_count}, 'data.pkl')

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)


class Copus(Dataset):
    def __init__(self, read_data, window_size=7):
        super(Copus, self).__init__()
        self.window_size = window_size
        self.data = read_data
        self.input_file = open(self.data.filepath, encoding="ISO-8859-1")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, index):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    us, vs = [], []

                    for i, u in enumerate(word_ids):
                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i + self.window_size]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            us.append(u)
                            vs.append(v)
                    return (us, vs)

    @staticmethod
    def collate(batches):
        all_u, all_v = zip(*batches)
        all_u = t.LongTensor(list(chain(*all_u)))
        all_v = t.LongTensor(list(chain(*all_v)))

        return all_u, all_v
