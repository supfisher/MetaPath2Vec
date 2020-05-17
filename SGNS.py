import torch.nn as nn
import torch as t
from torch.nn.parameter import Parameter

class Test(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        super(Test, self).__init__()
        self.embedding_dim = embedding_dim
        self.ivectors = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.init_params()

    def init_params(self):
        self.ivectors.weight.data.uniform_(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        self.ovectors.weight.data.uniform_(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim)


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.ivectors = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.init_params()

    def init_params(self):
        self.ivectors.weight.data.uniform_(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        self.ovectors.weight.data.uniform_(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim)

    def forward(self, iwords, owords, nwords):
        return self.ivectors[iwords], self.ovectors[owords], self.ovectors[nwords].neg()


class SGNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = Word2Vec(vocab_size, embedding_dim)
        self.n_negs = n_negs

        self.weights = None
        if weights == None:
            wf = t.pow(weights, 0.75)
            self.weights = (wf/wf.sum()).float()

    def forward(self, iwords, owords):
        batch_size = iwords.shape[0]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * self.n_negs, replacement=True).view(
                batch_size, -1)
        else:
            nwords = t.zeros(batch_size, self.n_negs).uniform_(0, self.vocab_size - 1).long()

        ivectors, ovectors, nvectors = self.embedding(iwords, owords, nwords)
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, self.n_negs).mean(1)
        return -(oloss + nloss).mean()

    def save_embeddings(self):
        t.save({'in_embeddings': self.embedding.ivectors, 'context_embeddings': self.embedding.ovectors}, './embeddings.pkl')


if __name__ == "__main__":
    t = Test(100,10)
    print(t.ivectors)