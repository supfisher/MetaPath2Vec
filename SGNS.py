import torch.nn as nn
import torch as t
from torch.nn import init


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.init_params()

    def init_params(self):
        initrange = 1.0 / self.embedding_dim
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):

        return self.u_embeddings(pos_u), self.v_embeddings(pos_v), self.v_embeddings(neg_v).neg()


class SGNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_negs=20):
        super(SGNS, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = Word2Vec(vocab_size, embedding_dim)
        self.n_negs = n_negs

    def forward(self, pos_u, pos_v, weights):
        batch_size = pos_u.shape[0]
        context_size = pos_v.size()[1]
        if weights is not None:
            nwords = t.multinomial(weights, batch_size * context_size * self.n_negs, replacement=True).view(
                batch_size, -1).long()
        else:
            nwords = t.zeros(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()

        emb_u, emb_v, emb_neg_u = self.embedding(pos_u, pos_v, nwords)
        emb_u = emb_u.unsqueeze(2)

        oloss = t.bmm(emb_v, emb_u).squeeze().clamp(max=10, min=-10).sigmoid().log().mean(1)
        nloss = t.bmm(emb_neg_u, emb_u).squeeze().clamp(max=10, min=-10).sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()

    def save_embeddings(self, file_path):
        t.save({'in_embeddings': self.embedding.u_embeddings.weight.cpu().data, 'context_embeddings': self.embedding.v_embeddings.weight.cpu().data}, file_path)

