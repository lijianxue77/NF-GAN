import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.init as init
import json


class Generator(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, max_seq_len, gpu=False, oracle_init=False):
        super(Generator, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.gru2out = nn.Linear(self.hidden_size, self.vocab_size)

        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden_layer(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))
        if self.gpu:
            return h.cuda()
        return h

    def forward(self, inp, hidden):
        emb = self.embedding(inp)  # batch_size*embedding_size
        emb = emb.view(1, -1, self.embedding_size)  # 1*batch_size*embedding_size
        out, hidden = self.gru(emb, hidden)  # out: 1*batch_size*hidden_size
        out = self.gru2out(out.view(-1, self.hidden_size))  # batch_size*vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, n_samples, start_letter=0):
        samples = torch.zeros(n_samples, self.max_seq_len).type(torch.LongTensor)
        h = self.init_hidden_layer(n_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter] * n_samples))
        if self.gpu:
            h = h.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)  # n_samples(batch_size)*vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # out: n_samples(batch_size)*1
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)  # input can be a 1D vector

        return samples

    def batchNLLLoss(self, inp, target):
        """return a loss function used for pre-training"""
        # inp should be target with start word <s> prepended
        # construct input sequence and target sequence ?
        criterion = nn.NLLLoss()
        batch_size, seq_len = inp.size()  # batch_size*seq_len
        inp = inp.permute(1, 0)  # seq_len*batch_size
        target = target.permute(1, 0)  # seq_len * batch_size
        h = self.init_hidden_layer(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += criterion(out, target[i])

        return loss

    def batchPGLoss(self, inp, target, reward):
        """ return a loss function as policy gradient for training"""
        # How to understand policy gradient?
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0) # seq_len*batch_size
        target = target.permute(1, 0) # seq_len*batch_size
        h = self.init_hidden_layer(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h) # out: batch_size*vocab
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]] * reward[j] # reward: batch_size

        return loss/batch_size


class Discriminator(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.gpu = gpu

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=2, bidirectional=True, dropout=self.dropout)
        self.gru2hidden = nn.Linear(4*hidden_size, hidden_size)
        self.dropout_linear = nn.Dropout(p=self.dropout)
        self.hidden2out = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(4, batch_size, self.hidden_size))
        if self.gpu:
            return h.cuda()
        return h

    def forward(self, inp, h):
        # input: batch_size*seq_len
        emb = self.embedding(inp) # batch_size*seq_len*embedding_size
        emb = emb.permute(1, 0, 2) # seq_len*batch_size*embedding_size
        _, h = self.gru(emb, h) # 4*batch_size*hidden_size
        h = h.permute(1, 0, 2).contiguous() # batch_size*4*hidden_size
        out = self.gru2hidden(h.view(-1, 4*self.embedding_size)) # batch_size*(4*hidden_size)
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out) # batch_size*1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp):
        """

        :param inp: batch_size*seq_len
        :return: batch_size
        """
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """

        :param inp: batch_size*seq_len
        :param target: batch_size
        :return:
        """
        criterion = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        loss = criterion(out, target)
        return loss


class Config:
    def __init__(self, path='./conf/params.json'):
        self.config = None
        with open(path, 'r') as file:
            self.config = json.load(file)

    def get(self, feature):
        if self.config is not None and self.config[feature] is not None:
            return self.config[feature]
        return None
