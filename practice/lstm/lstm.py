import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import random
import pickle
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F

train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')

def Corpus_Extr(df):
    print('Construct Corpus...')
    corpus = []
    for i in tqdm(range(len(df))):
        corpus.append(df.Phrase[i].lower().split())
    corpus = Counter(np.hstack(corpus))
    corpus = corpus
    corpus2 = sorted(corpus,key=corpus.get,reverse=True)
    print('Convert Corpus to Integers')
    vocab_to_int = {word: idx for idx,word in enumerate(corpus2,1)}
    print('Convert Phrase to Integers')
    phrase_to_int = []
    for i in tqdm(range(len(df))):
        phrase_to_int.append([vocab_to_int[word] for word in df.Phrase.values[i].lower().split()])
    return corpus,vocab_to_int,phrase_to_int
corpus,vocab_to_int,phrase_to_int = Corpus_Extr(train)

def Pad_sequences(phrase_to_int,seq_length):
    pad_sequences = np.zeros((len(phrase_to_int), seq_length),dtype=int)
    for idx,row in tqdm(enumerate(phrase_to_int),total=len(phrase_to_int)):
        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]
    return pad_sequences

pad_sequences = Pad_sequences(phrase_to_int,30)


class PhraseDataset(Dataset):
    def __init__(self, df, pad_sequences):
        super().__init__()
        self.df = df
        self.pad_sequences = pad_sequences

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if 'Sentiment' in self.df.columns:
            label = self.df['Sentiment'].values[idx]
            item = self.pad_sequences[idx]
            return item, label
        else:
            item = self.pad_sequences[idx]
            return item


def get_params(num_inputs, num_hiddens, num_outputs):
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, dtype=torch.float32), requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])

# 初始化(H,C)
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def mylstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    Y=b_q
    for sentence in inputs:
        for X in sentence:
            I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
            F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
            O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
            C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * C.tanh()
            Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    res=torch.stack(outputs, dim=0)
    return res, (H, C)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


class SentimentRNN(nn.Module):

    def __init__(self, corpus_size, output_size, embedd_dim, hidden_dim, n_layers):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.ebdim = embedd_dim

        self.embedding = nn.Embedding(corpus_size, embedd_dim)

        self.hidden = None
        self.lstm = mylstm
        self.params = get_params(embedd_dim, hidden_dim, hidden_dim)
        # 400 256 256

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden, self.params)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out.view(batch_size, -1)
        out = out[:, -5:]
        return out, hidden

    def init_hidden(self, batch_size, device):
        #         weight = next(self.parameters()).data
        #         hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
        #                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        self.hidden = (torch.zeros((batch_size, self.hidden_dim), device=device),
                       torch.zeros((batch_size, self.hidden_dim), device=device))
        return self.hidden
vocab_size = len(vocab_to_int)
print(vocab_size)
output_size = 5
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim,n_layers).to(device)

net.train()
clip=5
epochs = 50

print_every = 100
lr=0.01

# def criterion(myinput, target, size_average=True):
#     """Categorical cross-entropy with logits input and one-hot target"""
#     l = -(target * torch.log(F.softmax(myinput, dim=1) + 1e-10)).sum(1)
#     if size_average:
#         l = l.mean()
#     else:
#         l = l.sum()
#     return l
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.1, patience=10, verbose=False, threshold=0.0001,
                                                      threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
import gc

batch_size = 32
counter = 0
workers = 32
losses = []
accs = []
for e in range(epochs):
    a = np.arange(len(train))
    train_set = PhraseDataset(train, pad_sequences[a])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=workers)

    running_loss = 0.0
    running_acc = 0.0
    # batch loop
    for idx, (inputs, labels) in enumerate(train_loader):
        # initialize hidden state
        h = net.init_hidden(batch_size, device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        counter += 1
        gc.collect()
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        optimizer.zero_grad()

        if inputs.shape[0] != batch_size:
            break

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        # print(output.shape)
        # print(labels.shape)
        loss = criterion(output, labels)
        loss.backward(retain_graph=True)
        running_loss += loss.cpu().detach().numpy()
        running_acc += (output.argmax(dim=1) == labels).float().mean()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        if idx % 50 == 0:
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format((running_loss / (idx + 1))))
            print(output.argmax(dim=1))
            print(labels)
            losses.append(float(running_loss / (idx + 1)))
            print(f'acc:{running_acc / (idx + 1)}')
            accs.append(running_acc / (idx + 1))

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
plt.plot(accs)
plt.show()
plt.savefig("Loss&Acc.png")