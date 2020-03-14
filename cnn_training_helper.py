import torch
from data_prep import Password as P
from config import *

def get_train_dis(strings_in, pred, seq_len):
    real = get_real(strings_in, seq_len)
    fake = get_fake(real, pred, seq_len)
    return real, fake

def get_train_gen(strings_in, pred, seq_len):
    real = get_real(strings_in, seq_len)
    fake = get_fake(real, pred, seq_len)
    return fake

def get_fake(real, pred, seq_len):
    train_pred = torch.zeros(0, MAX_LEN+1, CHARMAP_LEN).to(device)
    real = real.to(torch.float)
    for i in range(1, seq_len+1):
        train_pred = torch.cat((
            train_pred,
            torch.cat((
                real[:BATCH_SIZE,:i,:],
                pred[:,i-1:i,:],
                #torch.zeros(BATCH_SIZE, MAX_LEN-i, CHARMAP_LEN).to(device)
                EOS_padding(torch.zeros(BATCH_SIZE, MAX_LEN-i, CHARMAP_LEN))
                ),dim = 1)
            ),dim = 0)
    return train_pred

def get_real(strings_in, seq_len):
    real = torch.FloatTensor(BATCH_SIZE, MAX_LEN+1, CHARMAP_LEN).zero_().to(device)
    for i in range(BATCH_SIZE):
        # l = len(strings_in[i]) if len(strings_in[i]) <= seq_len else seq_len
        if len(strings_in[i]) <= seq_len:
            l = len(strings_in[i])
            real[i][l][CHARMAP_LEN - 1] = 1
        else:
            l = seq_len
        for j in range(l):
            real[i][j][P.letterToIndex(strings_in[i][j])] = 1

    #realreal = torch.zeros(0, MAX_LEN+1, CHARMAP_LEN).to(device)

    for i in range(1, seq_len):
        real = torch.cat((
            real,
            torch.cat((
                real[:BATCH_SIZE,:i+1,:],
                torch.zeros(BATCH_SIZE, MAX_LEN-i, CHARMAP_LEN).scatter_(-1, torch.full([BATCH_SIZE, MAX_LEN-i, 1], CHARMAP_LEN-1).long(), 1).to(device)
                ),dim = 1)
            ),dim = 0)

    return real

def get_interpolate(real, fake):
    real = torch.autograd.Variable(real.data).data
    fake = torch.autograd.Variable(fake.data).data
    alpha = torch.rand(real.size()[0], 1, 1).to(device)
    alpha = alpha.expand(real.size())
    return torch.autograd.Variable(alpha * real + (1 - alpha) * fake).data

def EOS_padding(shape):
    if (shape.size()[1] != 0):
        shape = shape.scatter_(-1, torch.full([shape.size()[0], shape.size()[1],1], CHARMAP_LEN-1).long(), 1).to(device)
    return shape