import torch
from data_prep import Password as P
from config import *

def get_real(strings_in, seq_len):
    #init with all EOS
    real = torch.zeros(BATCH_SIZE, seq_len, CHARMAP_LEN).to(device)
    for i in range(BATCH_SIZE):
        # l = len(strings_in[i]) if len(strings_in[i]) <= seq_len else seq_len
        if len(strings_in[i]) <= seq_len-1:
            l = len(strings_in[i])
        else:
            l = seq_len
        for j in range(l):
            real[i][j][P.letterToIndex(strings_in[i][j])] = 1
        for j in range(l, seq_len):
            real[i][j][CHARMAP_LEN-1] = 1
    return real

def get_interpolate(real, fake):
    real = torch.autograd.Variable(real.data).data
    fake = torch.autograd.Variable(fake.data).data
    alpha = torch.rand(real.size()[0], 1, 1).to(device)
    alpha = alpha.expand(real.size())
    return torch.autograd.Variable(alpha * real + (1 - alpha) * fake).data