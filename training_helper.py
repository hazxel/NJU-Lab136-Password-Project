def get_train_dis(strings_in, pred, seq_len):
    real, real_sub = get_real(strings_in, seq_len)
    fake = get_fake(real, pred, seq_len)
    return real_sub, fake

def get_train_gen(strings_in, pred, seq_len):
    real, _ = get_real(strings_in, seq_len)
    fake = get_fake(real, pred, seq_len)
    return fake

def get_fake(real, pred, seq_len):
    train_pred = torch.zeros(0, seq_len+1, CHARMAP_LEN).to(device)
    real = real.to(torch.float)
    for i in range(1, seq_len+1):
        train_pred = torch.cat((
            train_pred,
            torch.cat((
                torch.zeros(BATCH_SIZE, seq_len-i, CHARMAP_LEN).to(device),
                real[:,:i,:],
                pred[:,i-1:i,:]
                ),dim = 1)
            ),dim = 0)
    return train_pred

def get_real(strings_in, seq_len):
    real = torch.LongTensor(BATCH_SIZE, seq_len+1, CHARMAP_LEN).zero_().to(device)
    real_sub = torch.FloatTensor(0, seq_len+1, CHARMAP_LEN).zero_().to(device)
    for i in range(BATCH_SIZE):
        # l = len(strings_in[i]) if len(strings_in[i]) <= seq_len else seq_len
        if len(strings_in[i]) <= seq_len:
            l = len(strings_in[i])
            real[i][l][CHARMAP_LEN - 1] = 1
        else:
            l = seq_len
        for j in range(l):
            real[i][j][P.letterToIndex(strings_in[i][j])] = 1
        
    for i in range(1, seq_len+1):
        real_sub = torch.cat((
            real_sub,
            torch.cat((
                torch.zeros(BATCH_SIZE, seq_len-i, CHARMAP_LEN).to(device),
                real[:,:i+1,:].to(torch.float)
            ), dim = 1)
        ), dim = 0)
    return real, real_sub

def get_interpolate(real, fake):
    real = torch.autograd.Variable(real.data).data
    fake = torch.autograd.Variable(fake.data).data
    alpha = torch.rand(real.size()[0], 1, 1).to(device)
    alpha = alpha.expand(real.size())
    return torch.autograd.Variable(alpha * real + (1 - alpha) * fake).data