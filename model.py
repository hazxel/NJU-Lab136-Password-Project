import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from config import *
from data_prep import Password as P

class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_neurons, layers = 1, dropout = 0, input_size = CHARMAP_LEN, output_size = 1):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.num_neurons = num_neurons
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.dropout = dropout
        
        self.embedding = nn.Linear(input_size, num_neurons)
        self.gru = nn.GRU(num_neurons, hidden_size, layers, batch_first = True, dropout = dropout)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, seq_len):
        #output, hidden = self.gru(input.view(1,1,-1), self.hidden)
        #output = self.h2o(output)
        input = self.embedding(input)
        hidden = self.initHidden(seq_len)
        output, _ = self.gru(input, hidden) 
        output = self.h2o(output[:,-1,:])
        return output

    def initHidden(self, seq_len):
        return torch.zeros(self.layers, BATCH_SIZE*seq_len, self.hidden_size).to(device)
    
    def requiresGrad(self):
        for p in self.parameters():
            p.requires_grad = True
            
    def requiresNoGrad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def pre_train(self, p, G):
        passwords = random.sample(p.passwords_string, PRE_DISC_ITERS)
        softmax = nn.LogSoftmax(dim=1)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        for real in passwords:
            fake = G.generate_from(real[0])
            fake = P.passwordToPretrainTensor(fake).float()
            fake = self.embedding(fake)
            real = P.passwordToPretrainTensor(real).float()
            real = self.embedding(real)
            hidden = torch.zeros(self.layers, 1, self.hidden_size).to(device)
            
            fake_out, _ = self.gru(fake, hidden) 
            real_out, _ = self.gru(real, hidden)
            fake_loss = self.h2o(fake_out[:,-1,:])[0][0]
            real_loss = -self.h2o(real_out[:,-1,:])[0][0]
            
            loss = fake_loss + real_loss
            loss.backward()
            optimizer.step()
            
    def test(self, password):
        tensor = P.passwordToPretrainTensor(password).float()
        tensor = self.embedding(tensor)
        hidden = torch.zeros(self.layers, 1, self.hidden_size).to(device)
        out, _ = self.gru(tensor, hidden)
        loss = self.h2o(out[:,-1,:])
        return loss
        
    
class Discriminator_CNN(nn.Module):
    def __init__(self, num_neurons, res_layers = 5, dropout = 0, max_len = 18, input_size = CHARMAP_LEN, output_size = 1):
        super(Discriminator_CNN, self).__init__()
        self.i2c = nn.Conv1d(input_size, num_neurons, 1)
        self.res = []
        self.c2o = nn.Linear(max_len * num_neurons, output_size)
        
        for i in range(res_layers):
            self.res.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv1d(num_neurons, num_neurons, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(num_neurons, num_neurons, 5, padding=2)
                )
            )
    
    def forward(self, input, seq_len):
        output = input.permute(0,2,1)
        
        for r in self.res:
            output = output + 0.3 * r(output)
        output = output.reshape(input.size()[0], -1)
        output = self.c2o(output)
        
        return output
    
    
    
class Generator(nn.Module):
    def __init__(self, hidden_size, num_neurons, layers = 1, dropout = 0, input_size = CHARMAP_LEN, output_size = CHARMAP_LEN):
        super(Generator, self).__init__()
        
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, num_neurons)
        self.gru = nn.GRU(num_neurons, hidden_size, layers, batch_first = True, dropout = dropout)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_len):
        real_ebdd_pack = toTensor(input, seq_len, self.embedding)
        #hidden = self.initHiddenZeros()
        hidden = self.initHiddenRand()
        output, _ = self.gru(real_ebdd_pack, hidden)
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first = True, total_length = seq_len)
        output = self.h2o(output[0])
        output = F.softmax(output,dim=2)
        return output

    def initHiddenZeros(self):
        return torch.zeros(self.layers, BATCH_SIZE, self.hidden_size).to(device)
    
    def initHiddenRand(self):
        return torch.rand(self.layers, BATCH_SIZE, self.hidden_size).to(device)
    
    def generatePassTensor(self, max_length = 18):
        start_letter = p.passwords_string[random.randint(0,len(p.passwords_string) - 1)][0]
        with torch.no_grad():
            input_tensor = P.passwordToInputTensor(start_letter).to(device)
            self.hidden = self.initHiddenZeros()
            password = start_letter

            for c in range(max_length):
                output = self(input_tensor[0])
                output = output.view(1,-1)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == P.n_letters - 1:
                    break
                else:
                    letter = P.all_letters[topi]
                    password += letter
                input_tensor = P.passwordToInputTensor(letter).to(device)
        
        return P.passwordToInputTensor(password)
    
    def generate_from(self, start_letter, max_length = MAX_LEN):
        input_tensor = P.passwordToInputTensor(start_letter)
        with torch.no_grad():
            hidden = torch.rand(self.layers, 1, self.hidden_size).to(device)
            output_password = start_letter

            for c in range(max_length):
                output, hidden = self.gru(self.embedding(input_tensor), hidden)
                output = self.h2o(output)
                output = output.view(1,-1)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == CHARMAP_LEN - 1:
                    break
                else:
                    letter = P.all_letters[topi]
                    output_password += letter
                input_tensor = P.passwordToInputTensor(letter).to(device)
                                
        return output_password
    
    
    def generate_N(self, p, n_generate = 20, max_length = MAX_LEN):
        generate_list = []
        samples = random.sample(p.passwords_string, n_generate)

        for i in range(n_generate):
            start_letter = samples[i][0]
            input_tensor = P.passwordToInputTensor(start_letter)
            with torch.no_grad():
                hidden = torch.rand(self.layers, 1, self.hidden_size).to(device)
                output_password = start_letter

                for c in range(max_length):
                    output, hidden = self.gru(self.embedding(input_tensor), hidden)
                    output = self.h2o(output)
                    output = output.view(1,-1)
                    topv, topi = output.topk(1)
                    topi = topi[0][0]
                    if topi == CHARMAP_LEN - 1:
                        break
                    else:
                        letter = P.all_letters[topi]
                        output_password += letter
                    input_tensor = P.passwordToInputTensor(letter).to(device)
                    
            generate_list.append(output_password)
            
        return generate_list

    def generate_rand_N(self, p, n_generate = 20, max_length = MAX_LEN):
        generate_list = []
        samples = random.sample(p.passwords_string, n_generate)

        for i in range(n_generate):
            start_letter = samples[i][0]
            input_tensor = P.passwordToInputTensor(start_letter)
            with torch.no_grad():
                hidden = torch.rand(self.layers, 1, self.hidden_size).to(device)
                output_password = start_letter

                for c in range(max_length):
                    output, hidden = self.gru(self.embedding(input_tensor), hidden)
                    output = self.h2o(output)
                    output = output.view(-1)
                    output = F.softmax(output,dim=0)
                    output = output.cpu().numpy()    
                    index = np.random.choice(range(len(output)), p=output)
                    if index == CHARMAP_LEN - 1:
                        break
                    else:
                        letter = P.all_letters[index]
                        output_password += letter
                    input_tensor = P.passwordToInputTensor(letter).to(device)
                    
            generate_list.append(output_password)
            
        return generate_list
    
    def pre_train(self, p):
        passwords = random.sample(p.passwords_string, PRE_GEN_ITERS)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        for password in passwords:
            index = [P.letterToIndex(letter) for letter in password]
            index.append(CHARMAP_LEN - 1)
            hidden = torch.rand(self.layers, 1, self.hidden_size).to(device)
            
            self.zero_grad()
            loss = torch.tensor(0, dtype = torch.float32, requires_grad = True, device = device)
            for j in range(len(password)):
                tensor_in = torch.LongTensor(1, 1).zero_().to(device)
                tensor_in[0][0] = index[j]
                tensor_in = self.embedding(tensor_in)
                
                tensor_out, hidden = self.gru(tensor_in, hidden)
                tensor_out = self.h2o(tensor_out)
                
                expected_out = torch.tensor([index[j+1]]).to(device)
                
                loss = loss + criterion(tensor_out[0], expected_out)
                
            loss.backward()
            optimizer.step()
                
    
    '''
    def pre_train(self, input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1)
        self.hidden = self.initHiddenZeros()

        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        self.zero_grad()
        loss = torch.tensor(0, dtype = torch.float32, requires_grad = True, device = device)

        for i in range(input_line_tensor.size(0)):
            output = self(input_line_tensor[i])
            output = output.view(1,-1)
            l = criterion(output, target_line_tensor[i])
            loss = loss + l

        loss.backward()
        optimizer.step()

        return output, loss.item() / input_line_tensor.size(0)
    
    def train(self, D, p, criterion = nn.NLLLoss(), learning_rate = 0.005, max_length = 18):
        start_letter = p.passwords_string[random.randint(0,len(p.passwords_string) - 1)][0]
        start_tensor = P.passwordToInputTensor(start_letter).to(device)
        output_tensor = start_tensor
        self.hidden = self.initHiddenZeros()
       
        output = start_tensor.to(device)
        for c in range(max_length):
            output = self(output[0])
            output_tensor = torch.cat((output_tensor, torch.exp(output)), -3)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == P.n_letters - 1:
                break
            
        output = D.discriminate(output_tensor)
        target_tensor = torch.tensor([1], dtype = torch.long, device = device)
        loss = -output
        logging.debug("gen loss: "+str(loss.item()))
        #loss = criterion(output, target_tensor)
        loss.backward()
        
        for p in self.parameters():
            p.data.add_(-learning_rate, p.grad.data)
    '''
            
def toTensor(strings_in, seq_len, embedding):
        tensor_in = torch.LongTensor(BATCH_SIZE, seq_len+1).zero_().to(device) # +1 because every real pwd has an EOS 
        pwd_len = []
        for i in range(len(strings_in)):
            l = len(strings_in[i]) if len(strings_in[i]) <= seq_len else seq_len
            pwd_len.append(l)
            for j in range(l):
                tensor_in[i][j] = P.letterToIndex(strings_in[i][j])
            tensor_in[i][l] = CHARMAP_LEN - 1
        tensor_in = embedding(tensor_in)
        return nn.utils.rnn.pack_padded_sequence(tensor_in, pwd_len, batch_first=True, enforce_sorted=False)