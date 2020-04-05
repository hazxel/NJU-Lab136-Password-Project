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
        self.hidden = torch.zeros(self.layers, BATCH_SIZE, self.hidden_size).to(device)
        self.gru = nn.GRU(num_neurons, hidden_size, layers, batch_first = True, dropout = dropout)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        #output, hidden = self.gru(input.view(1,1,-1), self.hidden)
        #output = self.h2o(output)
        input = self.embedding(input)
        output, _ = self.gru(input, self.hidden) 
        output = self.h2o(output[:,-1,:])
        return output
    
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
        
    
    
    
class Generator(nn.Module):
    def __init__(self, hidden_size, num_neurons, layers = 1, dropout = 0, input_size = CHARMAP_LEN, output_size = CHARMAP_LEN):
        super(Generator, self).__init__()
        
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.dropout = dropout
        
        self.hidden = torch.zeros(self.layers, BATCH_SIZE, self.hidden_size).to(device)
        #self.embedding = nn.Embedding(input_size, num_neurons)
        self.gru = nn.GRU(num_neurons, hidden_size, layers, batch_first = True, dropout = dropout)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, seq_len):
        rand_input = torch.randn([BATCH_SIZE, 1, self.num_neurons]).expand([BATCH_SIZE, seq_len, self.num_neurons]).to(device)
        output, _ = self.gru(rand_input, self.hidden)
        output = self.h2o(output)
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

    def generate_N(self, p, n_generate = 20, max_length = MAX_LEN):
        generate_list = []

        for i in range(n_generate):
            rand_input = torch.randn([1, 1, self.num_neurons]).to(device)
            hidden = self.hidden[:,0:1,:]
            with torch.no_grad():
                output_password = ""
                for c in range(max_length):
                    output, hidden = self.gru(rand_input, hidden)
                    output = self.h2o(output)
                    output = output.view(1,-1)
                    _, topi = output.topk(1)
                    topi = topi[0][0]
                    if topi == CHARMAP_LEN - 1:
                        break
                    else:
                        letter = P.all_letters[topi]
                        output_password += letter
                    
            generate_list.append(output_password)
            
        return generate_list
    
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
