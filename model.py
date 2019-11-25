import torch
import torch.nn as nn
import torch.nn.functional as F
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
            
    def calcGradientPenalty(self, real_data, fake_data, LAMBDA = .5):
        if real_data.size()[0] > fake_data.size()[0]:
            fake_data = P.zeroPadding(fake_data.to(torch.device("cpu")), real_data.size(0)).to(device)
        elif real_data.size()[0] < fake_data.size()[0]:
            real_data = P.zeroPadding(real_data.to(torch.device("cpu")), fake_data.size(0)).to(device)
        alpha = torch.rand(1).to(device)
        interpolate = alpha * real_data + (1 - alpha) * fake_data
        interpolate.requires_grad = True

        output = self.discriminate(interpolate)

        grad = torch.ones_like(interpolate).to(device)
        gradients = torch.autograd.grad(
                outputs=output,
                inputs=interpolate,
                grad_outputs=grad,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        # logging.debug(gradients.norm(2, dim=2))
        penalty = ((gradients.norm(2, dim=2) - 1 ) ** 2).mean() * LAMBDA
        return penalty
    
    def train(self, input_tensor, category):
        hidden = self.initHidden()
        self.zero_grad()
        
        for i in range(input_tensor.size()[0]):
            output, hidden = self(input_tensor[i], hidden)
            
        loss = -output * category
        # loss = loss * torch.sigmoid(100*loss)
        loss.backward(retain_graph = True)

        return output, loss.item()
    
    def discriminate(self, input_tensor):
        hidden = self.initHidden()
        for i in range(input_tensor.size()[0]):
            output, hidden = self(input_tensor[i], hidden)
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
        hidden = self.initHiddenZeros()
        output, _ = self.gru(real_ebdd_pack, hidden)
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.h2o(output[0])
        output = F.log_softmax(output,dim=2)
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
    
    def generate_N(self, p, n_generate = 100, max_length = 18):
        generate_list = []

        for i in range(n_generate):
            start_letter = p.passwords_string[random.randint(0,len(p.passwords_string) - 1)][0]
            with torch.no_grad():
                input_tensor = P.passwordToInputTensor(start_letter).to(device)
                self.hidden = self.initHiddenZeros()
                output_password = start_letter

                for c in range(max_length):
                    output = self(input_tensor[0])
                    output = output.view(1,-1)
                    topv, topi = output.topk(1)
                    topi = topi[0][0]
                    if topi == P.n_letters - 1:
                        break
                    else:
                        letter = P.all_letters[topi]
                        output_password += letter
                    input_tensor = P.passwordToInputTensor(letter).to(device)
                    
            generate_list.append(output_password)
            
        return generate_list

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