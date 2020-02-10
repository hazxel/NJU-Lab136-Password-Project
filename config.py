import torch

USE_CUDA = True

PRE_GEN_ITERS = 100
ADVERSIVE_ITERS = 1000
CRITIC_ITERS = 1
GEN_ITERS = 2

BATCH_SIZE = 16

CHARMAP_LEN = 80 #EOS included (52 + 10 + 17 + 1 = 80)

MAX_LEN = 18
MIN_LEN = 4

DISC_NEURON_SIZE = 512
DISC_HIDDEN_SIZE = 512
GEN_NEURON_SIZE = 1024
GEN_HIDDEN_SIZE = 1024

LAMBDA = 10

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")