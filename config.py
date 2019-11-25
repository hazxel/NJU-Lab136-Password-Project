import torch

USE_CUDA = True

PRE_GEN_ITERS = 100
ADVERSIVE_ITERS = 500
CRITIC_ITERS = 1

BATCH_SIZE = 2

CHARMAP_LEN = 80

MAX_LEN = 18
MIN_LEN = 4

DISC_NEURON_SIZE = 512
DISC_HIDDEN_SIZE = 512
GEN_NEURON_SIZE = 512
GEN_HIDDEN_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")