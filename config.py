import torch

USE_CUDA = True

#PRE_GEN_ITERS = 100 # no pre gen now
ITERS_PER_SEQ_LEN = 20000
CRITIC_ITERS = 5
GEN_ITERS = 25

BATCH_SIZE = 16

CHARMAP_LEN = 80 #EOS included (52 + 10 + 17 + 1 = 80)

MAX_LEN = 18
MIN_LEN = 4

DISC_NEURON_SIZE = 512
DISC_HIDDEN_SIZE = 512
GEN_NEURON_SIZE = 512
GEN_HIDDEN_SIZE = 512

LAMBDA = 10

SAVE_CHECKPOINTS_EVERY = 200
CHECKPOINT_PATH = './network_checkpoint/gan_checkpoint.pt'

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")