import torch

USE_CUDA = True

PRE_GEN_ITERS = 100 
PRE_DISC_ITERS = 100
ITERS_PER_SEQ_LEN = 5000
CRITIC_ITERS = 3
GEN_ITERS = 10

BATCH_SIZE = 512

CHARMAP_LEN = 80 # EOS included (52 + 10 + 17 + 1 = 80)

EXTRA_LEN = 2
MAX_LEN = 18
MIN_LEN = 4

CNN_NEURON_SIZE = 128
DISC_NEURON_SIZE = 512
DISC_HIDDEN_SIZE = 512
DISC_LAYERS = 4
GEN_NEURON_SIZE = 512
GEN_HIDDEN_SIZE = 512
GEN_LAYERS = 4
DEFAULT_DROPOUT = 0.01

LAMBDA = 10

SAVE_CHECKPOINTS_EVERY = 1000
DEFAULT_CHECKPOINT_PATH = './network_checkpoint/gan_checkpoint.pt'
DEFAULT_SAMPLE_PATH = './output/sample.json'
DEFAULT_TEST_PATH = './data/linkedin_cleaned.json'
DEFAULT_TRAINING_PATH = './data/rockyou_cleaned.json'

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
