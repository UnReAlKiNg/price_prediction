import torch

VOCAB_SIZE = 4096
MAX_SEQ_LEN = 8
D_MODEL = 6  #  D_MODEL % N_HEAD == 0, here should match window_size
N_HEAD = 6
D_FF = 2048
NUM_LAYER = 6

NUM_EPOCH = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARM_STEPS = 1000
DEVICE = "mps" if torch.mps.is_available() else "cpu"
