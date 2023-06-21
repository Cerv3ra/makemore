import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#-------------

torch.manual_seed(1337)

#wget ..
with open('input.txt', 'r' , encoding='utf-8') as f:
    text = f.read()

#here are all the unique chars for the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create mapping from integer to chars and back
stoi = {ch:i for i,ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encoder: take string, output integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder, take list of integer and output string, no tokenizer needed

#train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9+len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):