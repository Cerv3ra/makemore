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
    #generate a small data batch (inputs x and targets y)
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size] for i in ix])
    x,y = x.to(device) y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses == torch.zeros(eval_items)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel(vocab_size)
 
