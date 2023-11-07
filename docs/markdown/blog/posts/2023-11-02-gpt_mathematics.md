---
date: 2023-11-02
title: GPT knows arithmetics
authors: [andrey]
categories:
     - NLP
tags:
     - transformer
     - PyTorch
     - GPT
---

# **GPT learns mathematics**

In this notebook, we look at transformer models! Instead of using **huggingface**, we can turn to **PyTorch** and implement our own variation of a **generative transformer model**. We'll create a model from scratch, which we will teach how to do basic arithmetics. To do this, we'll need to create our own dataset of mathematical operations & train the **GPT model** from scratch! We might want to do this in order to get an indea of how powerful these generative models are, they are able to learn the combinations and help us when needed. Another reason is of course the need to understand how these models are structured inside.

![](images/transformer_id.jpg)

<!-- more -->

![](https://img.shields.io/badge/status-wip-blue) [![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)]()

## **Background**

### Generative Models

The combination of **transformers** & **generative task models** is one of the most useful and applicable to everyday life models (not to even mention exciting :)). As a minimum, we can teach a model to remember text & when necessary generate what text on related topics on which we trained the model. Of course, there are more interesting abilities the model can learn, however, in this post we'll limit ourselves to mathamatics! We'll teach our **GPT model** some basic mathematics such as addition, subtraction, multiplication & division!


## The Dataset

The dataset is generated using a loop, we'll use python to generate this dataset 

```python
n = 1000
strlen = len(f'{n - 1} + {n - 1} = {n * n - 2}')

text = set()
for i in range(n):
    for j in range(n):

        # addition
        example = f'{i} + {j} = {i + j}'
        example += ' ' * (strlen - len(example))
        text.add(example)
        
        # subtraction
        example = f'{i} - {j} = {i - j}'
        example += ' ' * (strlen - len(example))
        text.add(example)
        
        # multiplication
        example = f'{i} * {j} = {i * j}'
        example += ' ' * (strlen - len(example))
        text.add(example)
        
        # module
        if j:
            example = f'{i} / {j} = {i // j}'
            example += ' ' * (strlen - len(example))
            text.add(example)
```

```python
text = list(text)
text[-10:]
```

```
['55 - 256 = -201   ',
 '765 - 822 = -57   ',
 '899 - 295 = 604   ',
 '775 / 692 = 1     ',
 '301 - 797 = -496  ',
 '322 * 711 = 228942',
 '383 * 169 = 64727 ',
 '441 * 430 = 189630',
 '240 + 584 = 824   ',
 '599 + 24 = 623    ']
```


```python
 class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

```python
class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

```python
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
#             logits = logits[:, 8:, :]  # <-----
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C) # view
#             targets = targets[:, 8:]  # <-----
            targets = targets.reshape(B*T)  # view
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
#             idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx_next = torch.argmax(probs, axis=1).reshape(1, -1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```
