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

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)]()

## Background

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
