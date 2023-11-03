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

The combination of transformers & generative task models is one of the most useful and applicable to everyday life models. We can teach a model to remember text & when necessary generate what text on related topics on which we trained the model


## The Dataset

The dataset is generated using a loop, we'll use python to generate this dataset 
