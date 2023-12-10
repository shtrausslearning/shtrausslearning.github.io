---
date: 2023-11-29
title: GPT knows arithmetics
authors: [andrey]
draft: true
categories:
     - NLP
tags:
     - transformer
     - PyTorch
     - GPT
---

# **Transformer Decoder**

In this notebook, we look at transformer models! Instead of using **huggingface**, we can turn to **PyTorch** and implement our own variation of a **generative transformer model**. We'll create a model from scratch, which we will teach how to do basic arithmetics. To do this, we'll need to create our own dataset of mathematical operations & train the **GPT model** from scratch! We might want to do this in order to get an indea of how powerful these generative models are, they are able to learn the combinations and help us when needed. Another reason is of course the need to understand how these models are structured inside.

![](images/transformer_id.jpg)

<!-- more -->

![](https://img.shields.io/badge/status-wip-blue) [![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Colab-yellow?logo=Google&logoColor=FDBA18)]()

<div style="width: 100%; font-family: Trebuchet MS; font-weight: bold;">
    <div style="padding-top: 40%; position: relative; background-color: #000000; border-radius:10px;">
        <div style="background-image: url('https://i.imgur.com/vkdzYt7.png'); background-size: cover; background-position: center; position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.5; border-radius:10px">
        </div>
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;">
            <div style="position: relative; display: table; height: 75%; width: 100%;">
            </div>
            <div style="position: absolute; bottom: 30px; left: 30px;">
                <div style="font-family: Monospace; text-align: left; font-size: 14px; color: #E0E0E0; letter-spacing: .15rem;">
                    Andrey Shtrauss
                    <br>
                    <span style="color: #CCCCCC">
                        November, 2023
                    </span>
                </div>
            </div>
        </div>
    </div>
</div>


## **Background**

### **<span style='color:#686dec'>Generative Models</span>**
