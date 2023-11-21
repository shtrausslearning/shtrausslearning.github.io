---
date: 2023-11-17
title: Prediction of customer stable funds volume
authors: [andrey]
categories:
     - business
tags:
     - regression
--- 

# **Prediction of customer stable funds volume**

Твоей сегодняшней задачей как стажера нашего отдела будет научиться **прогнозировать объем стабильных средств клиентов** без сроков погашения, в данном конкретном случае это **расчетные счета клиентов**.

Почему это важно? Номинально, все средства на расчетных счетах клиенты могут в любой момент забрать из Банка, а в ожидании этого Банк не может их использовать в долгосрочном / среднесрочном плане (например, для выдачи кредитов). Получается, что в такой ситуации Банк ничего не зарабатывает, но платит клиентам проценты по средствам на их счетах, пусть и не высокие, но в масштабах бизнеса Банка эти убытки могут быть значительны.

![](images/sbe3.png)

<!-- more -->

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Colab-yellow?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/12NsWf3ePkrF7bhfTwEfJVVismvZmKXVb?usp=sharing)

## <b>Background</b>

In this project, we aim to help **Gala Groceries** who have approached **Cognizant** to help them with supply chain issues. Specifically, they are interested in wanting to know how better stock the items that they sell"

> Can we accurately predict the stock levels of products based on sales data and sensor data on an hourly basis in order to more intelligently procure products from our suppliers?” 

### Gala Groceries

**Gala Groceries** is a technology-led grocery store chain based in the USA. They rely heavily on new technologies, such as IoT to give them a competitive edge over other grocery stores. 

They pride themselves on providing the best quality, fresh produce from locally sourced suppliers. However, this comes with many challenges to consistently deliver on this objective year-round.

**Gala Groceries** approached Cognizant to **help them with a supply chain issue**. **Groceries are highly perishable items**. **If you overstock, you are wasting money on excessive storage and waste**, but if you **understock, then you risk losing customers**. They want to know how to better stock the items that they sell.

This is a high-level business problem and will require you to dive into the data in order to formulate some questions and recommendations to the client about what else we need in order to answer that question

### The dataset

The client has agreed to share more data in the form of sensor data. They use sensors to measure temperature storage facilities where products are stored in the warehouse, and they also use stock levels within the refrigerators and freezers in store. 
