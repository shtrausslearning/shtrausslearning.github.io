---
date: 2024-08-02
title: Heuristic Approach to Customer Segmentation
authors: [andrey]
draft: true
categories:
     - rfm segmentation
tags:
     - financial analysis
     - rfm segmentation
     - internship
comments: true
---

# **Heuristic Approach to Customer Segmentation**

<div style="width: 100%; font-family: Trebuchet MS; font-weight: bold;">
    <div style="padding-top: 40%; position: relative; background-color: #000000; border-radius:10px;">
        <div style="background-image: url('images/banner_rfm.jpeg''); background-size: cover; background-position: center; position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 1.0; border-radius:10px">
        </div>
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;">
            <div style="position: relative; display: table; height: 75%; width: 100%;">
            </div>
            <div style="position: absolute; bottom: 30px; left: 30px;">
            </div>
        </div>
    </div>
</div>

In today's post we will discuss heuristic segmentation, on a dataset of transactions provided as part of the [quantium internship](https://www.theforage.com/simulations/quantium/data-analytics-rqkb)

<div class="grid cards" markdown>

- :simple-kaggle:{ .lg .middle }&nbsp; <b>[Kaggle Dataset](https://www.kaggle.com/datasets/shtrausslearning/forage-internship-data)</b>

- :fontawesome-regular-rectangle-list:{ .lg .middle }&nbsp; <b>[Quantium Internship](https://www.theforage.com/simulations/commonwealth-bank/intro-data-science-sd7t)</b>

</div>

<!-- more -->

## **Background**

**Heuristic segmentation** aims to divide users/customers into different groups based on some empirical rules or models that have been deduced based on experience working with the group.

The positives of such methods are that they are quick to realise, interpretable 

## <b>Recency Frequency & Monetary Analysis</b>

One of the heuristic methods is the Recency, Frequency & Monetary (RFM) analysis. Its advantage over the previously mentioned approaches is that it takes into account the factor of time or **recency**. 


## <b>Quantium Dataset</b>

Lets explore our dataset, to understand the dataset that we are going to be working with, and figure out which columns we'll need for our RFM analysis.

```
+---+-------+-----------+----------------+--------+----------+------------------------------------------+----------+-----------+
|   | DATE  | STORE_NBR | LYLTY_CARD_NBR | TXN_ID | PROD_NBR |                PROD_NAME                 | PROD_QTY | TOT_SALES |
+---+-------+-----------+----------------+--------+----------+------------------------------------------+----------+-----------+
| 0 | 43390 |     1     |      1000      |   1    |    5     |  Natural Chip        Compny SeaSalt175g  |   2.0    |    6.0    |
| 1 | 43599 |     1     |      1307      |  348   |    66    |         CCs Nacho Cheese    175g         |   3.0    |    6.3    |
| 2 | 43605 |     1     |      1343      |  383   |    61    |  Smiths Crinkle Cut  Chips Chicken 170g  |   2.0    |    2.9    |
| 3 | 43329 |     2     |      2373      |  974   |    69    |  Smiths Chip Thinly  S/Cream&Onion 175g  |   5.0    |   15.0    |
| 4 | 43330 |     2     |      2426      |  1038  |   108    | Kettle Tortilla ChpsHny&Jlpno Chili 150g |   3.0    |   13.8    |
+---+-------+-----------+----------------+--------+----------+------------------------------------------+----------+-----------+
```

Lets take some quick notes about the data that we have, we probably don't need anything else:

!!! note

     - **DATE** : Date since 1899-12-30
     - **STORE_NBR** : The store identifier
     - **LYLTY_CARD_NBR** : Customer's loyalty identifier
     - **PROD_NAME** : Name of the product purchased
     - **PROD_QTY** : Products of type purchased
     - **TOT_SALES** : Sum of purchase


## **Loading Dataset**

Time to load our dataset, having given a glimpse of the data, we can define our data types, similar to how you would do in SQL. 


```python
import pandas as pd

# define data types
dtypes = {'DATE': int,
          'STORE_NBR':int,
          'LYLTY_CARD_NBR':int,
          'TXN_ID':int,
          'PROD_NBR':int,
          'PROD_NAME':str,
          'PROD_QTY':int,
          'TOT_SALES':float
          }

df = pd.read_csv('QVI_transaction_data.csv',dtype=dtypes)
```

As we can see in the data below, we have a customer identifier column ==LYLTY_CARD_NBR==, which we will need to do group by operations and determine aggregations for each unique customer that has made a purchase in our transactions dataset.

Our date column is in a rather odd format, what it represents is the number of days since "1899-12-30", so lets convert it to something we are more familiar with (datetime). However since it is a difference, we need it to be in the **time delta** format & not **datetime**, so lets use `pd.to_timedelta` setting the unit to days:

```python
# Convert days since "1899-12-30" to datetime
start_date = pd.to_datetime("1899-12-30")
df['DATETIME'] = start_date + pd.to_timedelta(df['DATE'], unit='d')
```

Lets also determine the first and last transaction date

```python
df['DATETIME'].min(),df['DATETIME'].max()
```

Looks like we have about a years worth of transactional data, the latest date being "2019-06-30", which is a little out of date for an RFM analysis. Let's assume that we received this data on "2019-07-01", and were asked to conduct the analysis

```
(Timestamp('2018-07-01 00:00:00'), Timestamp('2019-06-30 00:00:00'))
```


## **RFM Process**

### Determine the number of days since last transaction

First things first, we need to determine, for each customer when their last purchase occured, so we will need to determine for each transaction, when it last occured. We have also decided that the current date is "2019-07-01". 

To conduct subtraction, we need to create a datetime timestamp for this date, to do this we use the method `pd.to_datetime()` & to get the number of days we utilise `dt`

```python
today = pd.to_datetime('2019-07-01')

df['DT_LAST_TRANSACTION'] = (today - df['DATETIME']).dt.days
```

###