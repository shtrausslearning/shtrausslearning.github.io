---
date: 2024-04-07
title: Uplift Modeling
authors: [andrey]
draft: false
categories:
     - Uplift Modeling
tags:
     - Machine Learning
comments: true
---

# **Uplift Modeling**

<div style="width: 100%; font-family: Trebuchet MS; font-weight: bold;">
    <div style="padding-top: 40%; position: relative; background-color: #000000; border-radius:10px;">
        <div style="background-image: url('https://i.imgur.com/3l6a5To.png'); background-size: cover; background-position: center; position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.5; border-radius:10px">
        </div>
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;">
            <div style="position: relative; display: table; height: 75%; width: 100%;">
            </div>
            <div style="position: absolute; bottom: 30px; left: 30px;">
            </div>
        </div>
    </div>
</div>

In today's post we'll go through a predictive modeling technique known as **Uplift Modeling**. It is a technique that allows us to identify the subset of customers who are likely to be positively influenced by a marketing plot. The goal of such modeling is to allow us to target the customers who are most likely to respond favourably to a marketing campaign, rather than targeting customers indiscriminately.

<!-- more -->

[![GitHub](https://img.shields.io/badge/Github-Repository-97c446?logo=Github&logoColor=DAF7A6)](https://github.com/shtrausslearning/postgreSQL/blob/main/testing_problem.ipynb)

### **<span style='color:#686dec'> Uplift Modeling</span>**

So what it this modeling approach about:

**Uplift modeling** is a technique that allows us to identify the subset of objects who upon being influences by an event/action will do some action, and if not influenced will not do the same acton

So lets think of an example:

- We are selling a product and need to decide to whom we will be advertising, given that we cannot show it to all target audiences, we would like to find clients who will buy the product, if they see our advertisement and not buy it if they don't see it


In **uplift modeling** we need three components:

 - Have two arrays we will be working with; **Treatment Array**, **Target Array** and standard **customer related feature matrix**
 - The **treatment array** is a binary vector, where we have no influence (0) and influenced (1)
 - The **target vector** is also a binary vector, where we have no action (0) and action is made (1)
 - The standard feature matrix (like other machine learning problems)

### **<span style='color:#686dec'> Dataset</span>**

Lets introduce ourselves to the dataset we will be using in our notebook, by looking at the description provided with the dataset

> This dataset contains 64,000 customers who last purchased within twelve months.
> The customers were involved in an e-mail test.
>
> * 1/3 were randomly chosen to receive an e-mail campaign featuring Mens merchandise.
> * 1/3 were randomly chosen to receive an e-mail campaign featuring Womens merchandise.
> * 1/3 were randomly chosen to not receive an e-mail campaign.
>
> During a period of two weeks following the e-mail campaign, results were tracked.
> Your job is to tell the world if the Mens or Womens e-mail campaign was successful.

Having read the above, lets **summarise the important** bits:

- We have 64000 customers who recently made a purchase, for these customers we have a matrix of features relevant to each of these customers
- We randomly send emails to these customers (**treatment array**); we have an array containing a marketing campaign defined subset groupings
- Finally we have a target containing post marketing campaign monitored results (confirmations of whether the email campaign worked or not)

**Feature Matrix**

Let's also look at the feature matrix available to us:


```
+-------+---------------+-------+----+------+---------+------+-------+
|recency|history_segment|history|mens|womens| zip_code|newbie|channel|
+-------+---------------+-------+----+------+---------+------+-------+
|     10| 2) $100 - $200| 142.44|   1|     0|Surburban|     0|  Phone|
|      6| 3) $200 - $350| 329.08|   1|     1|    Rural|     1|    Web|
|      7| 2) $100 - $200| 180.65|   0|     1|Surburban|     1|    Web|
|      9| 5) $500 - $750| 675.83|   1|     0|    Rural|     1|    Web|
|      2|   1) $0 - $100|  45.34|   1|     0|    Urban|     0|    Web|
+-------+---------------+-------+----+------+---------+------+-------+
```

**Treatment Array**

The treatment array contains text data which we will need to convert into numerical data

```
t.sample(5)

3947       Mens E-Mail
48105      Mens E-Mail
15614      Mens E-Mail
58595        No E-Mail
21571    Womens E-Mail
```

**Target Array**

The target contains the result of the email marketing campaign influence and is already in numerical format

```
y.sample(5)

19624    0
38660    0
49813    0
23809    0
45957    1
```

### **<span style='color:#686dec'> Preprocessing</span>**

**Problem Simplification**

Lets do a little bit of preprocessing and problem simplification. As we saw in the above data, we have three categories in our treatment vector. Lets simplify it to just a binary case and not differentiate the male and female target cases, ie. marketing email has been sent or not sent. 

```python
t = t.map({'Womens E-Mail':1, 'Mens E-Mail':1, 'No E-Mail':0})
t.head()
```

**Train Test Splitting**

Lets also split the data into training & test subsets, we will need some unseen data to validate our models.

```python
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(X, y, t, test_size=0.3, random_state=42)
```

**Categorical Feature Treatment**

We have to also pay attention to categorical features which are present in our feature matrix, a common and most straightforward approach is to use **One Hot Encoding**, which will be applied to three columns. 

We will fit the one hot encoder on our training dataset, and only apply it to the test dataset

```python
cat_columns = ['history_segment', 'zip_code', 'channel']
enc = OneHotEncoder(sparse=False)

X_train_cat = enc.fit_transform(X_train[cat_columns])
X_train_cat = pd.DataFrame(X_train_cat, 
                           index=X_train.index,
                           columns=enc.get_feature_names_out(cat_columns))

X_test_cat = enc.transform(X_test[cat_columns])
X_test_cat = pd.DataFrame(X_test_cat, 
                          index=X_test.index,
                          columns=enc.get_feature_names_out(cat_columns))

X_train = pd.concat([X_train_cat, X_train.drop(cat_columns, axis=1)], axis=1)
X_test = pd.concat([X_test_cat, X_test.drop(cat_columns, axis=1)], axis=1)
```

### **<span style='color:#686dec'> Modeling Approaches</span>**

Now that we have our data ready, lets talk libraries and approaches. There is a commonly used uplift modeling library called **scikit-uplift**, its based on scikit-learn machine learning models, but modified for uplift modeling. Lets remind ourselves of what the modeling actually wants to achieve:

> Uplift modeling focuses on predicting the impact of a treatment or intervention on an individual's behavior

and run over a few modeling approaches:

**s-learner**

Starting with **s-learner** approach, we train two separate models

- We train a base model & apply the model assuming we have interacted with all customers, ie. (t=1 for all customers), and ask to return the probability of a successful outcome (y=1) for this group
- We then repeat the process, but assuming that these has been no interaction with any customer (t=0 for all customers)

The difference between these two vectors will be taken as our uplift, to be more specific:

> model generates **uplift scores** that represent the **estimated impact of a treatment** on each individual's behavior

The **s-learner** model can be used by importing **SoloModel** from `from sklift.models import SoloModel`, we just need to specify the base model we will be using in the two models


```python
name = 'slearner'

base_model = RandomForestClassifier(random_state=42)
uplift_model = SoloModel(base_model)
uplift_model = uplift_model.fit(X_train, y_train, t_train)

model_predictions[name] = uplift_model.predict(X_test)
```



****



### **<span style='color:#686dec'> Concluding Remarks</span>**



***

**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)** or simply below!