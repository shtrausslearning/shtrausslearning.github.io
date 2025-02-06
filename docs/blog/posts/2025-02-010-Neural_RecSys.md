---
date: 2025-02-01
title: Neural RecSys
authors: [andrey]
draft: true
categories:
     - sql
tags:
     - recsys
     - neural
comments: true
---

# **Neural Networks for Recommendation Systems**

***

In this notebook we will look at how to use a neural network approach to recommendations

- Implicit feedback will be used
- Scalar product of both the **`user_id`** and **`item_id`** embeddings will be our relevancy scores
- User film interactions will be **`positive`** feedback & negative samples which will be created randomly are our **`negative`** samples
- The dataset is split into two, **`train`** will be used to train a model on historical user data, **`test`** will be used to provide user recommendations
- What we will be telling the model is to learn and differentiate between

***

<!-- more -->

<!-- <div class="grid cards" markdown>

  - :simple-github:{ .lg .middle }&nbsp; <b>[GitHub Repository](https://github.com/shtrausslearning/postgreSQL/blob/main/testing_problem.ipynb)</b>

</div> -->


## **Setup**

We have summarised all preprocessing steps before the definition of **`datasets`** and **`dataloaders`**, so the section will take up a little less space.*

- **`MovieLensPrepare`** : Initialising this class will read the dataset
- **`preprocess`** : Calling this method will define filtration of low item count envents for each user, reset the user and item identifiers and convert the time based feature into something we can work with
- **`split_data`** : Calling this method will create two subsets based on the last 20% of datetime split
- **`filter_test`** : Calling this method will remove all the poorly rated events for each user


```python
class MovieLensPrepare:

    def __init__(self):
        rs = MovieLens('100k')
        self.data = rs.ratings
        self.u_features = rs.users
        self.i_features = rs.items


    def preprocess(self):

        data = self.data
        u_features = self.u_features
        i_features = self.i_features
        
        data = MinCountFilter(num_entries=20).transform(data)

        # interactions and user & item features must be synchronised
        data = data[data[config.USER_COL].isin(u_features[config.USER_COL].unique())]
        data = data[data[config.ITEM_COL].isin(i_features[config.ITEM_COL].unique())]

        print(f"Number of unique users {data['user_id'].nunique()}")
        print(f"Number of unique items {data['item_id'].nunique()}")

        # interactions and user & item features must be synchronised
        data = data[data[config.USER_COL].isin(u_features[config.USER_COL].unique())]
        data = data[data[config.ITEM_COL].isin(i_features[config.ITEM_COL].unique())]

        data[config.TIMESTAMP] = pd.to_datetime(data['timestamp'],unit='s')

        self.data = data

    def split_data(self):

        data = self.data
        u_features = self.u_features
        i_features = self.i_features

        splitter = TimeSplitter(time_threshold=0.2,  # 20% into test subset
                            drop_cold_users=True,
                            drop_cold_items=True,
                            query_column=config.USER_COL)
        
        train,test = splitter.split(data)
        print('train size',train.shape[0])
        print('test size', test.shape[0])

        # user features and item features must be present in interactions dataset and only
        u_features = u_features[u_features[config.USER_COL].isin(train[config.USER_COL].unique())]
        i_features = i_features[i_features[config.ITEM_COL].isin(train[config.ITEM_COL].unique())]

        # encoders for users
        encoder_user = LabelEncoder()
        encoder_user.fit(train[config.USER_COL])
        
        # encoders for items
        encoder_item = LabelEncoder()
        encoder_item.fit(train[config.ITEM_COL])

        train[config.USER_COL] = encoder_user.transform(train[config.USER_COL])
        train[config.ITEM_COL] = encoder_item.transform(train[config.ITEM_COL])
        
        test[config.USER_COL] = encoder_user.transform(test[config.USER_COL])
        test[config.ITEM_COL] = encoder_item.transform(test[config.ITEM_COL])
        
        u_features[config.USER_COL] = encoder_user.transform(u_features[config.USER_COL])
        i_features[config.ITEM_COL] = encoder_item.transform(i_features[config.ITEM_COL])

        self.train = train 
        self.test = test

        self.u_features = u_features
        self.i_features = i_features
        
    def filter_test(self):
        filter_rating = LowRatingFilter(value=4)        
        self.test = filter_rating.transform(self.test)
        
study = MovieLensPrepare()
```







```python
from dataclasses import dataclass

@dataclass
class config:

    USER_COL : str = 'user_id'
    ITEM_COL : str = 'item_id'
    RATING_COL : str = 'rating'
    TIMESTAMP : str = 'timestamp'
    NUM_EPOCHS : int = 30

    K = 10
    SEED = 123

config = config()
random.seed(config.SEED)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
```




**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)** or simply below!