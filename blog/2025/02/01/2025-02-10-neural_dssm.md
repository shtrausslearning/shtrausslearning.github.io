---
date: 2025-02-01
title: Neural RecSys
authors: [andrey]
draft: false
categories:
     - sql
tags:
     - recsys
     - neural
     - replay
comments: true
---

# **Neural Networks for Recommendation Systems**

In this notebook we will look at how to use a neural network approach to making recommendations

- The user/item pairings are the main source of data used to create recommendations
- Scalar product of both the **`user_id`** and **`item_id`** embeddings will be our relevancy scores
- User film interactions will be **`positive`** feedback & negative samples which will be created randomly are our **`negative`** samples
- The dataset is split into two, **`train`** will be used to train a model on historical user data, **`test`** will be used to provide user recommendations
- What we will be telling the model is to learn and differentiate between the films they actually watched apart from those they haven’t (ideally)
- We have already looked at **`DSSM`** in a **[previous notebook ](https://shtrausslearning.github.io/notebooks/course_recsys/dssm-towers)** , well be simplifying things a little here, not including user and item features and will keep things more simple.

<!-- more -->

<div class="grid cards" markdown>

- :simple-jupyter:{ .lg .middle }&nbsp; <b>[Jupyter Notebook](https://shtrausslearning.github.io/notebooks/course_recsys/dssm-simple.ipynb)</b>
- :simple-github:{ .lg .middle }&nbsp; <b>[Replay Library](https://developers.sber.ru/portal/products/replay)</b>

</div>


## **Setup**

We have summarised all preprocessing steps before the definition of **`datasets`** and **`dataloaders`**, so the sections associated with preprocessing take up less space 

- **`replay`** will help us keep things more compact, by utilising existing methods for preprocessing
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
```

The parameters which we will be using are as follows, mainly noting that we are using **`rating`** as our feedback column

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

## **1 | Load Dataset**

We will be using a simplified dataset **`MovieLens`** with 100,000 interactions of **`user_id`** with films **`item_id`**

Today, we will be focusing on recommendations using only **interactions**

```python
study = MovieLensPrepare()
```

## **2 | Preprocessing** 

- **`Replay`** contains a handy & quick way for preprocessing **interactions**
- **`MinCountFilter`** can be used for filtering our interactions that have less than **num_entries**
- Lets use this method for removing user interactions with less than 20 items 


```python
study.preprocess()
```

## **3 | Splitting Dataset in time**

- The next step after preprocessing the dataset to our liking is to split it into subsets, so we can train the model on one subset and use another for model validation (20%)
- **replay** has a function named **`TimeSplitter`**, which we will use to create our subsets

> class TimeSplitter(replay.splitters.base_splitter.Splitter)
 |  TimeSplitter(time_threshold: Union[datetime.datetime, str, float], query_column: str = 'query_id', drop_cold_users: bool = False, drop_cold_items: bool = False, item_column: str = 'item_id', timestamp_column: str = 'timestamp', session_id_column: Optional[str] = None, session_id_processing_strategy: str = 'test', time_column_format: str = '%Y-%m-%d %H:%M:%S')

```python
study.split_data()
```

## **4 | Rating Filter**

We want to recommend only items that have been rated highly, so for the **`test`** subset, we will be using **`LowRatingFilter`** to remove iteractions with low ratings

```python
study.filter_test()
```

So what we have going into the next part

- **`study.train`** (training subsett
- **`study.test`** (test subset)

Let's take a look at a sample from the training set

|         |   user_id |   item_id |   rating | timestamp           |
|--------:|----------:|----------:|---------:|:--------------------|
| 1000138 |      5399 |       789 |        4 | 2000-04-25 23:05:32 |
| 1000153 |      5399 |      2162 |        4 | 2000-04-25 23:05:54 |
|  999873 |      5399 |       573 |        5 | 2000-04-25 23:05:54 |
| 1000007 |      5399 |      1756 |        4 | 2000-04-25 23:06:17 |
| 1000192 |      5399 |      1814 |        5 | 2000-04-25 23:06:17 | 


## **5 | Create Torch Dataset**

We need to create a torch dataset from our matrix of interactions **`data`**, which will be passing data to our model

- The dataset **`TowerTrain`** **`get_item`** for each index inputs the **`user_id`** and **`item_id`** (which will be our positive feedback) from the interaction dataset : (positive_item_id)
- Additionally for this user **`user_id`**, we generate an additional number of random **`item_id`** which will be the negative samples, which the user hasn't watched, we’ll be adding 10 to the 1 positive
- Both of these are concatenated into a single array vector (**`items`**)
- Lastly we also return the labels, corresponding to either the **`positive (1)`** or **`negative (0)`** sample id 

```python
from torch.utils.data import Dataset, DataLoader

class TowerTrain(Dataset):
    
    def __init__(self, 
                 data, 
                 num_negatives=10, 
                 i_features=None, 
                 u_features=None):

        # user, item
        self.data = data[[config.USER_COL,config.ITEM_COL]].to_numpy()
        self.num_negatives = num_negatives
        self.num_items = len(np.unique(self.data[:, 1]))
        self.i_features = i_features
        self.u_features = u_features

    def __len__(self):
        return len(self.data)

    # get item of row in data
    def __getitem__(self, idx):

        # index to -> user_id, item_id
        user_id, pos_item_id = self.data[idx, 0], self.data[idx, 1]

        # create positive, negative samples
        # torch tensor for each item_id (pos sample) create 10 neg samples
        items = torch.tensor(np.hstack([pos_item_id,
                                       np.random.randint(
                                           low=0,
                                           high=self.num_items,
                                           size=self.num_negatives)]),
                             dtype=torch.int32)

        # set all labels to 0
        labels = torch.zeros(self.num_negatives + 1, dtype=torch.float32)
        labels[0] = 1. # positive label

        return {'user_ids': torch.tensor([user_id], dtype=torch.int32),
                'item_ids': items,
                'labels': labels}

```

To demonstrate the output of the data class, let’s create the **`dataset`** and subsequent **`dataloader`**, setting a batch size of 2

```python
# create dataset
ds_train = TowerTrain(study.train)

# create data loader
dl_train = DataLoader(ds_train,
                          batch_size=2,
                          shuffle=True,
                          num_workers=0)

batch = next(iter(dl_train))
batch
```

As we can see we get a batch of user identifiers, their array of items and corresponding labels to specify which item is a positive or negative sample

```
{'user_ids': tensor([[ 91],
         [320]], dtype=torch.int32),
 'item_ids': tensor([[ 565, 1534, 1389, 1406, 1346, 1122, 1041,  106, 1147, 1593, 1238],
         [ 317,   96,  113,  638,   47,   73, 1568,  942,  224,  111, 1433]],
        dtype=torch.int32),
 'labels': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
```

## **6 | Model Definition**

We will be creating a subclass **`SimpleTower`**, which only includes the embeddings of both **`user_id`** and **`item_id`** when we’ll define them in the main class

- We can recall that for matrix factorisation approaches, we get the score matrix by using the scalar product of user and item embedding, similarly we will take the same approach to calculate the score for each user/item combination in the row
- The **`forward`** method, when called simply returns the user/item row of the corresponding embedding matrix
- And calculates the dot product between the **`user_id`** & **`item_id`** matrices returning the array for all user/item combinations (batch,11)

```python
# subclass contains only embedding layer but we can 
# expand on this by importing user, item features
class SimpleTower(nn.Module):
    def __init__(self, num_embeddings, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim)

    def forward(self, ids, features=None):
        return self.emb(ids)


class BaseTwoHead(nn.Module):
    
    def __init__(self, 
                 emb_dim, 
                 user_config=None,
                 item_config=None):
        
        super().__init__()
        self.emb_dim = emb_dim
        self.user_tower = SimpleTower(emb_dim=emb_dim, **user_config) # (emb_dim,n_users)
        self.item_tower = SimpleTower(emb_dim=emb_dim, **item_config) # (emb_dim,n_items)

    # forward method defines two 'towers'
    # and the scalar product of the two
    # which will gives us the scores
    def forward(self, batch):
    
        item_emb = self.item_tower(batch["item_ids"]) # (batch,1,16) 
        user_emb = self.user_tower(batch["user_ids"]) # (batch,11,16)
        dot_product = (user_emb * item_emb).sum(dim=-1) # (batch,11)
        return dot_product

    # methods for extracting embeddings
    def infer_users(self, batch):
        return self.user_tower(batch["user_ids"])

    def infer_items(self, batch):
        return self.item_tower(batch["item_ids"])
```

We’ll be defining several dictionaries, which will store the common settings, setting for users and items

- **`embed_config`** : stores the common embedding dimension size **emb_dim**
- **`user_config`** : stores data about the user
- **`item_config`** : stores data about the item

```python
# model parameters
embed_config = {'emb_dim' : 16}  # embedding dimension
user_config = {'num_embeddings' : study.train[config.USER_COL].max() + 1,} # number of users
item_config = {'num_embeddings' : study.train[config.ITEM_COL].max() + 1,} # number of items

# import the embedding dimension 
model = BaseTwoHead(**embed_config, 
                    user_config=user_config, 
                    item_config=item_config)
model
```

```
BaseTwoHead(
  (user_tower): SimpleTower(
    (emb): Embedding(751, 16)
  )
  (item_tower): SimpleTower(
    (emb): Embedding(1616, 16)
  )
)
```

**Model `forward` pass**

- The output of the model will give us the logits for each of the 11 items, for each user row

```python
# output for a single batch
output = model(batch)
output
```

```
tensor([[  1.6632,   5.8888,   0.0997,   7.6885,   8.2156,   4.0495,   3.0272,
           1.9775,  -1.8750,   4.3952,   0.2714],
        [  5.3873, -10.4797,  -4.2230,  -0.4488,   0.9215,  -5.0823,  -0.5018,
           4.9579,   0.8251,  -6.3608,  -4.5723]], grad_fn=<SumBackward1>)
```



## **7 | Preparing Loaders**

We have already previously created a sample dataloder, now let’s create both for the two subsets

```python
# create train dataset
ds_train = TowerTrain(study.test)

# create test data loader
dl_train = DataLoader(ds_train,
                      batch_size=1024,
                      shuffle=True,
                      num_workers=0)

# create test dataset
ds_test = TowerTrain(study.test)

# create test data loader
dl_test = DataLoader(ds_test,
                      batch_size=1024,
                      shuffle=True,
                      num_workers=0)
```

## **8 | Modeling Iteration**

Let’s define the **optimiser** and **loss function** which are pretty much standard across other **binary classification** problems

```python
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()
```

And also the training loop is standard, we’ll be looping through a fixed number of **`epoch`** and passing the batches and predicting, calculating the loss, do a step of **`backpropagation`**, calculating the gradients and updating the model weights via the **`optimiser`**

```python
train_loss_per_epoch = []
test_loss_per_epoch = []

# loop through all epochs
for epoch in tqdm(range(config.NUM_EPOCHS)):

    # training loop for all batches
    model.train()
    train_loss = 0.0
    for iteration, batch in enumerate(dl_train):
        optimizer.zero_grad()
        preds = model(batch)
        loss = loss_fn(preds, batch['labels'])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dl_train)

    # evaluation loop for all batches
    model.eval()
    test_loss = 0
    for iteration, batch in enumerate(dl_test):
        preds = model(batch)
        loss = loss_fn(preds, batch['labels'])
        test_loss += loss.item()

    # evaluation of loss
    test_loss /= len(dl_test)
    test_loss_per_epoch.append(test_loss)
    train_loss_per_epoch.append(train_loss)
```

- So in turn, our model is learning to classify between **`positive`** and **`negative`** samples for each row of data
- Once the model is finished learning, we can utilise the model methods and extract the embeddings from the two towers.
- And save the model as well for future use!

```python
# save our model state
torch.save(model.state_dict(), f"/content/model_{config.NUM_EPOCHS}"
```

## **9 | Generating user recommendations**

Time has come to use our trained model

- We will be making recommendations by using the model that we trained on the **train** dataset and using the **test** users to make predictions
- To make predictions, we will extract the **embedding** matrix weights for user and items, calculate the scores, get the top k results for each user based on the largest score values


### **9.1. Load Weights**

First things first, we need to load the model weights, and put it in inference mode

```python
model = BaseTwoHead(**config, user_config=user_config, item_config=item_config)
model.load_state_dict(torch.load(f"/content/model_{config.NUM_EPOCHS}"))
model.eval()
```

### **9.2. Get test users**

Get the user identifiers that are in the test test, the test set was saved in **`study.test`**

```python
test_users = study.test[[config.USER_COL]].drop_duplicates().reset_index(drop=True)
```

|    |   user_id |
|---:|----------:|
|  0 |         2 |
|  1 |       233 |
|  2 |       736 |
|  3 |        49 |
|  4 |       600 |

### **9.3. Extract Weights**

Extract the embedding weights for all users and items which is located in the model

```
# extract the user / item embedding weights
user_embed = model.user_tower.emb.weight.detach().cpu().numpy()
item_embed = model.item_tower.emb.weight.detach().cpu().numpy()
user_embed.shape, item_embed.shape
```


### **9.4. Scalar product**

Calculate the scores for each user & item combination by calculating the scalar product of them

```python
# calcualate the scores (751,1616)
scores = user_embed[test_users[config.USER_COL].values] @ item_embed.T
```

```
[[-2.219962   -2.8183699  -1.2701275  ... -1.7878596  -2.3029149
  -5.1351438 ]
 [-0.2002018  -3.269224   -3.5974343  ... -5.4825845  -4.0557184
  -4.9202886 ]
 [-0.24603942 -1.9250925  -1.2330636  ... -4.066546   -3.6852539
  -6.3292623 ]
 ...
 [ 1.3434778  -2.2150192  -1.8992031  ... -4.7611713  -4.1526904
  -5.917045  ]
 [ 0.067677   -2.6156569  -2.6362207  ... -3.8871505  -3.1315584
  -3.5736673 ]
 [-1.3127992  -1.5567051  -1.1855109  ... -2.6913378  -3.2935755
  -5.5215263 ]]
```

### **9.5. Get highest scores**

Get the highest value indicies (idx) & their corresponding values (scores). The scores correspond to the index of the item in the **encoder** **`encoder_item`**, which we stored in class instance **`study`**

```python
# get top 10 idx by value & get its value
ids = np.argpartition(scores, -config.K)[:, -config.K:]
scores = np.take_along_axis(scores, ids, axis=1)
scores[:5]
```

```
array([[ 1.3017656 ,  1.4262905 ,  1.4305891 ,  1.5401053 ,  1.5945268 ,
         1.9945638 ,  1.9178314 ,  2.8111196 ,  1.5959901 ,  2.221249  ],
       [-0.02534078,  1.0504715 ,  0.6823742 ,  0.6663627 , -0.00748574,
         0.5298525 ,  0.49601346,  0.32487705,  0.04160966,  0.02862556],
       [ 1.7142106 ,  1.8349895 ,  2.43454   ,  2.896079  ,  3.0631516 ,
         2.1554096 ,  1.8832399 ,  2.087269  ,  3.876807  ,  2.2215443 ],
       [ 0.2731401 ,  0.30537376,  0.3488819 ,  0.53589934,  1.0000901 ,
         0.77159363,  0.6785181 ,  0.7471067 ,  0.55528575,  1.0426229 ],
       [ 0.89288795,  0.92402935,  0.97583646,  0.98947227,  1.0060023 ,
         1.1556187 ,  1.4170016 ,  1.4296795 ,  1.7379148 ,  1.2944818 ]],
      dtype=float32)
```


### **9.6. Recommendations Matrix**

Prepare the usual format, **`user_id`**, **`item_id`** and rating **`rating`**, which will enable us to quickly evaluate the metrics using **`experiment`** function from **replay**. We need to add both lists to each user & expand them together



```python
# prepare recommendations matrix
def prepare_recs(test_users, 
                 rec_item_ids, 
                 rec_relevances):
    
    predict = test_users.copy()
    predict[config.ITEM_COL] = rec_item_ids.tolist()  # add list of indicies for each user
    predict['rating'] = rec_relevances.tolist() # add rating list of scores for each user
    predict = predict.explode(column=[config.ITEM_COL, 'rating']).reset_index(drop=True) # expand both lists
    predict[config.ITEM_COL] = predict[config.ITEM_COL].astype(int)
    predict['rating'] = predict['rating'].astype("double")
    return predict


model_recommendations = prepare_recs(test_users,      # user columns 
                                     rec_item_ids=ids,  # indicies of top 10 in scores
                                     rec_relevances=scores) # scores of top 10
```

|    |   user_id |   item_id |   rating |
|---:|----------:|----------:|---------:|
|  0 |         2 |       302 |  1.30177 |
|  1 |         2 |       218 |  1.42629 |
|  2 |         2 |       233 |  1.43059 |
|  3 |         2 |       139 |  1.54011 |
|  4 |         2 |         6 |  1.59453 |


We'll evaluate the prediction & test overlapping items using **hitrate**, to measure how well the model predicts at least one relevant recommendation for users. **NDCG**, for the evaluation of how well the model can correcly order the relevant items & **coverage** to measure how well the model predicts a range of items from all available items

```python
metrics = Experiment(
    [NDCG(config.K), HitRate(config.K), Coverage(config.K)],
    study.test,
    study.train,
    query_column=config.USER_COL, 
    item_column=config.ITEM_COL,
)
metrics.add_result("dssm_model", model_recommendations)
metrics.results
```

|            |   NDCG@10 |   HitRate@10 |   Coverage@10 |
|:-----------|----------:|-------------:|--------------:|
| dssm_model | 0.0345152 |     0.221053 |      0.191213 |



**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)** or simply below!
