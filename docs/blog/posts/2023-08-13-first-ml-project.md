---
title: My first Machine Learning Project
date: 2023-08-8 12:00:00 +0000
categories: [Binary Classification]
tags: [sklearn,binary classification, beginner]
image:
  path: https://i.imgur.com/Y0jqbcJ.jpg
  alt: Prediction of Survival on the Titanic
---

> In this post we'll go through a machine learning project from start to finish, If you have any questions, contact me  <sub><a href="https://t.me/mldsai_info"><img src="https://img.shields.io/static/v1?&message=Telegram&color=84BF95&logo=Telegram&logoColor=FFFFFF&label=" /></a></sub>  or write a reply below!
{: .prompt-tip }

## **Background**

A rather interesting classification project I've never actually tried doing. I believe one of my first machine learning projects was **[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**, which was quite a difficult endevour, but quite a lot of fun.

In this post, I'll be walking through anyone new to machine learning, hopefully it will be helpful and an interesting read! 

## **Read Dataset**

Let's read the dataset, it is separated into `train` & `test` set, which we will combine together

```python
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/train.csv")
train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
```

|    |   PassengerId |   Survived |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |    Fare | Cabin   | Embarked   |
|---:|--------------:|-----------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|--------:|:--------|:-----------|
|  0 |             1 |          0 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25   | nan     | S          |
|  1 |             2 |          1 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.2833 | C85     | C          |
|  2 |             3 |          1 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.925  | nan     | S          |
|  3 |             4 |          1 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.1    | C123    | S          |
|  4 |             5 |          0 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.05   | nan     | S          |

## **Exploratory Data Analysis**

### Payment relation to Survival

```python
g = sns.displot(train, x="Fare", hue="Survived", col='Sex',
                element="step",kind = 'hist', palette = "mako")
```

![](https://i.imgur.com/cFBfAmt.png){: .left width="300" } The cost of ticket `Fare` can seem like insightful feature as we can see from the right figure. We can note that higher ticket prices tend to be associated with higher survival, however if we check the division by gender, we can see that this wasn't really the case for males, and that most of this data comes from women. Nevertheless, lets keep all these features.

<br><br>

![](https://i.imgur.com/tFrzvhD.png)

### Embarkment relation to Survival

```python
g = sns.displot(train, x="Fare", 
                hue="Survived", 
                col='Embarked',element="step",
                kind = 'hist', palette = "mako")
```

![](https://i.imgur.com/EeoGYvA.png){: .right width="300" } **Embarkment** location is an interesting feature, we have three different embarkment locations & as we can see from the barplot, it is an insightful feature as we have distribution variation for both target subset classes. From the figure below, we can see that a large portion of passengers that were in `C` class had higher price tickets, all in all, the features shown in the figure have 

<br><br>

![](https://i.imgur.com/EH4yxbf.png)

### Checking for missing data

Datasets often contain some missing data, let's impute each column by it respective mean value 

```
PassengerId       0
Survived          0
Pclass            0
Name              0
Sex               0
Age             354
SibSp             0
Parch             0
Ticket            0
Fare              0
Cabin          1374
Embarked          4
dtype: int64
```

```python
dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)
dataset["Age"].fillna(dataset['Age'].mean(), inplace=True)
```


## **Training Model**

Today we'll train four different machine learning models:

#### RandomForest

`RandomForest` is a machine learning algorithm used for classification, regression, and other tasks. It is an ensemble learning approach that **combines multiple decision trees** to create a more robust and accurate model. Each `DecisionTree` in the `RandomForest` is **built using a random subset of the training data** (Bagging) and a **random subset of the features** (RSS). The final prediction is made by aggregating the predictions of all the `DecisionTrees` in the forest

> class sklearn.ensemble.**RandomForestClassifier**(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
{: .prompt-info }

#### ExtraTrees

The main difference between **Random Forest** and **Extra Trees** classifier is the way they select the features and split the nodes in the decision trees:

- In `RandomForest`, a subset of features is randomly selected for each split, and the best feature is chosen from that subset based on a criterion (eg. `gini`).
- In `ExtraTrees` classifier, **all features are randomly selected for each split**, and the threshold for splitting the nodes is also chosen randomly.

This results in a higher degree of randomness and variability in the decision trees, which can help to **reduce overfitting** and improve the generalization of the model. However, this randomness also means that `ExtraTrees` classifier may require more trees to achieve the same level of accuracy as Random Forest

The model arguments:

> class sklearn.ensemble.**ExtraTreesClassifier**(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
{: .prompt-info }

#### GradientBoosting

> class sklearn.ensemble.GradientBoostingClassifier(*, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
{: .prompt-info }

> class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
{: .prompt-info }

### Validation Curve

One way of determining 

### Hyperparameter tuning using Grid Search 

**GridsearchCV** is used to find the optimal hyperparameters for a machine learning model. Hyperparameters are parameters that are set before training a model and cannot be learned from the data. They include things like the learning rate, number of hidden layers in a neural network, and the regularization strength. For the models we will use, we have specified the full set of hyperparameters which are available for each model.

It automates the process of trying different combinations of hyperparameters and selecting the best one based on a predefined scoring metric. This saves time and effort compared to manually trying different hyperparameter combinations. 

Let's select a few combinations of hyperparameters for each model:

#### RandomForest

In this section, we'll be training our enseble model that often performs very well

```python
# store data
predictions = []

# model
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {
    "max_depth": np.arange(5, 11, 1),
    "max_features": np.arange(8, 11, 1),
    "min_samples_split": np.arange(2, 7, 1),
    "min_samples_leaf": np.arange(1, 5, 1),
    "bootstrap": [False],
    "n_estimators": [170],
    "criterion": ["gini"]
}

# GridSearch Wrapper
gsRFC = GridSearchCV(RFC,
                     param_grid=rf_param_grid,
                     cv=kfold,
                     scoring="accuracy",
                      n_jobs= 4,
                     verbose = 0)

gsRFC.fit(X_train, y_train)
RFC_best = gsRFC.best_estimator_

# Best score
print("score = ", gsRFC.best_score_)
print(gsRFC.best_params_)
```
```
# RandomForest 
score =  0.8473456678154664
{'bootstrap': False, 'criterion': 'gini', 'max_depth': 9, 'max_features': 8, 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 170}
```

#### ExtraTrees

```python

# ExtraTrees Model
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [6, 7, 8, 10],
              "max_features": [7, 8, 9, 10, 11],
              "min_samples_split": [4, 5, 6],
              "min_samples_leaf": [1, 2],
              "bootstrap": [False],
              "n_estimators" :[150],
              "criterion": ["gini"]}

# GridSearch Wrapper
gsExtC = GridSearchCV(ExtC,
                      param_grid = ex_param_grid, 
                      cv=kfold, 
                      scoring="accuracy",  
                      n_jobs= 4, 
                      verbose = 0)

gsExtC.fit(X_train,y_train)

# Best score
ExtC_best = gsExtC.best_estimator_
gsExtC.best_score_
```

```
ExtraTrees
score =  0.8428713948848178
{'bootstrap': False, 'criterion': 'gini', 'max_depth': 10, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 150}
```

#### GradientBoostClassifier

```python
# Gradient Boosting Model
GBC = GradientBoostingClassifier()

gb_param_grid = {
    'learning_rate': np.arange(0.010, 0.04, 0.004),
    'max_depth': np.arange(11, 13, 1),
    'min_samples_leaf': [7, 8, 9],
    'min_samples_split': [2, 3],
    'n_estimators': [200]
}

# start_time = time.time()

# GridSearch Wrapper
gsGBC = GridSearchCV(GBC,
                     param_grid=gb_param_grid,
                     cv=kfold,
                     scoring="accuracy",
                     n_jobs= 4,
                     verbose = 0)

gsGBC.fit(X_train, y_train)

# print('time = ', int(time.time()-start_time), ' sec')
GBC_best = gsGBC.best_estimator_

# Best score
print("score = ", gsGBC.best_score_)
print(gsGBC.best_params_)
```

```
score =  0.8417603845456193
{'learning_rate': 0.026000000000000002, 'max_depth': 12, 'min_samples_leaf': 7, 'min_samples_split': 2, 'n_estimators': 200}
```

#### SVD

```python

### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': np.arange(0.0001, 0.004, 0.0003),
                  'C': [900, 1000, 1100, 1200],
                  'tol': [1e-4, 1e-3]
                 }

# GridSearch Wrapper
gsSVMC = GridSearchCV(SVMC,
                      param_grid = svc_param_grid,
                      cv=kfold,
                      scoring="accuracy",
                      n_jobs= 4,
                      verbose = 0)

gsSVMC.fit(X_train, y_train)
SVMC_best = gsSVMC.best_estimator_

# Best score
print("score = ", gsSVMC.best_score_)
print(gsSVMC.best_params_)
```

```
score =  0.8350036277888626
{'C': 900, 'gamma': 0.001, 'kernel': 'rbf', 'tol': 0.0001}
```

### Learning Curve

A **learning curve** in sklearn is a graphical representation of the performance of a machine learning model as the size of the training dataset increases. It is used to evaluate the effectiveness of a model by plotting the accuracy or error rate of the model against the number of training examples. The learning curve helps to identify if a model is overfitting or underfitting by showing how well the model performs on both the training and testing datasets. It can also be used to determine the optimal size of a training dataset for a particular model.

```python
import seaborn as sns;
sns.set_style("whitegrid", {"grid.color": ".2", "grid.linestyle": ":"})
import warnings;warnings.filterwarnings('ignore')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    
    """Generate a simple plot of the test and training learning curve"""
    plt.figure(figsize=(6,4))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    # mean & std values
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                  train_scores_mean + train_scores_std, 
                     alpha=0.1)
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                  test_scores_mean + test_scores_std, 
                     alpha=0.1)
    
    plt.plot(train_sizes,train_scores_mean,marker='o', ls='--',mfc='none',ms=6,mew=2,label="Training score")
    plt.plot(train_sizes,test_scores_mean,marker='o', ls='--',mfc='none',ms=6,mew=2,label="Cross-validation score")

    plt.legend(loc="best")
    sns.despine(top=True,right=True,bottom=True,left=True)
    return plt
```

<table> 
  <tr style="background-color: #ffffff;"> 
    <td><img src="https://i.imgur.com/VQw1w3H.png"></td> 
    <td><img src="https://i.imgur.com/rrxcTnd.png"></td> 
  </tr> 
</table> 


<table> 
  <tr style="background-color: #ffffff;"> 
    <td><img src="https://i.imgur.com/ZRose6f.png"></td> 
    <td><img src="https://i.imgur.com/f048yza.png"></td> 
  </tr> 
</table> 


<br>

***

**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**

