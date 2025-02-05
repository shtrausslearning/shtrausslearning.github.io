---
date: 2023-08-21
title: Creating ML Models with PySpark
authors: [andrey]
categories:
    - PySpark
tags:
    - pyspark
    - classification
    - modeling
comments: true
---

# **Training ML Models with PySpark**

***

In this post, we will introduce ourselves to **`pyspark`**

- We are continuing on from the previous post **[PySpark Titanic Preprocessing](https://shtrausslearning.github.io/posts/spark-preprocess/)**, where we did some basic **data preprocessing**, here we will continue on with the **modeling** stage of our project
- We will be using **`spark.ml.classification`** to train binary classification models
- There are quite a number of differences from **`pandas`**, for example the formulation of a **`VectorAssembler`** columns, which combines all column features into one

***



<!-- more -->

<div class="grid cards" markdown>

  - :simple-google:{ .lg .middle }&nbsp; <b>[Run on Colab](https://colab.research.google.com/drive/12w6EXoyRByFT6q2cmprac1msn7h4aC3o?usp=sharing)</b>

- :simple-github:{ .lg .middle }&nbsp; <b>[Download dataset](https://raw.githubusercontent.com/AlexKbit/stepik-ds-course/master/Week3/spark-practice/train.csv)</b>



</div>

## **Introduction**

We'll continue on where we left of **[PySpark Titanic Preprocessing](https://shtrausslearning.github.io/posts/spark-preprocess/)**

- In the last post, we focused on general preprocessing data, mostly **data cleaning**. 
- In this post, we'll focus on finishing off data preprocessing, transformation steps that a required before passing the data to the model

## **Preprocessing Summary**

Let's summarise our preprocessing stages that we did last post:

- We learned how to drop columns that we won't be needing at all in our preprocessing using **`.drop`**
- We learned how to extract statistical data from our dataframe, using **`.select`** and functions **`f.avg('column')`**
- We known how to fill missing data in different columns using a single value with a dictionary; **`f.fillna({'column':'value'})`**
- We know how to add or replace a column, using **`f.withColumn`**

```python
df = spark.read.csv('train.csv',header=True,inferSchema=True)
df = df.drop('Ticket','Name','Cabin')
av_age = df.select(f.avg(f.col('age')))
df = df.fillna({'age':round(av_age.collect()[0][0],2)})
df = df.withColumn('Family_Size',f.col('SibSp') + f.col('Parch'))  # add values from two columns
df = df.withColumn('Alone',f.lit(0))  # fill all with 0
df = df.withColumn('Alone',f.when(df['Family_size'] == 0,1).otherwise(df['Alone'])) # conditional filling
df = df.drop('any')
df.show()

# +-----------+--------+------+------+----+-----+-----+-------+--------+-----------+-----+
# |PassengerId|Survived|Pclass|   Sex| Age|SibSp|Parch|   Fare|Embarked|Family_Size|Alone|
# +-----------+--------+------+------+----+-----+-----+-------+--------+-----------+-----+
# |          1|       0|     3|  male|22.0|    1|    0|   7.25|       S|          1|    0|
# |          2|       1|     1|female|38.0|    1|    0|71.2833|       C|          1|    0|
# |          3|       1|     3|female|26.0|    0|    0|  7.925|       S|          0|    1|
# |          4|       1|     1|female|35.0|    1|    0|   53.1|       S|          1|    0|
# |          5|       0|     3|  male|35.0|    0|    0|   8.05|       S|          0|    1|
# +-----------+--------+------+------+----+-----+-----+-------+--------+-----------+-----+
```

## **String Indexing**

We have left two columns which contain **categorical (string)** data, with which we want to work with in our modeling process; **`Sex`**,**`Embarked`**. As we saw in an exploratory data analysis from **[a previous post](https://shtrausslearning.github.io/posts/first-ml-project/)**, these two features do contain data distributions, which allow us to distinguish between whether a passenger survived or not, which means it probably would help a model improve its accuracy. However these features will need to be modified in order for us to use them in our model.

- In **`sklearn`** there is a method called **[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)**. 
- In pyspark, there is a method called **StringIndexer**, which work in a similar way.

```python
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

indexers = [StringIndexer(inputCol=column,outputCol=column+'_index').fit(df) for column in ['Sex','Embarked']]
pipeline = Pipeline(stages=indexers)

df = pipeline.fit(df).transform(df)
df = df.drop('PassengerId','Embarked','Sex')
df.show()

# +--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+
# |Survived|Pclass| Age|SibSp|Parch|   Fare|Family_Size|Alone|Sex_index|Embarked_index|
# +--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+
# |       0|     3|22.0|    1|    0|   7.25|          1|    0|      0.0|           0.0|
# |       1|     1|38.0|    1|    0|71.2833|          1|    0|      1.0|           1.0|
# |       1|     3|26.0|    0|    0|  7.925|          0|    1|      1.0|           0.0|
# |       1|     1|35.0|    1|    0|   53.1|          1|    0|      1.0|           0.0|
# |       0|     3|35.0|    0|    0|   8.05|          0|    1|      0.0|           0.0|
# +--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+
```

Once we are done indexing string columns, we need to remove them!

## **Combine Features**

Once we are happy with all the features that we want to utilise in our model, we need to assemble them into a single column. 

- To do so we need to utilise method **`VectorAssembler`**. 
- We need to write the names of the input feature columns we want to use **`inputCols`**
and define the output feature name **`outputCol`**, the resulting feature will be placed in the input dataframe.

```python
from pyspark.ml.feature import VectorAssembler

feature = VectorAssembler(inputCols=df.columns[1:],
                          outputCol='features')
feature_vector = feature.transform(df)
feature_vector.show()

# +--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+--------------------+
# |Survived|Pclass| Age|SibSp|Parch|   Fare|Family_Size|Alone|Sex_index|Embarked_index|            features|
# +--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+--------------------+
# |       0|     3|22.0|    1|    0|   7.25|          1|    0|      0.0|           0.0|[3.0,22.0,1.0,0.0...|
# |       1|     1|38.0|    1|    0|71.2833|          1|    0|      1.0|           1.0|[1.0,38.0,1.0,0.0...|
# |       1|     3|26.0|    0|    0|  7.925|          0|    1|      1.0|           0.0|[3.0,26.0,0.0,0.0...|
# |       1|     1|35.0|    1|    0|   53.1|          1|    0|      1.0|           0.0|[1.0,35.0,1.0,0.0...|
# |       0|     3|35.0|    0|    0|   8.05|          0|    1|      0.0|           0.0|(9,[0,1,4,6],[3.0...|
# +--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+--------------------+
```

## **Train-Test Splitting**

Once our data is ready, we should think of a strategy to confirm the accuracy of our model
 
- Train-Test Splitting is a common strategy to verify how well a model generalises on data it wasn't trained on. In **`spark`**, we can reference to the dataframe itself to split it using **`df.randomSplit`**

```python
(training_data, test_data) = feature_vector.randomSplit([0.8,0.2],42)
```

## **Training & Evaluation**

Training & evaluation of different models follow the same template of actions, the only thing that changes is we load different models from **`spark.ml.classification`**

### :material-numeric-1-box-multiple-outline: LogisticRegression

The first step is to load the relevant model from **`.ml.classification`**, in this case we start with a simplistic LogisticRegression model, which is named the same as in **sklearn**. Inputs into the model instance require us to specify the vectorised feature columns **`featuresCol`** and the target variable column, **`labelCol`**

The model should be **`fit`** on training data and saved into varaible **`lrModel`**, which is a little different to how you would do it in **`sklearn`**

```python
from pyspark.ml.classification import LogisticRegression

# initialise model
lr = LogisticRegression(labelCol='Survived',
                        featuresCol='features')

# returns a transformer which is our model
lrModel = lr.fit(training_data)
```

Variable **`lrModel`** can then be used to make a prediction on the test set, to get its generalisation score on new data, 
we can see which rows of data matches by using **`.select`**

```python
# make prediction on test set
lr_prediction = lrModel.transform(test_data)
lr_prediction.select(['prediction','Survived']).show(5)

# +----------+--------+
# |prediction|Survived|
# +----------+--------+
# |       1.0|       0|
# |       1.0|       0|
# |       1.0|       0|
# |       1.0|       0|
# |       0.0|       0|
# +----------+--------+
```

Finally, having the relevant prediction, we can evaluate the overall performance of the model using **`MulticlassClassificationEvaluator`**

One nuance that may seem odd is that we opted to use **multiclass**, even though our problem is a binary classification problem. 
The reasoning can be explained by **[this post](https://stackoverflow.com/questions/60772315/how-to-evaluate-a-classifier-with-pyspark-2-4-5)**, which states that **`MulticlassClassificationEvaluator`** utilises class weighting

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# evaluator
evaluator = MulticlassClassificationEvaluator(labelCol='Survived',
                                              predictionCol='prediction',
                                              metricName='accuracy')
evaluator.evaluate(lr_prediction)
# 0.7586206896551724
```

### :material-numeric-2-box-multiple-outline: DecisionTree

A powerful binary tree based algorith, which is used by both gradient boosting and random forest:

```python
from pyspark.ml.classification import DecisionTreeClassifier

# initialise model
dt = DecisionTreeClassifier(labelCol='Survived',
                            featuresCol='features')

# returns a transformer which is our model
dtModel = dt.fit(training_data)   
dt_prediction = dtModel.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol='Survived',
                                              predictionCol='prediction',
                                              metricName='accuracy')
evaluator.evaluate(dt_prediction)
# 0.7448275862068966
```

### :material-numeric-3-box-multiple-outline: RandomForest

One ensemble approach based on randomised generation of **`DecisionTrees`** we can try is **`RandomForest`**, which even is named the same as in **`sklearn`**

```python
from pyspark.ml.classification import RandomForestClassifier

# initialise model
rf = RandomForestClassifier(labelCol='Survived',
                            featuresCol='features')

# returns a transformer which is our model
rfModel = rf.fit(training_data)   
rf_prediction = rfModel.transform(test_data)

# evaluator
evaluator = MulticlassClassificationEvaluator(labelCol='Survived',
                                              predictionCol='prediction',
                                              metricName='accuracy')
evaluator.evaluate(rf_prediction)
# 0.7586206896551724
```

### :material-numeric-4-box-multiple-outline: GradientBoosting

Another enseble method which uses **`DecisionTrees`** is Gradient Boosting, its name varies from that of **`sklearn`**

```python
from pyspark.ml.classification import GBTClassifier

# initialise model
gb = GBTClassifier(labelCol='Survived',
                            featuresCol='features')

# returns a transformer which is our model
gbModel = gb.fit(training_data)   
gb_prediction = gbModel.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol='Survived',
                                              predictionCol='prediction',
                                              metricName='accuracy')
evaluator.evaluate(gb_prediction)
# 0.7517241379310344
```

## **Saving & Loading Model**

We have tested different models and found the one which gives us the best metric, which in our case is **`accuracy`**

- To save a model we need to save **`model.fit`**. The best performing model in our case was **RandomForest**, so let's save **`rfModel`**

```python
rfModel.save('rf_model')
```

To load the model, we need to load the relevant module from **`classification`**; **`RandomForestClassificationModel`**, which is different from **`RandomForestClassifier`**, and call the method **`.load('folder')`**

```python
from pyspark.ml.classification import RandomForestClassificationModel

RandomForestClassificationModel.load('rf_model')
# RandomForestClassificationModel: uid=RandomForestClassifier_f17b9c33fe1c, numTrees=20, numClasses=2, numFeatures=9
```

## **Summary**

Let's review what we have covered in this post:

- We learned how to drop columns, using **`.drop`**
- We learned how to extract statistical data from our dataframe, using **`.select`** and functions **`f.avg('column')`**
- We known how to fill missing data in different columns using a single value with a dictionary; **`f.fillna({'column':'value'})`**
- Add or replace a column, using **`f.withColumn`**
- **`StringIndexer(inputCol,outputCol).fit(data)`** - convert categorical into a numerical representation
- Once we are done with our feature matrix, we can convert all the relevant features into a single feature that will be used as input into the model using **`VectorAssembler(inputCols,outputCol).transform(data)`**
- To split the data into a training & validation dataset, we can use the **dataframe** method **`df.randomSplit`**

**Training a model** requires identical steps for whichever model we choose:

- Import the model class from `pyspark.ml.classification`
- Instantiate the model by specifying **`labelCol`** and **`featuresCol`**
- Train the model using **`trained_model = model.fit(data)`**
- Use the model to make predictions using **`y_pred = trained_model.transform(data)`**
- Once we have both a model prediction and training labels, we can make an evaluation using an evaluator **`MulticlassClassificationEvaluator`** with evaluator.evaluate(data)
- And to finish off our modeling state, we can save our model that we will use in production by saving **`trained_model.save('name')`** and load with the relevant **`XModel.load()`**
