---
date: 2023-08-22
title: Hyperparameter Tuning with Pipelines
authors: [andrey]
categories:
    - pyspark
tags:
    - pyspark
    - pipeline
    - classification
    - hyperparameter tuning
comments: true
---

# **Hyperparameter Tuning with Pipelines**

***

This post is the last of the three posts on the titanic classification problem in **`pyspark`**

- In the last post, we started with a clearned dataset, which we prepared for machine learning, by utilising **`StringIndexer`** & **`VectorAssembler`**, and then the model training stage itself. 
- These steps are a series of stages in the construction of a model, which we can group into a single **`pipline`**. **`pyspark`** like **`sklearn`** has such pipeline classes that help us keep things organised

***

<!-- more -->

<div class="grid cards" markdown>

  - :simple-google:{ .lg .middle }&nbsp; <b>[Run on Colab](hhttps://colab.research.google.com/drive/12w6EXoyRByFT6q2cmprac1msn7h4aC3o?usp=sharing)</b>

- :simple-github:{ .lg .middle }&nbsp; <b>[Download dataset](https://raw.githubusercontent.com/AlexKbit/stepik-ds-course/master/Week3/spark-practice/train.csv)</b>

</div>

## **Background**

**`pyspark`** pipeline is a tool for **building and executing data processing workflows**. It allows users to chain together multiple data processing tasks into a single workflow, making it easier to organize and manage complex data processing tasks. The pipeline provides a high-level API for building and executing these workflows, which makes it easier for users to quickly prototype and test new data processing pipelines.

Having created a pipeline, we now need to tune the parameters and find the most optimal **`hyperparameter`** combination. This is necessary because the default hyperparameter seldom is most optimal, so we can test different combinations, and store a model, which for example shows the best generalisation performance.

## **Preprocessing Recap**

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

## **Creating Subsets**

The idea of utilising a **`pipeline`** is to process a given set of data using the same preprocessing steps as all input data on which we apply the pipeline on. 
For our problem, our input data will look like the table above, lets first split the available data into two datasets; **`train`** & **`test`**. We will construct
our pipeline on the **`training`** data and then use the same pipeline on the **`test`** dataset.

```python
train,test = ldf.randomSplit([0.8,0.2],42)
```

## **Machine Learning Pipeline**

### Creating a Pipeline

To build a **`pipeline`**, we define the steps that make it up in a list **`stages`**, our pipeline consists of four steps:

- **`indexer_sex`** : **`StringIndexer`**
- **`indexer_embarked`** : **`StringIndexer`**
- **`feature` transformation`** : **`VectorAssembler`**
- **`rf_classifier`** : model which inputs the final result of all transformations above

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler


# Create indexer columns (step 1 & step 2)
indexer_sex = StringIndexer(inputCol='Sex',
                            outputCol='Sex_index')
indexer_embarked = StringIndexer(inputCol='Embarked',
                                 outputCol='Embarked_index')

train_sex = indexer_sex.fit(train).transform(train)
train_embarked = indexer_embarked.fit(train_sex).transform(train_sex)

# Assemble all features into a single column using VectorAssembler (step 3)
feature = VectorAssembler(inputCols=['Pclass','Age','SibSp','Parch','Fare',
                                     'Family_Size','Embarked_index','Sex_index'],
                          outputCol='features')

result = feature.transform(train_embarked)

# Define model (step 4)
rf_classifier = RandomForestClassifier(labelCol='Survived',
                                       featuresCol='features')

# merge all pipeline components
pipeline = Pipeline(stages=[indexer_sex,
                            indexer_embarked,
                            feature,
                            rf_classifier])
# Pipeline_67a25ab8838b
```

Let's use the **`pipeline`** model on the training data & save it so we can use it on new data. Our pipeline model **`pipeline`** stores in itself information about what transformations we need to apply to any incoming data, so we won't need to call **`pipeline`** components (eg. **`indexer_embarked`**) if we want to use the model later.

```python
p_model = pipeline.fit(train)
p_model.save('p_model')
```

Now let's load and use the model on new data (the **`test`** data, which we haven't touched yet). The **`pipline`** model can be loaded in the same way we load standard model, which had the ability to **`.load`**; **`PipelineModel`**

```python
from pyspark.ml.pipeline import PipelineModel

# load pipeline model
model = PipelineModel.load('p_model')
```

```python
# transform model on new data
yv_pred = model.transform(test)
yv_pred.select(['features','rawPrediction','probability','prediction']).show(5)

# +--------------------+--------------------+--------------------+----------+
# |            features|       rawPrediction|         probability|prediction|
# +--------------------+--------------------+--------------------+----------+
# |(8,[0,1,4,7],[3.0...|[8.61510797966188...|[0.43075539898309...|       1.0|
# |(8,[0,1,4],[1.0,5...|[15.1208402070771...|[0.75604201035385...|       0.0|
# |[3.0,27.0,0.0,2.0...|[7.50139928738481...|[0.37506996436924...|       1.0|
# |[3.0,39.0,1.0,5.0...|[16.7158036061856...|[0.83579018030928...|       0.0|
# |[3.0,29.7,0.0,0.0...|[5.75011597218139...|[0.28750579860906...|       1.0|
# +--------------------+--------------------+--------------------+----------+
```

Now let's evaluate the model prediction using **`MulticlassClassificationEvaluator`**

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# evaluator
evaluator = MulticlassClassificationEvaluator(labelCol='Survived',
                                              predictionCol='prediction',
                                              metricName='accuracy')
evaluator.evaluate(yv_pred)
# 0.8275862068965517
```

### Hyperparameter Tuning

Our pipeline contains a model **`RandomForestClassifier`** with default hyperparameters. 
We can can find a better combination of hyperparameters which will give a better model utilising **`TrainValidationSplit`**, which is a lightweight generalisation evaluator in comparison to **`CrossValidator`**

First thigs first, we need to create a parametric grid, which will store all the hyperaparameter combinations we will test.

```python
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# create a parameter grid
grid = ParamGridBuilder().addGrid(rf_classifier.maxDepth,[4,5,6])\
                         .addGrid(rf_classifier.maxBins,[3,4,5])\
                         .addGrid(rf_classifier.minInfoGain,[0.05,0.1,0.15]).build() 

print(len(grid),'combinations')
# 27 combinations*
```

**`TrainValidationSplit`** requires an estimator; which is our **`pipeline`** model, the defined grid, evaluator and train-test split ratio

```python
# Similar to CrossValidator, but more lightweight
gs = TrainValidationSplit(estimator=pipeline,
                          estimatorParamMaps=grid,
                          evaluator=evaluator,
                          trainRatio=0.8)

model = gs.fit(train)
```

To get the best model, we simply write model.bestModel, which has hyperparameters, which can be extracted as per below:

```python
javobj = model.bestModel.stages[-1]._java_obj
print('maxdepth',javobj.getMaxDepth())
print('maxdepth',javobj.getMaxBins())
print('min_IG',javobj.getMinInfoGain())

# maxdepth 5
# maxdepth 5
# min_IG 0.05
```

## **Conclusion**

Important imports for **`pipeline`**

- **`pyspark.ml import Pipeline`** (create model)
- **`pyspark.ml.pipeline import PipelineModel`** (load model)

Important imports for finetuning

- **`pyspark.ml.tuning import ParamGridBuilder`** (Create a parameter grid)
- **`pyspark.ml.tuning import TrainValidationSplit`** (create an evaluator)

***

**❯❯** **`pipeline`** allows us to group together preprocessing steps in one model

- Merge all **`pipeline`** components together

  - ❯ **`Pipeline(stages=[steps1,steps2])`**
    
- **`Pipeline`** then needs to be fit on training data

  - ❯ **`pipeline.fit(train)`** and saved via **`pipeline.save('path')`**
    
- To **load the model**;

  - Load a **`pipeline`** model; from **`PipelineModel`**, which has the method **`load(path)`**
    
- To utilise the **`pipeline`** model on new data;
  
  - **`pipeline.transform(test)`** <br>

***

**❯❯** When we want to **`finetune`** our machine learning model, which is part of a **`pipeline`**:

- Create a parameter grid using **`pyspark.ml.tuning`** **ParamGridBuilder**

  - **`ParamGridBuilder().addGrid(model.maxDepth,params)`** <br>

- From **`pyspark.ml.tuning`** import **`TrainValidationSplit`** evaluator

  - **`TrainValidationSplit(estimator,estimatorParamMaps,evaluator,trainRatio)`**
  - Which the requires a fit **`evaluator.fit(data)`** <br>
 
- Having fitted the evaluator:

  - The best model can be obtained from **`model.bestModel`**
  - Its parameters **`model.bestModel.stages[-1]._java_obj`** <br>

***

**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**

