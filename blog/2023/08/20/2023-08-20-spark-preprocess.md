---
date: 2023-08-20
title: Data Preprocessing with PySpark
authors: [andrey]
categories:
    - PySpark
tags:
    - pyspark
    - preprocessing
comments: true
---

# **Data Preprocessing with PySpark**

---

In this post, we will introduce ourselves to **`pyspark`**, a framework that allows us to work with big data

- We'll look at how to start a **`spark_session`**
- Setting up data types for the dataset using **`StructType`**
- This post focuses on data preparation in the preprocessing state

---

<!-- more -->

## **Introduction**

In this post, we'll introduce ourselves to **`pyspark`**, working on a commonly used classification problem; the titanic. Our focus will be to learn the basics of how to work with **`pyspark`** when we work on machine learning projects. We'll split this little project into two parts; <kdb>part I</kdb> part will include **data loading**, **data preprocessing** (**feature engineering** and **data cleaning**). Second part, <kdb>part II</kdb> will include data preparation for **machine learning** and subsequent **model training** and **evaluation**


## **Spark over Pandas**

In the **[previous post](https://shtrausslearning.github.io/posts/first-ml-project/)**, we used **`pandas`** for working with **tabular data**. **`pandas`** is indeed quite convenient to use as it has a very rich functionality to work with tabular data, **`pyspark`** in comparison is much simplier, however it offers the user to work with **`big data`**, which **`pandas`** tends to strugle with.

Some advantages of **`pyspark`** over **`pandas`**:

> 1. Scalability: PySpark is designed to handle large datasets that cannot be processed on a single machine. It can distribute the processing of data across a cluster of machines, which makes it suitable for big data applications. Pandas, on the other hand, is limited by the memory of a single machine. <br>
> 2. Speed: PySpark is faster than pandas when dealing with large datasets. This is because PySpark uses distributed computing and can process data in parallel across multiple machines. Pandas, on the other hand, is limited by the processing power of a single machine. <br>
> 4. Machine learning: PySpark has built-in machine learning libraries such as MLlib and MLflow, which makes it easy to perform machine learning tasks on large datasets. Pandas does not have built-in machine learning libraries. <br>

Overall, **`pyspark`** is a better choice for **big data** applications that require **distributed computing** and machine learning capabilities. Pandas is suitable for smaller datasets that can be processed on a single machine.


## **Start Spark Session**

First of all, we create a **`spark session`**, importing **`SparkSession`** from **`pyspark.sql`** and creating an instance, to which we will refence to when reading our data

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

# create spark session
spark = SparkSession.builder\
                    .appName('titanic')\
                    .getOrCreate()
```

## **Loading Data**

The dataset can be found from **[this source](https://raw.githubusercontent.com/AlexKbit/stepik-ds-course/master/Week3/spark-practice/train.csv)**, which we can simply download using **`wget`** in Jupyter

**`pyspark`** supports a variety of input formats, to load a **`csv`** file, we can call **`spark.read`**. When reading our table, we should ideally specify the data types. We'll see below, which types are loaded when we don't specify them. 
- Another alternative is to **automaticaly detect** suitable types for each column, we can do this by writing **`inferSchema=True`** in either **`.csv`** or **`.options`**
- Or we can specify our own **schema** using **`.schema(schema)`**, like something shown below:

```python
from pyspark.sql.types import StructTypes

schema = StructType() \
      .add("PassengerId",IntegerType(),True) \
      .add("Name",StringType(),True) \
      .add("Fare",DoubleType(),True) \
      .add("Decommisioned",BooleanType(),True)
```

Let's read our dataset, that contains a header, which requires **`header=True`**, like in **`pandas`**

```python
df = spark.read.csv('train.csv',header=True)
df.show(1,vertical=True)

# -RECORD 0---------------------------
#  PassengerId | 1                    
#  Survived    | 0                    
#  Pclass      | 3                    
#  Name        | Braund, Mr. Owen ... 
#  Sex         | male                 
#  Age         | 22.0                 
#  SibSp       | 1                    
#  Parch       | 0                    
#  Ticket      | A/5 21171            
#  Fare        | 7.25                 
#  Cabin       | null                 
#  Embarked    | S                    
# only showing top 1 row
```

## **Data Preprocessing**

### DataFrame Column Types

Like in **`pandas`**, we can call the method **`.dtypes`**, to show the column types. Default column type interpretations aren't always ideal, so its useful to load your own **`schema`**

```python
df.dtypes

# [('PassengerId', 'string'),
#  ('Survived', 'string'),
#  ('Pclass', 'string'),
#  ('Name', 'string'),
#  ('Sex', 'string'),
#  ('Age', 'string'),
#  ('SibSp', 'string'),
#  ('Parch', 'string'),
#  ('Ticket', 'string'),
#  ('Fare', 'string'),
#  ('Cabin', 'string'),
#  ('Embarked', 'string')]
```

### DataFrame Statistics

Like in **`pandas`**, we can utilise method **`describe`**, in order to show column statistics. 

```python
df.describe(['Sex','Age']).show()

# +-------+------+------------------+
# |summary|   Sex|               Age|
# +-------+------+------------------+
# |  count|   891|               714|
# |   mean|  null| 29.69911764705882|
# | stddev|  null|14.526497332334035|
# |    min|female|              0.42|
# |    max|  male|                 9|
# +-------+------+------------------+
```

### Show Missing Data

If we want to count the missing data in all our columns we can do the following:

```python
df.select([f.count(f.when(f.isnan(c) | f.col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# +-----------+--------+------+---+---+-----+-----+----+--------+-----------+-----+
# |PassengerId|Survived|Pclass|Sex|Age|SibSp|Parch|Fare|Embarked|Family_Size|Alone|
# +-----------+--------+------+---+---+-----+-----+----+--------+-----------+-----+
# |          0|       0|     0|  0|  0|    0|    0|   0|       2|          0|    0|
# +-----------+--------+------+---+---+-----+-----+----+--------+-----------+-----+
```

We can show rows with missing data using **`.where`** and **`.f.col('column').isNull()`**

```python
age_miss = df.where(f.col('Age').isNull())
age_miss.show(5)

# +-----------+--------+------+--------------------+------+----+-----+-----+------+------+-----+--------+
# |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|Ticket|  Fare|Cabin|Embarked|
# +-----------+--------+------+--------------------+------+----+-----+-----+------+------+-----+--------+
# |          6|       0|     3|    Moran, Mr. James|  male|null|    0|    0|330877|8.4583| null|       Q|
# |         18|       1|     2|Williams, Mr. Cha...|  male|null|    0|    0|244373|  13.0| null|       S|
# |         20|       1|     3|Masselmani, Mrs. ...|female|null|    0|    0|  2649| 7.225| null|       C|
# |         27|       0|     3|Emir, Mr. Farred ...|  male|null|    0|    0|  2631| 7.225| null|       C|
# |         29|       1|     3|"O'Dwyer, Miss. E...|female|null|    0|    0|330959|7.8792| null|       Q|
# +-----------+--------+------+--------------------+------+----+-----+-----+------+------+-----+--------+
```


### Dropping Irrelovant Columns

We can decide to remove columns that we won't be needing in our project by calling **`.drop`**, which is the same in **`pandas`**

```python
df = df.drop('Ticket','Name','Fare','Cabin')
df.show(5)

# +-----------+--------+------+------+---+-----+-----+--------+
# |PassengerId|Survived|Pclass|   Sex|Age|SibSp|Parch|Embarked|
# +-----------+--------+------+------+---+-----+-----+--------+
# |          1|       0|     3|  male| 22|    1|    0|       S|
# |          2|       1|     1|female| 38|    1|    0|       C|
# |          3|       1|     3|female| 26|    0|    0|       S|
# |          4|       1|     1|female| 35|    1|    0|       S|
# |          5|       0|     3|  male| 35|    0|    0|       S|
# +-----------+--------+------+------+---+-----+-----+--------+
```

### Adding Columns to DataFrame

Column additions do however work a little differently, to add a column we add **`.withColumn`**

```python
df = df.withColumn('FamilySize',f.col('SibSp') + f.col('Parch') + 1)
df.show(5)

# +-----------+--------+------+------+---+-----+-----+--------+----------+
# |PassengerId|Survived|Pclass|   Sex|Age|SibSp|Parch|Embarked|FamilySize|
# +-----------+--------+------+------+---+-----+-----+--------+----------+
# |          1|       0|     3|  male| 22|    1|    0|       S|       2.0|
# |          2|       1|     1|female| 38|    1|    0|       C|       2.0|
# |          3|       1|     3|female| 26|    0|    0|       S|       1.0|
# |          4|       1|     1|female| 35|    1|    0|       S|       2.0|
# |          5|       0|     3|  male| 35|    0|    0|       S|       1.0|
# +-----------+--------+------+------+---+-----+-----+--------+----------+
```

We can also define a condition, based on which we'll create a unique feature

```python
ndf = ndf.withColumn('M',f.col('Sex') == 'male')
ndf = ndf.withColumn('F',f.col('Sex') == 'female')
ndf = ndf.drop('sex')
```

### Data Imputation

Data imputation can be done via **`fillna`**, we pass a dictionary containing key,value pair for column name and value respectively 

```python
av_age = df.select(f.avg(f.col('age')))
av_age.show()

# +-----------------+
# |         avg(age)|
# +-----------------+
# |29.69911764705882|
# +-----------------+

# To convert to python types, we can write:
av_age.collect()[0][0]  # 29.69911764705882
```

```python
ndf = df.fillna({'age':av_age.collect()[0][0]})
ndf.show(5)

# +-----------+--------+------+--------------------+------+-----------------+-----+-----+----------------+-------+-----+--------+
# |PassengerId|Survived|Pclass|                Name|   Sex|              Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
# +-----------+--------+------+--------------------+------+-----------------+-----+-----+----------------+-------+-----+--------+
# |          1|       0|     3|Braund, Mr. Owen ...|  male|             22.0|    1|    0|       A/5 21171|   7.25| null|       S|
# |          2|       1|     1|Cumings, Mrs. Joh...|female|             38.0|    1|    0|        PC 17599|71.2833|  C85|       C|
# |          3|       1|     3|Heikkinen, Miss. ...|female|             26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|
# |          4|       1|     1|Futrelle, Mrs. Ja...|female|             35.0|    1|    0|          113803|   53.1| C123|       S|
# |          5|       0|     3|Allen, Mr. Willia...|  male|             35.0|    0|    0|          373450|   8.05| null|       S|
# +-----------+--------+------+--------------------+------+-----------------+-----+-----+----------------+-------+-----+--------+
```

## **Conclusion**

Let's review what we have covered in this post:
- We learned how to drop columns, using **`.drop`**
- We learned how to extract statistical data from our dataframe, using **`.select`** and functions **`f.avg('column')`**
- We known how to fill missing data in different columns using a single value with a dictionary; **`f.fillna({'column':'value'})`**
- Add or replace a column, using **`f.withColumn`**
- **`StringIndexer(inputCol,outputCol).fit(data)`** - convert categorical into a numerical representation
- Once we are done with our feature matrix, we can convert all the relevant features into a single feature that will be used as input into the model using **`VectorAssembler(inputCols,outputCol).transform(data)`**

