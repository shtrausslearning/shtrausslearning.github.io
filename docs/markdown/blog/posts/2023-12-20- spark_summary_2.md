---
date: 2023-12-20
title: PySpark Daily December Summary II
authors: [andrey]
draft: false
categories:
     - PySpark
tags:
     - pyspark
comments: true
---

# **PySpark Daily December Summary II**

<div style="width: 100%; font-family: Trebuchet MS; font-weight: bold;">
    <div style="padding-top: 40%; position: relative; background-color: #000000; border-radius:10px;">
        <div style="background-image: url('images/spark_intro.png'); background-size: cover; background-position: center; position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.5; border-radius:10px">
        </div>
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;">
            <div style="position: relative; display: table; height: 75%; width: 100%;">
            </div>
            <div style="position: absolute; bottom: 30px; left: 30px;">
            </div>
        </div>
    </div>
</div>

Continuing on where we left off last post, I'll be exploring **pypspark** on a daily basis, just to get more used to it. Here I will be posting summaries that cover roughtly 10 days worth of posts that I make **[on Kaggle](https://www.kaggle.com/code/shtrausslearning/mldsai-pyspark-daily-posts)**, so that would equate to three posts a month

<!-- more -->

[![Open Notebook](https://img.shields.io/badge/Kaggle-View-006eca?logo=Jupyter&logoColor=3094e7)](https://www.kaggle.com/code/shtrausslearning/mldsai-pyspark-daily-posts)

### <b>10/12/2023</b>

<h4><b><span style='color:#E888BB'>❯❯❯</span> SQL like functions (ORDER BY)</b></h4>

Ordering a column using **orderBy** based on ascending **f.col.asc()** or descending order **f.col.desc()**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Create a DataFrame
data = [("Alice", 25), 
        ("Bob", 30), 
        ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])
df.show()

+-------+---+
|   name|age|
+-------+---+
|  Alice| 25|
|    Bob| 30|
|Charlie| 35|
+-------+---+
```

```python
# Order DataFrame by age in descending order
df.orderBy(f.col('age').desc()).show()

+-------+---+
|   name|age|
+-------+---+
|Charlie| 35|
|    Bob| 30|
|  Alice| 25|
+-------+---+
```

If we want to sort with **two columns**, which is useful when we have the same values in the first column, we would do something like this:

```python
# Create a DataFrame
data = [("Alice", 25, 180), 
        ("Bob", 25, 150), 
        ("Charlie", 35, 167)]
df = spark.createDataFrame(data, ["name", "age","height"])
df.show()
```

```python
# Order DataFrame by age in descending order
df.orderBy(f.col('age').desc(),f.col('height').asc()).show()

+-------+---+------+
|   name|age|height|
+-------+---+------+
|Charlie| 35|   167|
|    Bob| 25|   150|
|  Alice| 25|   180|
+-------+---+------+
```

### <b>11/12/2023</b>

<h4><b><span style='color:#E888BB'>❯❯❯</span> SQL like functions (JOIN)</b></h4>

Joining dataframes is of course an important part of data analysis:

- Using **pyspark.sql**, we can join dataframes with the notation shown in 01/12/2023, however **pyspark** dataframe has its own method for joining dataframe tables **join()**
- As with the pandas notation of **merge** df1.merge(df2), we can join dataframes using the a similar notation **df1.join(df2,'on','how')**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder.getOrCreate()

# Create sample data for dataset1 (January)
dataset1 = spark.createDataFrame([
    ("2021-01-01", 10),
    ("2021-01-02", 20),
    ("2021-01-03", 30),
    ("2021-01-04", 70)
], ["date", "value1"])

# Create sample data for dataset2 (February)
dataset2 = spark.createDataFrame([
    ("2021-02-01", 40),
    ("2021-02-02", 50),
    ("2021-02-03", 60),
    ("2021-02-05", 70)
], ["date", "value2"])

# Show the joined dataframe
dataset1.show()
dataset2.show()

+----------+------+
|      date|value1|
+----------+------+
|2021-01-01|    10|
|2021-01-02|    20|
|2021-01-03|    30|
|2021-01-04|    70|
+----------+------+

+----------+------+
|      date|value2|
+----------+------+
|2021-02-01|    40|
|2021-02-02|    50|
|2021-02-03|    60|
|2021-02-05|    70|
+----------+------+
```

If we choose to **inner join**, we will be left with only three rows, since they both share them

```python
dataset1.join(dataset2,on='date',how='inner').show()

+----------+------+------+
|      date|value1|value2|
+----------+------+------+
|2021-01-01|    10|    40|
|2021-01-02|    20|    50|
|2021-01-03|    30|    60|
+----------+------+------+
```

**LEFT JOIN** uses all the values in the left table, if some data is missing in the right it will be replaced with **NULL**

```python
dataset1.join(dataset2,on='date',how='inner').show()

+----------+------+------+
|      date|value1|value2|
+----------+------+------+
|2021-01-01|    10|    40|
|2021-01-02|    20|    50|
|2021-01-03|    30|    60|
|2021-01-04|    70|  NULL|
+----------+------+------+
```

Similar to **RIGHT JOIN**


```python
dataset1.join(dataset2,on='date',how='right').show()

+----------+------+------+
|      date|value1|value2|
+----------+------+------+
|2021-01-01|    10|    40|
|2021-01-02|    20|    50|
|2021-01-03|    30|    60|
|2021-01-05|  NULL|    70|
+----------+------+------+
```

And **FULL OUTER** join as well:

```python
dataset1.join(dataset2,on='date',how='outer').show()

+----------+------+------+
|      date|value1|value2|
+----------+------+------+
|2021-01-01|    10|    40|
|2021-01-02|    20|    50|
|2021-01-03|    30|    60|
|2021-01-04|    70|  NULL|
|2021-01-05|  NULL|    70|
+----------+------+------+
```

### <b>12/12/2023</b>

<h4><b><span style='color:#E888BB'>❯❯❯</span> PySpark UDF (Standard UDF)</b></h4>

PySpark **UDFs** are custom functions that can be created and applied to DataFrame columns in PySpark. They allow users to perform custom computations or transformations on DataFrame data by defining their own functions and applying them **to specific columns**, 

A UDF example which takes one column value, the only thing to note is that we need to define the **output type** of the function:

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.getOrCreate()

# Create a DataFrame
data = [("Alice", 25), 
        ("Bob", 30), 
        ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])

# UDF function which accepts one column & multiplies it by itself
def square(num):
    return num * num

# register UDF
square_udf = udf(square, IntegerType())

# Apply the UDF to a DataFrame column
new_df = df.withColumn("square", square_udf(df["age"]))
new_df.show()

+-------+---+------+
|   name|age|square|
+-------+---+------+
|  Alice| 25|   625|
|    Bob| 30|   900|
|Charlie| 35|  1225|
+-------+---+------+
```

We can also use more than one column of data

```python
def add(num1,num2):
    return num1 + num2

add_udf = udf(add,IntegerType())

df = df.withColumn("added2",add_udf(df['age'],df['added2']))
df.show()

+-------+---+------+
|   name|age|added2|
+-------+---+------+
|  Alice| 25|    50|
|    Bob| 30|    60|
|Charlie| 35|    70|
+-------+---+------+
```

**Lambda functions** can also be used instead of standard python functions

```python
add_udf = udf(lambda x,y : x + y,IntegerType())
df = df.withColumn("added3",add_udf(df['age'],df['age']))
df.show()

+-------+---+------+
|   name|age|added3|
+-------+---+------+
|  Alice| 25|    50|
|    Bob| 30|    60|
|Charlie| 35|    70|
+-------+---+------+
```



***

**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)** or simply below!