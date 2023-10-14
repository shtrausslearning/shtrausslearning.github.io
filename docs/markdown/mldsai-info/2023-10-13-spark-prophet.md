
# **Prophet with PySpark**

![](https://i.imgur.com/b6BBtXv.jpg)

:material-timetable: 2023-08-10 00:00:00 <br>
:material-group: `pyspark` <br>
:fontawesome-solid-tags: `pyspark`,`regression`

![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)

In this notebook, we look at how to use a popular machine learning library `prophet` with `pyspark`. `pyspark` itself does not contain such an additive regression model, however we can utilise user defined functions (`UDF`), which allows us to use different functionality that is not available in `pyspark`.

## <b>Background</b>

### <b><span style='color:#be61c7;text-align:center'>❯❯ </span>Prophet</b> 

`Prophet` is a time series forecasting model. It is based on an additive regression model that takes into account trends, seasonality, and holidays. `Prophet` also allows for the inclusion of external regressors and can handle missing data and outliers. It uses Bayesian inference to estimate the parameters of the model and provides uncertainty intervals for the forecasts. 

### <b><span style='color:#be61c7;text-align:center'>❯❯ </span>UDF</b> 

Pandas `UDFs` (User-Defined Functions) allow you to apply a Python function that operates on pandas data frames to Spark data frames. This allows you to leverage the power of pandas, which is a popular data manipulation library in Python, in your PySpark applications. Pandas `UDFs` can take one or more input columns and return one or more output columns, which can be of any data type supported by Spark. With Pandas `UDFs`, you can perform complex data manipulations that are not possible using built-in Spark SQL functions.

### <b><span style='color:#be61c7;text-align:center'>❯❯ </span>Avocado Price Prediction</b> 

Avocado price prediction is the process of using machine learning algorithms to forecast the future prices of avocados based on historical data and other relevant factors such as weather patterns, consumer demand, and supply chain disruptions. This can help stakeholders in the avocado industry make informed decisions about when and where to sell their avocados, as well as how much to charge for them. Avocado price prediction can also provide insights into the factors that affect avocado sales and help optimize the industry's efficiency and profitability.

### <b><span style='color:#be61c7;text-align:center'>❯❯ </span>Objective</b> 

Having done some posts on `pyspark`, it seems like a very intuitive library to use

## <b>The Dataset</b>

It is a well known fact that Millenials LOVE Avocado Toast. It's also a well known fact that all Millenials live in their parents basements.Clearly, they aren't buying home because they are buying too much Avocado Toast! But maybe there's hope… if a Millenial could find a city with cheap avocados, they could live out the Millenial American Dream.

The dataset can be found on **[Kaggle](https://www.kaggle.com/datasets/neuromusic/avocado-prices)** & its original source found **[here](https://hassavocadoboard.com/)**

### <b><span style='color:#be61c7;text-align:center'>❯❯ </span>Loading data</b> 

To load the data, we start a spark session on local

```python
! pip install pyspark

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pandas as pd

# Start spark session
spark = SparkSession.builder\
                    .master("local")\
                    .appName("prophet")\
                    .getOrCreate()
```

To read the data, we'll use the `session.read.csv`, together with `inferSchema` method and look at the table schematics using `printSchema()` method

```python
# read csv
sales = spark.read.csv('/kaggle/input/avocado-prices/avocado.csv',header=True,inferSchema=True)
sales.printSchema()
```

```
root
 |-- _c0: string (nullable = true)
 |-- Date: string (nullable = true)
 |-- AveragePrice: string (nullable = true)
 |-- Total Volume: string (nullable = true)
 |-- 4046: string (nullable = true)
 |-- 4225: string (nullable = true)
 |-- 4770: string (nullable = true)
 |-- Total Bags: string (nullable = true)
 |-- Small Bags: string (nullable = true)
 |-- Large Bags: string (nullable = true)
 |-- XLarge Bags: string (nullable = true)
 |-- type: string (nullable = true)
 |-- year: string (nullable = true)
 |-- region: string (nullable = true)
```

```python
sales.select('Date','type','Total Volume','region').orderBy('Date').show(5)

+----------+------------+------------+----------------+
|      Date|        type|Total Volume|          region|
+----------+------------+------------+----------------+
|2015-01-04|conventional|   116253.44|BuffaloRochester|
|2015-01-04|conventional|   158638.04|        Columbus|
|2015-01-04|conventional|   5777334.9|      California|
|2015-01-04|conventional|   435021.49|         Atlanta|
|2015-01-04|conventional|   166006.29|       Charlotte|
+----------+------------+------------+----------------+
```




**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**