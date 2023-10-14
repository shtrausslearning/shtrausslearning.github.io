
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

To read the data, we'll use the `session.read.csv`, together with `inferSchema` method and look at the table schematics using `printSchema()` method to automatically assign types to table columns

```python
# read csv
sales = spark.read.csv('/kaggle/input/avocado-prices/avocado.csv',header=True,inferSchema=True)
sales.printSchema()
```

```
root
 |-- _c0: integer (nullable = true)
 |-- Date: date (nullable = true)
 |-- AveragePrice: double (nullable = true)
 |-- Total Volume: double (nullable = true)
 |-- 4046: double (nullable = true)
 |-- 4225: double (nullable = true)
 |-- 4770: double (nullable = true)
 |-- Total Bags: double (nullable = true)
 |-- Small Bags: double (nullable = true)
 |-- Large Bags: double (nullable = true)
 |-- XLarge Bags: double (nullable = true)
 |-- type: string (nullable = true)
 |-- year: integer (nullable = true)
 |-- region: string (nullable = true)


```

## <b>Exploring Data</b>

Having loaded our data, we sure can do some data exploration, first lets take a peek at our dataset, we'll use `select`,`orderBy` & `show` methods

```python
sales.select('Date','type','Total Volume','region')\
     .orderBy('Date')\
     .show(5)
```
```
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

The `Date` unique values can be called and checked, we have weekly data for different regions

```python
sales.select(col('Date')).distinct().orderBy('Date').show(5)
```

```
+----------+
|      Date|
+----------+
|2015-01-04|
|2015-01-11|
|2015-01-18|
|2015-01-25|
|2015-02-01|
+----------+
only showing top 5 rows
```

We will be using `Total Volume` as our target variable we'll be predicting. We also can note that we have different types `type` of avocados (organic and conventional)

```python
sales.select(col('type')).distinct().show()
```

```
+------------+
|        type|
+------------+
|     organic|
|conventional|
+------------+
```

So what we'll be doing is creating a model to predict the sales for both of these types, which is something we'll need to incorporate into our `UDF`

We can also check the `region` limits for `Total Volume`, we can do this by using `agg` method with the `groupby` dataframe type (`pyspark.sql.group.GroupedData`):

```
# min and maximum of sale volume
sales.groupby('region').agg(f.max('Total Volume')).show()  # get max of column
sales.groupby('region').agg(f.min('Total Volume')).show()  # get min of column
```

```
+------------------+-----------------+
|            region|max(Total Volume)|
+------------------+-----------------+
|     PhoenixTucson|       2200550.27|
|       GrandRapids|        408921.57|
|     SouthCarolina|        706098.15|
|           TotalUS|    6.250564652E7|
|  WestTexNewMexico|       1637554.42|
|        Louisville|        169828.77|
|      Philadelphia|         819224.3|
|        Sacramento|         862337.1|
|     DallasFtWorth|       1885401.44|
|      Indianapolis|        335442.41|
|          LasVegas|        680234.93|
|         Nashville|        391780.25|
|        GreatLakes|       7094764.73|
|           Detroit|        880540.45|
|            Albany|        216738.47|
|          Portland|       1189151.17|
|  CincinnatiDayton|        538518.77|
|          SanDiego|        917660.79|
|             Boise|        136377.55|
|HarrisburgScranton|        395673.05|
+------------------+-----------------+
only showing top 20 rows

+------------------+-----------------+
|            region|min(Total Volume)|
+------------------+-----------------+
|     PhoenixTucson|          4881.79|
|       GrandRapids|           683.76|
|     SouthCarolina|           2304.3|
|           TotalUS|        501814.87|
|  WestTexNewMexico|          4582.72|
|        Louisville|           862.59|
|      Philadelphia|           1699.0|
|        Sacramento|          3562.52|
|     DallasFtWorth|          6568.67|
|      Indianapolis|           964.25|
|          LasVegas|           2988.4|
|         Nashville|          2892.29|
|        GreatLakes|         56569.37|
|           Detroit|          4973.92|
|            Albany|            774.2|
|          Portland|          7136.88|
|  CincinnatiDayton|          6349.77|
|          SanDiego|          5564.87|
|             Boise|           562.64|
|HarrisburgScranton|           971.81|
+------------------+-----------------+
only showing top 20 rows
```

We can note that we have data for not only the different `regions`, but also for the entire country `TotalUS`. Also interesting to note is that the difference in `max` and `min` values is quite high.

Let's find the locations (`region`) with the highest `total volumes` 

```
from pyspark.sql.functions import desc,col

by_volume = sales.orderBy(desc("Total Volume"))\
                 .where(col('region') != 'TotalUS')
by_volume.show(5)
```

```
+---+----------+------------+-------------+----------+----------+---------+----------+----------+----------+-----------+------------+----+----------+
|_c0|      Date|AveragePrice| Total Volume|      4046|      4225|     4770|Total Bags|Small Bags|Large Bags|XLarge Bags|        type|year|    region|
+---+----------+------------+-------------+----------+----------+---------+----------+----------+----------+-----------+------------+----+----------+
| 47|2017-02-05|        0.66|1.127474911E7|4377537.67|2558039.85|193764.89| 4145406.7|2508731.79|1627453.06|    9221.85|conventional|2017|      West|
| 47|2017-02-05|        0.67|1.121359629E7|3986429.59|3550403.07|214137.93| 3462625.7|3403581.49|   7838.83|   51205.38|conventional|2017|California|
|  7|2018-02-04|         0.8|1.089467777E7|4473811.63|4097591.67|146357.78|2176916.69|2072477.62|  34196.27|    70242.8|conventional|2018|California|
|  7|2018-02-04|        0.83|1.056505641E7|3121272.58|3294335.87|142553.21|4006894.75|1151399.33|2838239.39|   17256.03|conventional|2018|      West|
| 46|2016-02-07|         0.7|1.036169817E7|2930343.28|3950852.38| 424389.6|3056112.91|2693843.02| 344774.59|    17495.3|conventional|2016|California|
+---+----------+------------+-------------+----------+----------+---------+----------+----------+----------+-----------+------------+----+----------+
only showing top 5 rows
```

We can note that `California` & `West` regions have had the highest values for `Total Volume` on 2017-02-05

Its also interest to note the difference in `Total Volume` for both types of avocado, so lets check that, lets just check the difference in `max` values

```
by_volume.groupby('type').agg(f.max('Total Volume')).show()
```

```
+------------+-----------------+
|        type|max(Total Volume)|
+------------+-----------------+
|     organic|        793464.77|
|conventional|    1.127474911E7|
+------------+-----------------+
```

So we can note that tehre is a significant diffence in `Total Volume`, let's also check when this actually occured:

```
by_volume.filter(f.col('Total Volume') == 1.127474911E7).show()
by_volume.filter(f.col('Total Volume') == 793464.77).show()

```

```
+---+----------+------------+-------------+----------+----------+---------+----------+----------+----------+-----------+------------+----+------+
|_c0|      Date|AveragePrice| Total Volume|      4046|      4225|     4770|Total Bags|Small Bags|Large Bags|XLarge Bags|        type|year|region|
+---+----------+------------+-------------+----------+----------+---------+----------+----------+----------+-----------+------------+----+------+
| 47|2017-02-05|        0.66|1.127474911E7|4377537.67|2558039.85|193764.89| 4145406.7|2508731.79|1627453.06|    9221.85|conventional|2017|  West|
+---+----------+------------+-------------+----------+----------+---------+----------+----------+----------+-----------+------------+----+------+
```

```
+---+----------+------------+------------+--------+---------+-----+----------+----------+----------+-----------+-------+----+---------+
|_c0|      Date|AveragePrice|Total Volume|    4046|     4225| 4770|Total Bags|Small Bags|Large Bags|XLarge Bags|   type|year|   region|
+---+----------+------------+------------+--------+---------+-----+----------+----------+----------+-----------+-------+----+---------+
|  5|2018-02-18|        1.39|   793464.77|150620.0|425616.86|874.9| 216353.01| 197949.51|   18403.5|        0.0|organic|2018|Northeast|
+---+----------+------------+------------+--------+---------+-----+----------+----------+----------+-----------+-------+----+---------+
```


**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**