---
date: 2023-08-07
title: Exploring Taxi Trip Data
authors: [andrey]
draft: true
categories:
     - pyspark
tags:
     - pyspark
     - eda
comments: true
---

# Exploring Taxi Trip Data

В этом посте мы будем использовать `spark` для раздедовательного анализа поездак такси для данных их города Чикаго. Когда данных становится слишком много, `pandas` работает медленее чем `spark`, поэтому будем использовать этот инструмент для разведовательного анализа!

<!-- more -->

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1l77AlQZtF0CiVS6g_WsSonlQ2Dj2SlPF?usp=sharing)

## Введение

### Spark over Pandas

`pyspark` может быть быстрее `pandas` благодаря использованию распределенных вычислений и оптимизации выполнения запросов. В отличие от `pandas`, который работает на одной машине и обрабатывает данные в памяти, `spark` может распределять задачи на кластере из сотен или тысяч узлов, что позволяет обрабатывать большие объемы данных и ускорять выполнение запросов. Кроме того, `pyspark` использует оптимизацию выполнения запросов, которая позволяет минимизировать время выполнения запросов и ускорять обработку данных.

`spark` имеет несколько преимуществ перед `pandas`:

> 1. Масштабируемость: Spark может обрабатывать данные в масштабах, которые Pandas не может обработать. Spark может работать с данными, которые не помещаются в память одной машины, используя распределенные вычисления. <br>
> 2. Распределенные вычисления: Spark может работать на кластерах из сотен или тысяч узлов, что позволяет распределять задачи и ускорять обработку данных. <br>
> 3. Быстродействие: Spark может обрабатывать данные быстрее, чем Pandas, благодаря оптимизации выполнения запросов и использованию распределенных вычислений.

Будем использовать `python`, в этой среде можно установть `spark` с помощью `pip install pyspark`


### Цель задачи

В этом посте мы проведем разведовательный анализ данных; зададим себе ряд вопросов на которые мы найдем ответ в наших данных 

## Данные

В компании есть ряд таксистов которые регистриуют поедки, и если в ней работают много таксистов, то данные о поездках очень стремительно набираются. Соответсвенно для разведовательного анализа больших данных будем использовать `spark`.

Данные поездок можно найти [по ссылке](https://github.com/AlexKbit/stepik-ds-course/blob/master/Week3/spark-tasks/taxi_data.parquet), они в формате `parquet` (бинарный формат), нам так же понадобится еще одна таблица из базы данных, ее можно [найти сдесь](https://github.com/AlexKbit/stepik-ds-course/raw/master/Week3/spark-tasks/taxi_cars_data.parquet)

Этот датасет собранный на основе данных <a href="https://www.kaggle.com/chicago/chicago-taxi-rides-2016">Chicago Taxi Rides 2016</a>

<br>Схема данны:
<br>|-- <hlt>taxi_id</hlt> = идентификатор таксиста
<br>|-- trip_start_timestamp = время начала поездки
<br>|-- trip_end_timestamp = время окончания поездки
<br>|-- trip_seconds = время длительности поездки в секундах
<br>|-- trip_miles = мили проиденные во время поездки
<br>|-- fare = транспортные расходы
<br>|-- tips = назначенные чаевые
<br>|-- trip_total = общая стоимость поездки (Итоговая с учетом чаевых и расходов)
<br>|-- payment_type = тип оплаты

## SparkSession

Начинаем `spark` сессию на локальном компе:

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as f 

spark = SparkSession.builder \
                    .appName('Task') \
                    .getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "GMT+3")
```

Подгружаем данные c помощью `read.parquet`

```python
taxi = spark.read.parquet('taxi_data.parquet')
taxi.show(5)

# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles| fare|tips|trip_total|payment_type|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+
# |   5240| 2016-12-15 23:45:00|2016-12-16 00:00:00|         900|       2.5|10.75|2.45|      14.7| Credit Card|
# |   1215| 2016-12-12 07:15:00|2016-12-12 07:15:00|         240|       0.4|  5.0| 3.0|       9.5| Credit Card|
# |   3673| 2016-12-16 16:30:00|2016-12-16 17:00:00|        2400|      10.7| 31.0| 0.0|      31.0|        Cash|
# |   5400| 2016-12-16 08:45:00|2016-12-16 09:00:00|         300|       0.0| 5.25| 2.0|      7.25| Credit Card|
# |   1257| 2016-12-03 18:45:00|2016-12-03 18:45:00|         360|       0.3|  5.0| 0.0|       5.0|        Cash|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+
```

## Количество Партиции

В `spark` мы можем распараллеливать нашу коллекцию и паралельно выполнять задачи, и выполнять задачи на мощьных кластерах

Мы загрузили данные которые находяться в одном `parquet`, но мы так же можем работать и с папками в которых хранятся в отдельных файлах (частями)

Чтобы показать количество действующих партиции; сдесь 2 партиции

```python
taxi.rdd.getNumPartitions()
# 2
```

Мы можем перераспределить количество партиции для нашей задачи

```python
taxi.repartition(16).write.parquet('data_source')
```

Если мы сейчас сохраним `parquet`, то наш файл сохранился бы в папке частями. Воспользуемся `repartition` для увеличения количество партиции, а `coalesce` для уменьшения

```python
# df.repartition(16) -> increase split
# df.rdd.getNumPartitions() = 2
# df.coalesce(1) -> decrease split (eg. assemble into one)
```

## Разведовательный Анализ

Все готово, давайте теперь будем задавать себе вопросы которые нас интересуют!  

### Посчитайте количество загруженных строк

```python
taxi.count()
# 2540712

# len(taxi.columns) # для получения количество столбцов
```

Мы загрузили данные без схемы типов, посмотрим все ли нас устраивает, если нет то мы можем добавить `schema` при чтении данных

```python
taxi.printSchema()

root
 # |-- taxi_id: integer (nullable = true)
 # |-- trip_start_timestamp: timestamp (nullable = true)
 # |-- trip_end_timestamp: timestamp (nullable = true)
 # |-- trip_seconds: integer (nullable = true)
 # |-- trip_miles: double (nullable = true)
 # |-- fare: double (nullable = true)
 # |-- tips: double (nullable = true)
 # |-- trip_total: double (nullable = true)
 # |-- payment_type: string (nullable = true)
```

### Чему равна корреляция и ковариация между длиной маршрута и ценой за поездку?

У `dataframe` есть методы `cov` и `corr`, подробнее <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.corr.html?highlight=corr#pyspark-sql-dataframe-corr">corr</a> & <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.cov.html?highlight=cov#pyspark-sql-dataframe-cov">cov</a>

- Длина маршрута `trip_miles`, цена поездки `trip_total`

```python
print(f"correlation: {round(taxi.corr('trip_miles','trip_total'),5)}")
print(f"covariance: {round(taxi.cov('trip_miles','trip_total'),5)}")

# correlation: 0.44816
# covariance: 71.96914
```

### Найдите количество, среднее, cреднеквадратическое отклонение, минимум и максимум для длины маршрута и цены за поездку?

№3 Найдите количество, среднее, cреднеквадратическое отклонение, минимум и максимум для длины маршрута и цены за поездку? Подробнее <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.describe.html?highlight=describe#pyspark-sql-dataframe-describe">describe</a>

```python
taxi.describe().show()

#+-------+-----------------+------------------+------------------+------------------+------------------+------------------+------------+
#|summary|          taxi_id|      trip_seconds|        trip_miles|              fare|              tips|        trip_total|payment_type|
#+-------+-----------------+------------------+------------------+------------------+------------------+------------------+------------+
#|  count|          2539547|           2540178|           2540677|           2540672|           2540672|           2540672|     2540712|
#|   mean|4370.670382946249|  801.015424509621|3.0005873828090266|13.248720862039738|1.5209281087837443|15.913560215564042|        null|
#| stddev|2513.977996552665|1199.4924572375417|  5.25716922943536|22.579448541941893|2.7990862329785924|30.546699217618237|        null|
#|    min|                0|                 0|               0.0|               0.0|               0.0|               0.0|        Cash|
#|    max|             8760|             86399|             900.0|           9276.62|             422.0|           9276.69|    Way2ride|
#+-------+-----------------+------------------+------------------+------------------+------------------+------------------+------------+

```

### Найдите самый непопулярный вид оплаты

<br>Подробнее <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.groupBy.html?highlight=groupby#pyspark-sql-dataframe-groupby">groupBy</a> <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.orderBy.html?highlight=orderby#pyspark-sql-dataframe-orderby">orderBy</a>

```python
taxi.groupBy('payment_type').count().orderBy('count', ascending=True).show()

# +------------+-------+
# |payment_type|  count|
# +------------+-------+
# |    Way2ride|      3|
# |       Pcard|    878|
# |      Prcard|    968|
# |     Dispute|   1842|
# |     Unknown|   5180|
# |   No Charge|  12843|
# | Credit Card|1108843|
# |        Cash|1410155|
# +------------+-------+
```



### Найдите идентификатор `taxi_id` таксиста выполнившего наибольшее число заказов


```python
taxi.groupby('taxi_id').count().orderBy('count',ascending=False).show(5)

# +-------+-----+
# |taxi_id|count|
# +-------+-----+
# |    316| 2225|
# |   6591| 2083|
# |   5071| 2080|
# |   8740| 2067|
# |   6008| 2033|
# +-------+-----+
```


### Чему равна средняя цена среди поездок, оплаченных наличными?

<br> Подробней <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.where.html?highlight=where#pyspark-sql-dataframe-where">where</a>

- Наличные (`Cash`)
- `filter` or `where` для выбора подвыборки 

```python
cash = taxi.filter(f.col('payment_type') == 'Cash').show(1)

# +-------+--------------------+-------------------+------------+----------+----+----+----------+------------+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles|fare|tips|trip_total|payment_type|
# +-------+--------------------+-------------------+------------+----------+----+----+----------+------------+
# |   3673| 2016-12-16 16:30:00|2016-12-16 17:00:00|        2400|      10.7|31.0| 0.0|      31.0|        Cash|
# +-------+--------------------+-------------------+------------+----------+----+----+----------+------------+
```


```python
cash = taxi.filter(f.col('payment_type') == 'Cash')
taxi.groupBy(['payment_type']).agg(f.mean('trip_total')).show()
# cash.groupBy(['payment_type']).agg(f.mean('trip_total')).show()

# +------------+------------------+
# |payment_type|   avg(trip_total)|
# +------------+------------------+
# | Credit Card| 20.88679166806002|
# |   No Charge| 14.35603753017208|
# |     Unknown|13.079387794515267|
# |      Prcard|11.893749999999997|
# |        Cash|12.035261470840307|
# |     Dispute|14.686368078175896|
# |    Way2ride| 4.793333333333333|
# |       Pcard|10.450512528473803|
# +------------+------------------+
```


### Сколько таксистов проехало больше 1000 миль за все время выполнения заказов?

```python
driver_distances = taxi.groupby(['taxi_id']).agg(f.sum('trip_miles').alias('distance')).orderBy('distance',ascending=False)
driver_distances.filter(f.col('distance') > 1000).count()
# 2860
```

### Сколько миль проехал пассажир в самой долгой поездке?

```python
taxi2 = taxi.dropna()
taxi2.orderBy(['trip_seconds'],ascending=False).show(5)

# +-------+--------------------+-------------------+------------+----------+----+----+----------+------------+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles|fare|tips|trip_total|payment_type|
# +-------+--------------------+-------------------+------------+----------+----+----+----------+------------+
# |   4161| 2016-11-14 16:00:00|2016-11-15 16:00:00|       86399|       0.0|3.25| 0.0|      3.25|        Cash|
# |   5667| 2016-11-04 21:30:00|2016-11-05 21:30:00|       86399|       0.0|3.25| 0.0|      4.75|        Cash|
# |   1954| 2016-11-03 00:15:00|2016-11-04 00:15:00|       86399|       0.0|3.25| 0.0|      3.25|        Cash|
# |   4219| 2016-11-08 16:00:00|2016-11-09 16:00:00|       86392|       0.0|3.25| 0.0|      3.25|        Cash|
# |   4551| 2016-11-03 16:15:00|2016-11-04 16:15:00|       86389|       0.0|3.25| 0.0|      3.25|        Cash|
# +-------+--------------------+-------------------+------------+----------+----+----+----------+------------+
```


###Каков средний заработок всех таксистов?

<br>Отсеките неизвестные машины (не определенный taxi_id).

```python
# we have null taxi_id
taxi.select('taxi_id').distinct().orderBy('taxi_id').show(4)

# +-------+
# |taxi_id|
# +-------+
# |   null|
# |      0|
# |      3|
# |      5|
# +-------+
```

```python
# drop rows in taxi_id that are null
taxi_cleaned = taxi.dropna(how='all',subset=('taxi_id'))
taxi_cleaned.show()

# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles| fare|tips|trip_total|payment_type|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+
# |   5240| 2016-12-15 23:45:00|2016-12-16 00:00:00|         900|       2.5|10.75|2.45|      14.7| Credit Card|
# |   1215| 2016-12-12 07:15:00|2016-12-12 07:15:00|         240|       0.4|  5.0| 3.0|       9.5| Credit Card|
# |   3673| 2016-12-16 16:30:00|2016-12-16 17:00:00|        2400|      10.7| 31.0| 0.0|      31.0|        Cash|
# |   5400| 2016-12-16 08:45:00|2016-12-16 09:00:00|         300|       0.0| 5.25| 2.0|      7.25| Credit Card|
# |   1257| 2016-12-03 18:45:00|2016-12-03 18:45:00|         360|       0.3|  5.0| 0.0|       5.0|        Cash|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+
```

Удаляем все строки в `taxi_id` в которых у нас есть пробелы


```python
# mean total income of all drivers
driver_total.select('total_driver').agg(f.mean('total_driver')).show()

# +-----------------+
# |avg(total_driver)|
# +-----------------+
# |8218.856265256327|
# +-----------------+
```

```python
# driver total for each taxi_id
driver_total = taxi_cleaned.groupby('taxi_id').agg(f.sum('trip_total').alias('total_driver'))
driver_total.show()

# +-------+------------------+
# |taxi_id|      total_driver|
# +-------+------------------+
# |   3997|10372.800000000003|
# |   6620|13823.139999999994|
# |   4900| 9867.160000000002|
# |   7833| 8439.289999999999|
# |   1829| 9979.389999999998|
# +-------+------------------+
```


### Сколько поездок начиналось в самый загруженный час?

<br>Используйте функцию <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.functions.hour.html?highlight=hour#pyspark-sql-functions-hour">hour</a>

```python
# extact hour from date, like datetime in pandas
# dont' use select and then join; use withColumn('name',operation)

col_hour = taxi.withColumn('hour',f.hour('trip_start_timestamp'))
col_hour.show(5)

# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles| fare|tips|trip_total|payment_type|hour|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----+
# |   5240| 2016-12-15 23:45:00|2016-12-16 00:00:00|         900|       2.5|10.75|2.45|      14.7| Credit Card|  23|
# |   1215| 2016-12-12 07:15:00|2016-12-12 07:15:00|         240|       0.4|  5.0| 3.0|       9.5| Credit Card|   7|
# |   3673| 2016-12-16 16:30:00|2016-12-16 17:00:00|        2400|      10.7| 31.0| 0.0|      31.0|        Cash|  16|
# |   5400| 2016-12-16 08:45:00|2016-12-16 09:00:00|         300|       0.0| 5.25| 2.0|      7.25| Credit Card|   8|
# |   1257| 2016-12-03 18:45:00|2016-12-03 18:45:00|         360|       0.3|  5.0| 0.0|       5.0|        Cash|  18|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----+
```


```python
col_hour.groupby('hour').count().orderBy('count',ascending=False).show(5)

# +----+------+
# |hour| count|
# +----+------+
# |  18|181127|
# |  19|173779|
# |  17|169886|
# |  16|156519|
# |  20|146602|
# +----+------+
```

### Сколько поездок началось во второй четверти суток?

```python
taxi_add = taxi.withColumn('start_hour',f.hour('trip_start_timestamp'))
taxi_add.show(5)

# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles| fare|tips|trip_total|payment_type|start_hour|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+
# |   5240| 2016-12-15 23:45:00|2016-12-16 00:00:00|         900|       2.5|10.75|2.45|      14.7| Credit Card|        23|
# |   1215| 2016-12-12 07:15:00|2016-12-12 07:15:00|         240|       0.4|  5.0| 3.0|       9.5| Credit Card|         7|
# |   3673| 2016-12-16 16:30:00|2016-12-16 17:00:00|        2400|      10.7| 31.0| 0.0|      31.0|        Cash|        16|
# |   5400| 2016-12-16 08:45:00|2016-12-16 09:00:00|         300|       0.0| 5.25| 2.0|      7.25| Credit Card|         8|
# |   1257| 2016-12-03 18:45:00|2016-12-03 18:45:00|         360|       0.3|  5.0| 0.0|       5.0|        Cash|        18|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+
```

```python
after_six = taxi_add.filter((f.col('start_hour') >= 6) & (f.col('start_hour') <= 11))
after_six.count()

# df.groupBy((hour('trip_start_timestamp')).alias('day')).count()
# .where(col('day').between(6,11)).groupBy().sum().show()

# 538737
```

### Найдите топ три даты, в которые было суммарно больше всего чаевых?

<br>Вам может понадобится конвертация типов <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.Column.cast.html?highlight=cast#pyspark-sql-column-cast">cast</a>

```python
from pyspark.sql.types import DateType

# extract date only from type [timestamp] -> [date]
# we use f.to_date ( f.col('column), 'order' )

taxi_add = taxi.withColumn("date",f.to_date(f.col("trip_end_timestamp"),"yyyy-MM-dd"))
taxi_add.show(5)

# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles| fare|tips|trip_total|payment_type|      date|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+
# |   5240| 2016-12-15 23:45:00|2016-12-16 00:00:00|         900|       2.5|10.75|2.45|      14.7| Credit Card|2016-12-16|
# |   1215| 2016-12-12 07:15:00|2016-12-12 07:15:00|         240|       0.4|  5.0| 3.0|       9.5| Credit Card|2016-12-12|
# |   3673| 2016-12-16 16:30:00|2016-12-16 17:00:00|        2400|      10.7| 31.0| 0.0|      31.0|        Cash|2016-12-16|
# |   5400| 2016-12-16 08:45:00|2016-12-16 09:00:00|         300|       0.0| 5.25| 2.0|      7.25| Credit Card|2016-12-16|
# |   1257| 2016-12-03 18:45:00|2016-12-03 18:45:00|         360|       0.3|  5.0| 0.0|       5.0|        Cash|2016-12-03|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+
```

```python
# convert/case data type date to string

# taxi_add.select(taxi_add.date.cast("string").alias('date_str')).show() # don't do this
taxi_add = taxi_add.withColumn('date_str',f.col('date').cast("string"))
taxi_add.show(5)

# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+----------+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles| fare|tips|trip_total|payment_type|      date|  date_str|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+----------+
# |   5240| 2016-12-15 23:45:00|2016-12-16 00:00:00|         900|       2.5|10.75|2.45|      14.7| Credit Card|2016-12-16|2016-12-16|
# |   1215| 2016-12-12 07:15:00|2016-12-12 07:15:00|         240|       0.4|  5.0| 3.0|       9.5| Credit Card|2016-12-12|2016-12-12|
# |   3673| 2016-12-16 16:30:00|2016-12-16 17:00:00|        2400|      10.7| 31.0| 0.0|      31.0|        Cash|2016-12-16|2016-12-16|
# |   5400| 2016-12-16 08:45:00|2016-12-16 09:00:00|         300|       0.0| 5.25| 2.0|      7.25| Credit Card|2016-12-16|2016-12-16|
# |   1257| 2016-12-03 18:45:00|2016-12-03 18:45:00|         360|       0.3|  5.0| 0.0|       5.0|        Cash|2016-12-03|2016-12-03|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+----------+----------+
```

```python
# now use groupby date (which is a string) & find aggregate sum of all trips
taxi_add.groupby(['date_str']).agg(f.sum('tips').alias('total')).orderBy('total',ascending=False).show(3)

# +----------+------------------+
# |  date_str|             total|
# +----------+------------------+
# |2016-11-03|110102.37000000013|
# |2016-11-09|106187.87999999986|
# |2016-11-16| 99993.77000000038|
# +----------+------------------+
```

### Сколько было заказов в дату с наибольшим спросом?

```python
# we need to use starting time

taxi_add = taxi.withColumn("date",f.to_date(f.col("trip_start_timestamp"),"yyyy-MM-dd"))
taxi_add = taxi_add.withColumn('date_str',f.col('date').cast("string"))
taxi_add.groupby(['date_str']).count().orderBy('count',ascending=False).show(5)

# +----------+-----+
# |  date_str|count|
# +----------+-----+
# |2016-11-03|61259|
# |2016-12-16|59137|
# |2016-12-09|58583|
# |2016-12-15|57084|
# |2016-11-04|56800|
# +----------+-----+

```

Подгрузите данные о марках машин из датасета <a href="https://github.com/AlexKbit/stepik-ds-course/raw/master/Week3/spark-tasks/taxi_cars_data.parquet">taxi_cars_data.parquet</a>

```python
# load taxi_id car_modl type
df_car = spark.read.parquet('taxi_cars_data.parquet')
df_car.show(5)

# +-------+-------------------+
# |taxi_id|          car_model|
# +-------+-------------------+
# |   1159|       Toyota Prius|
# |   7273|Ford Crown Victoria|
# |   2904|        Honda Civic|
# |   3210|        Ford Fusion|
# |   2088|       Toyota Camry|
# +-------+-------------------+
```

### Какая марка машины самая распрастранненая среди таксистов?

<br>Подробнее <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.functions.split.html?highlight=split#pyspark-sql-functions-split">split</a>

Мы можем разделить колонку по некому критерии (delimiter), добавим данные в новою колонку `newest`

```python
# we are given the name and model; we need to extract the maker
# like in pandas (.str.split())

df_car.withColumn('newest',f.split(f.col('car_model')," "))
df_car.show(5)

# +-------+-------------------+--------------------+
# |taxi_id|          car_model|              newest|
# +-------+-------------------+--------------------+
# |   1159|       Toyota Prius|     [Toyota, Prius]|
# |   7273|Ford Crown Victoria|[Ford, Crown, Vic...|
# |   2904|        Honda Civic|      [Honda, Civic]|
# |   3210|        Ford Fusion|      [Ford, Fusion]|
# |   2088|       Toyota Camry|     [Toyota, Camry]|
# +-------+-------------------+--------------------+
```

```python
# add columns with each split
df_car_add = df_car.withColumn('maker', f.split(f.col('car_model'), ' ').getItem(0)) \
                   .withColumn('model', f.split(f.col('car_model'), ' ').getItem(1))
df_car_add.show(5)

# +-------+-------------------+------+------+
# |taxi_id|          car_model| maker| model|
# +-------+-------------------+------+------+
# |   1159|       Toyota Prius|Toyota| Prius|
# |   7273|Ford Crown Victoria|  Ford| Crown|
# |   2904|        Honda Civic| Honda| Civic|
# |   3210|        Ford Fusion|  Ford|Fusion|
# |   2088|       Toyota Camry|Toyota| Camry|
# +-------+-------------------+------+------+
```

```python
df_car_add.groupby('maker').agg(f.count('maker').alias('total')).orderBy('total',ascending=False).show(5)

# +---------+-----+
# |    maker|total|
# +---------+-----+
# |     Ford| 1484|
# |  Hyundai|  792|
# |   Toyota|  691|
# |Chevrolet|  473|
# |      Kia|  265|
# +---------+-----+
```

### Сколько раз и какая модель машин чаще всего встречается в поездках?

<br>Подробнее <a href="https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.functions.split.html?highlight=split#pyspark-sql-functions-split">join</a>

```python
taxi_add = taxi.join(df_car_add,taxi.taxi_id == df_car_add.taxi_id)
taxi_add.show(5)

# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+-------+-------------------+-------+-------+
# |taxi_id|trip_start_timestamp| trip_end_timestamp|trip_seconds|trip_miles| fare|tips|trip_total|payment_type|taxi_id|          car_model|  maker|  model|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+-------+-------------------+-------+-------+
# |   5240| 2016-12-15 23:45:00|2016-12-16 00:00:00|         900|       2.5|10.75|2.45|      14.7| Credit Card|   5240|     Toyota Corolla| Toyota|Corolla|
# |   1215| 2016-12-12 07:15:00|2016-12-12 07:15:00|         240|       0.4|  5.0| 3.0|       9.5| Credit Card|   1215|Ford Crown Victoria|   Ford|  Crown|
# |   3673| 2016-12-16 16:30:00|2016-12-16 17:00:00|        2400|      10.7| 31.0| 0.0|      31.0|        Cash|   3673|        Ford Taurus|   Ford| Taurus|
# |   5400| 2016-12-16 08:45:00|2016-12-16 09:00:00|         300|       0.0| 5.25| 2.0|      7.25| Credit Card|   5400|     Toyota Corolla| Toyota|Corolla|
# |   1257| 2016-12-03 18:45:00|2016-12-03 18:45:00|         360|       0.3|  5.0| 0.0|       5.0|        Cash|   1257|    Hyundai Elantra|Hyundai|Elantra|
# +-------+--------------------+-------------------+------------+----------+-----+----+----------+------------+-------+-------------------+-------+-------+
```

```python
taxi_add.groupBy('car_model').agg(f.count('car_model').alias('orders')).orderBy('orders',ascending=False).show(5)

# +-------------------+------+
# |          car_model|orders|
# +-------------------+------+
# |Ford Crown Victoria|388682|
# |     Hyundai Accent|150764|
# |           Kia Ceed|143649|
# |     Hyundai Sonata|141570|
# |        Ford Mondeo|135466|
# +-------------------+------+
```

## Выводы

Кратко о том что мы использовали для разведовательного анализа

- Сорреляция и ковариация `.corr('column A','column B')`
- Как и в pandas есть метод `.descibe()` 
- Группировать данные можно с `.groupBy` и сортировать `.orderBy`
  - taxi**.groupBy**('payment_type').count()**.orderBy**('count', ascending=True).show()
- Фильтрация данных (подвыборка) с `.filter` или `.where` (как в SQL)
  - cash = taxi**.filter**(f.col('payment_type') == 'Cash').show(1)
  - Так же можно отметить выбор колонок с `f.col('column')`
- functions так же имеет функции для агрегации (eg. f.mean, f.max, f.min, f.sum)
  - taxi.groupBy(['payment_type']).agg(**f.mean**('trip_total')).show()
  - добавляем alias для новой колонке `.agg(f.sum('trip_miles').alias('name'))`
- Есть taxi.dropna() как и в pandas
  - taxi_cleaned = taxi**.dropna**(**how='all'**,**subset=('taxi_id')**)
- Добавляем колонку в действующий dataframe
  - taxi**.withColumn**('hour',f.hour('trip_start_timestamp'))
  - **f.hour** из datetime выводит отдельно количество частов
  - **f.to_date** из datetime выводит отдельно дату f**.to_date**(f.col("trip_end_timestamp"),**"yyyy-MM-dd"**)
- Меняем тип данных в колонке с `f.cast`
  - taxi_add.withColumn('date_str',f.col('date')**.cast("string")**)
- Разбиваем string с `f.split`
  - df_car.withColumn('newest',**f.split**(f.col('car_model'),**" "**)).show(5)
- `join` для таблиц
  - df1**.join**(df2,df1.taxi_id == df2.taxi_id)

