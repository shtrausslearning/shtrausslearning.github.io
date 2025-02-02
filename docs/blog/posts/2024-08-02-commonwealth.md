---
date: 2024-08-02
title: Commonwealth Bank Internship
authors: [andrey]
draft: true
categories:
     - internship
tags:
     - financial analysis
     - internship
     - commonwealth bank
comments: true
---

# Commonwealth Bank Internship

In today's we'll be going through one of internships provided by one of the leading banks in Australia; **Commonwealth Bank**. The internship is related to **Data Science** and is aimed to be a small project and interesting project in one's portfolio. The internship covers topics such a data wrangling (rearrangement and exploration of data), data annonymisation, as well as work with unstructured text data. Hopefully it will be interesting to read along!

<div class="grid cards" markdown>

  - :simple-kaggle:{ .lg .middle }&nbsp; <b>[Kaggle Dataset](https://www.kaggle.com/datasets/shtrausslearning/forage-internship-data)</b>
- :material-bank-check:{ .lg .middle }&nbsp; <b>[Commbank Internship](https://www.theforage.com/simulations/commonwealth-bank/intro-data-science-sd7t)</b>

</div>

<!-- more -->

The internship is split into a few segments:

- ==**Task One**== : Data Aggregation and Analysis
- ==**Task Two**== : Data Anonymisation 
- ==**Task Three**== : Propose Data Analysis Approaches
- ==**Task Four**== : Designing a Database

In this post, we'll cover only the first two parts, the rest your can try for yourself **[@Introduction to Data Science](https://www.theforage.com/simulations/commonwealth-bank/intro-data-science-sd7t)**

## **Data Aggregation and Analysis**

Let's look at the first topic which covers some data insight analysis.

### <span style='color:#686dec'>Problem Background </span>

Create a data engineering pipeline to aggregate and extract insights from a data set
What you'll learn

<div class="grid cards" markdown>

- Understand the importance of data engineering and analysis in a long-term data-focused vision.
- Gain experience in using real-world transactional data for analysis.
- Learn to perform data analysis and answer specific questions based on the dataset provided.

</div>

What you'll do:

<div class="grid cards" markdown>

- Analyze the **["CSV Data Set"](https://github.com/shtrausslearning/Data-Science-Portfolio/blob/main/Commonwealth%20Bank%20Internship/supermarket_transactions.csv)** of supermarket transactions to address key questions.
- Document the analysis process, including formulas used, using spreadsheet software.
Submit the analyzed file for review.

</div>

### <span style='color:#686dec'>Problem Statement</span>

Let's take a look at what is asked of us:

    Hello,

    I have provided you with a data set named “supermarket_transactions.csv.” This data set contains three years of transactional data we collected from supermarkets across Australia. 

    InsightSpark wants to access our data to perform some analysis across Australian supermarkets. Before this can happen, I need you to perform some analysis on the provided CSV data set (a sample of the database) to answer the following questions:

    - Across locations, how many apples were purchased in cash?
    - How much total cash was spent on these apples?
    - Across all payment methods, how much money was spent at the Bakershire store location by non-member customers?

### <span style='color:#686dec'>Dataset Preview</span>

Quite a simple task at hand; we are given a set of transactional data (of course downscaled) which we first need to take a quick look at in order to understand what we're actually have. The first two problems are associated with a single product, so we'll grab a sample of that dataset:

```python
apple_transactions = transactions[transactions['product_name'] == 'apple']
apple_transactions.head()
```

```
+-------+-------+--------------------------------------+---------------------+----------+--------------------------------------+--------------+------------+--------------+-----------------+----------------+--------------------------------------+---------------+
|       | index |                  id                  |      timestamp      | quantity |              product_id              | product_name | unit_price | total_amount |      store      | payment_method |             customer_id              | customer_type |
+-------+-------+--------------------------------------+---------------------+----------+--------------------------------------+--------------+------------+--------------+-----------------+----------------+--------------------------------------+---------------+
| 23742 | 23742 | 619db75d-bc5d-40b9-acf8-93476d29a36b | 2021-12-15 17:56:00 |    10    | 34bc5a93-7c74-464b-94bb-f9b31f0edca5 |    apple     |    4.59    |     45.9     | South Billyview |      cash      | 9de2834c-9a6b-4682-b314-8ec31063ee06 |    member     |
| 23743 | 23743 | bbc7a9e8-c272-431d-9703-9c3724eab634 | 2021-04-04 16:51:00 |    8     | 34bc5a93-7c74-464b-94bb-f9b31f0edca5 |    apple     |    4.59    |    36.72     |  West Stefanie  |  credit card   | 7fcc7661-c7ef-4ac2-b7de-e6479930a414 |  non-member   |
| 23744 | 23744 | 2f9b0c56-b281-4b42-b103-95f2bbd51586 | 2021-03-29 12:07:00 |    9     | 34bc5a93-7c74-464b-94bb-f9b31f0edca5 |    apple     |    4.59    |    41.31     |   Andreburgh    |  contactless   | 0e86f2a0-75db-4480-bf9b-c7726f76b6a0 |    member     |
| 23745 | 23745 | e2fe4f8e-421c-4c72-9310-b5a17cc2fcd4 | 2019-10-10 17:39:00 |    9     | 34bc5a93-7c74-464b-94bb-f9b31f0edca5 |    apple     |    4.59    |    41.31     |   Anthonyton    |   debit card   | 71d16b34-726d-4097-b65e-78d0defcbf4b |    premium    |
| 23746 | 23746 | 4e8c369b-fe2d-42fa-8dcb-2484c4529ded | 2019-10-09 17:35:00 |    4     | 34bc5a93-7c74-464b-94bb-f9b31f0edca5 |    apple     |    4.59    |    18.36     |  West Stefanie  |      cash      | 247e3a5a-8a7a-4c81-80a6-a58145e70621 |     gold      |
+-------+-------+--------------------------------------+---------------------+----------+--------------------------------------+--------------+------------+--------------+-----------------+----------------+--------------------------------------+---------------+
```

### <span style='color:#686dec'>Data Column Explanations</span>

Lets try to summarise what the data contains:

- `id` : Transaction unique identifier
- `timestamp` : Time of the transaction 
- `quantity` : Number of items purchased of the product `product_id`
- `product_name` : Name of the product
- `unit_price` : Price for each item
- `total_amount` : The total ammount for the purchase
- `store` : Location where the item was purchased
- `payment_method` : The method of payment
- `customer_id` : Unique customer identification number
- `customer_type` : Customer type 

### Transaction / Purchase Format

Transactions can be stored in data in a number of ways, we can see that our dataset has a rather simple structure, each customer (customer_id) has made a single product purchase (product_id), which is represented by its unique transaction identifier (id), nothing too fancy.

```python
transactions['customer_id'].unique().shape
(50783,)
```

```python
transactions['id'].unique().shape
(50783,)
```

### <span style='color:#686dec'>Across locations, how many apples were purchased in cash?</span>

The first request which is asked of us is to extract the number of purchases of apples for each location. We can do this by using the `pivot_table` method. Since we are interested in the amount, we need to make sure we select quantity for **values** and set aggfunc to **sum**

```python
pd.pivot_table(apple_transactions,index='store',columns='payment_method',values='quantity',aggfunc='sum',fill_value=0,margins=True).sort_values(by='cash',ascending=False)
```

```
+---------------------+------+-------------+-------------+------------+-----+
|        store        | cash | contactless | credit card | debit card | All |
+---------------------+------+-------------+-------------+------------+-----+
|         All         | 117  |     104     |     138     |     88     | 447 |
|    South Cynthia    |  18  |      0      |      0      |     0      | 18  |
|     Swansonfurt     |  17  |      0      |     10      |     0      | 27  |
|   South Billyview   |  12  |      0      |      0      |     0      | 12  |
|  East Suzanneside   |  10  |      0      |      0      |     0      | 10  |
|  South Michaelfurt  |  10  |      0      |      0      |     0      | 10  |
|    South Edward     |  8   |      2      |      0      |     0      | 10  |
|     Lake Bryan      |  8   |      9      |      0      |     0      | 17  |
|      Tracyton       |  7   |      0      |      0      |     0      |  7  |
|      Erichaven      |  7   |      0      |     10      |     0      | 17  |
|      Alexmouth      |  6   |      0      |      0      |     0      |  6  |
|    West Stefanie    |  4   |      0      |     13      |     0      | 17  |
|     Jordanmouth     |  4   |      8      |      0      |     0      | 12  |
|      Julieview      |  3   |      0      |      1      |     0      |  4  |
|     Port Angela     |  2   |      0      |      8      |     4      | 14  |
|    Andersonland     |  1   |      0      |     10      |     8      | 19  |
|   Christopherfurt   |  0   |      2      |      5      |     5      | 12  |
| South Christineside |  0   |      0      |      0      |     8      |  8  |
|     Charlesbury     |  0   |     10      |      0      |     6      | 16  |
|      Aprilside      |  0   |      7      |      0      |     0      |  7  |
|     Anthonyton      |  0   |      0      |      0      |     11     | 11  |
|      East Sara      |  0   |      0      |     11      |     5      | 16  |
|  South Rachaelport  |  0   |      0      |      1      |     3      |  4  |
|    South Alyssa     |  0   |      2      |      0      |     0      |  2  |
|     Andreburgh      |  0   |      9      |      0      |     2      | 11  |
|    Vincentville     |  0   |      0      |      9      |     0      |  9  |
|      West John      |  0   |      7      |      0      |     2      |  9  |
|    Anthonymouth     |  0   |      0      |     17      |     3      | 20  |
|    North Joyfort    |  0   |      8      |      0      |     0      |  8  |
|   Port Emilymouth   |  0   |      8      |      0      |     0      |  8  |
|   East Jeremytown   |  0   |      0      |      0      |     1      |  1  |
|    North Charles    |  0   |      0      |      2      |     0      |  2  |
|      New Lisa       |  0   |      0      |      0      |     7      |  7  |
|      New Glenn      |  0   |      0      |     10      |     0      | 10  |
|      New Eric       |  0   |     16      |      5      |     1      | 22  |
|      East Ann       |  0   |      6      |      0      |     0      |  6  |
|     Justinstad      |  0   |      0      |      0      |     4      |  4  |
|     Jessicafort     |  0   |     10      |      4      |     0      | 14  |
|      Jaredside      |  0   |      0      |      0      |     4      |  4  |
|      Irwinport      |  0   |      0      |     16      |     0      | 16  |
|   East Candiceton   |  0   |      0      |      0      |     7      |  7  |
|    Michelemouth     |  0   |      0      |      6      |     7      | 13  |
+---------------------+------+-------------+-------------+------------+-----+

```

### <span style='color:#686dec'>How much total cash was spent on these apples?</span>

The next question is pretty much the same, except we set the column to **total_amount**, since we are interested in the total ammount spend on this item across all transaction purchases.

```python
df = pd.pivot_table(apple_transactions,index='store',columns='payment_method',values='total_amount',aggfunc='sum',fill_value=0,margins=True).sort_values(by='cash',ascending=False).round(2)
```

```
+---------------------+--------+-------------+-------------+------------+---------+
|        store        |  cash  | contactless | credit card | debit card |   All   |
+---------------------+--------+-------------+-------------+------------+---------+
|         All         | 537.03 |   477.36    |   633.42    |   403.92   | 2051.73 |
|    South Cynthia    | 82.62  |     0.0     |     0.0     |    0.0     |  82.62  |
|     Swansonfurt     | 78.03  |     0.0     |    45.9     |    0.0     | 123.93  |
|   South Billyview   | 55.08  |     0.0     |     0.0     |    0.0     |  55.08  |
|  South Michaelfurt  |  45.9  |     0.0     |     0.0     |    0.0     |  45.9   |
|  East Suzanneside   |  45.9  |     0.0     |     0.0     |    0.0     |  45.9   |
|    South Edward     | 36.72  |    9.18     |     0.0     |    0.0     |  45.9   |
|     Lake Bryan      | 36.72  |    41.31    |     0.0     |    0.0     |  78.03  |
|      Tracyton       | 32.13  |     0.0     |     0.0     |    0.0     |  32.13  |
|      Erichaven      | 32.13  |     0.0     |    45.9     |    0.0     |  78.03  |
|      Alexmouth      | 27.54  |     0.0     |     0.0     |    0.0     |  27.54  |
|    West Stefanie    | 18.36  |     0.0     |    59.67    |    0.0     |  78.03  |
|     Jordanmouth     | 18.36  |    36.72    |     0.0     |    0.0     |  55.08  |
|      Julieview      | 13.77  |     0.0     |    4.59     |    0.0     |  18.36  |
|     Port Angela     |  9.18  |     0.0     |    36.72    |   18.36    |  64.26  |
|    Andersonland     |  4.59  |     0.0     |    45.9     |   36.72    |  87.21  |
|   Christopherfurt   |  0.0   |    9.18     |    22.95    |   22.95    |  55.08  |
| South Christineside |  0.0   |     0.0     |     0.0     |   36.72    |  36.72  |
|     Charlesbury     |  0.0   |    45.9     |     0.0     |   27.54    |  73.44  |
|      Aprilside      |  0.0   |    32.13    |     0.0     |    0.0     |  32.13  |
|     Anthonyton      |  0.0   |     0.0     |     0.0     |   50.49    |  50.49  |
|      East Sara      |  0.0   |     0.0     |    50.49    |   22.95    |  73.44  |
|  South Rachaelport  |  0.0   |     0.0     |    4.59     |   13.77    |  18.36  |
|    South Alyssa     |  0.0   |    9.18     |     0.0     |    0.0     |  9.18   |
|     Andreburgh      |  0.0   |    41.31    |     0.0     |    9.18    |  50.49  |
|    Vincentville     |  0.0   |     0.0     |    41.31    |    0.0     |  41.31  |
|      West John      |  0.0   |    32.13    |     0.0     |    9.18    |  41.31  |
|    Anthonymouth     |  0.0   |     0.0     |    78.03    |   13.77    |  91.8   |
|    North Joyfort    |  0.0   |    36.72    |     0.0     |    0.0     |  36.72  |
|   Port Emilymouth   |  0.0   |    36.72    |     0.0     |    0.0     |  36.72  |
|   East Jeremytown   |  0.0   |     0.0     |     0.0     |    4.59    |  4.59   |
|    North Charles    |  0.0   |     0.0     |    9.18     |    0.0     |  9.18   |
|      New Lisa       |  0.0   |     0.0     |     0.0     |   32.13    |  32.13  |
|      New Glenn      |  0.0   |     0.0     |    45.9     |    0.0     |  45.9   |
|      New Eric       |  0.0   |    73.44    |    22.95    |    4.59    | 100.98  |
|      East Ann       |  0.0   |    27.54    |     0.0     |    0.0     |  27.54  |
|     Justinstad      |  0.0   |     0.0     |     0.0     |   18.36    |  18.36  |
|     Jessicafort     |  0.0   |    45.9     |    18.36    |    0.0     |  64.26  |
|      Jaredside      |  0.0   |     0.0     |     0.0     |   18.36    |  18.36  |
|      Irwinport      |  0.0   |     0.0     |    73.44    |    0.0     |  73.44  |
|   East Candiceton   |  0.0   |     0.0     |     0.0     |   32.13    |  32.13  |
|    Michelemouth     |  0.0   |     0.0     |    27.54    |   32.13    |  59.67  |
+---------------------+--------+-------------+-------------+------------+---------+
```

### <span style='color:#686dec'>Across all payment methods, how much money was spent at the Bakershire store location by non-member customers?</span>

Next up, we need to verify what the column name is for non members, so we can select the appropriate subset of data and evaluate how much was spent in the particular store. We can see that the specified name does match that found in the dataset.

```python
transactions['customer_type'].value_counts()
```

```
customer_type
non-member    8650
corporate     8536
premium       8427
staff         8397
member        8395
gold          8378
Name: count, dtype: int64
```

Next, we select the subset of interest using two condition that need to be simultaneously met, this gives us 179 transactions for which we will find the transaction sum.

```python
bakershire_nonmembers = transactions[(transactions['store'] == 'Bakershire') & (transactions['customer_type'] == 'non-member')]
bakershire_nonmembers.shape
(179, 12)
```

And we calculate the total income from all the transactions

```python
bakershire_nonmembers['total_amount'].sum().round(3)
2857.51
```

So that wasn't too difficult! The answers are provided below:

??? success

    Example Answer Explanation

    The example answer is just one way to approach this problem, not the only solution. For this question, first, we opened the CSV file `supermarket_transactions.csv` in Excel. Then we aggregated and analysed the data using filters and formulas to answer the following questions:

    Question 1: Across locations, how many apples were purchased in cash?

    To answer this question, we filtered the data sheet to include only rows where the
    `product_name` is “apple” and where the `payment_method` is “cash.” Then, we summed the
    `quantity` column to get an answer of 117 apples.

    Question 2: How much total cash was spent on these apples?

    Here, we left the data sheet filtered to include only rows where the `product_name` is “apple” and where the `payment_method` is “cash.” Then, we summed the `total_amount` column to get an answer of $537.03.

    Question 3: Across all payment methods, how much money was spent at the Bakershire store location by non-member customers?

    For this question, we cleared all the previous filter criteria. Then, we filtered the data sheet to include only rows where the `customer_type` was non-member, and the `store` was Bakershire. Then, we summed the `total_amount` column to get an answer of $2,857.51.

<br>

## **Data Anonymisation**

Now let's look at the second topic, which involves the use of data anonimisation.

### <span style='color:#686dec'>Problem Background </span>

<div class="grid cards" markdown>

- Understand the importance of data anonymization in protecting sensitive customer information.
- Learn various techniques for anonymizing data while preserving its utility.
- Comprehend the risks associated with linkage attacks on anonymized data.

</div>

What you'll do:

<div class="grid cards" markdown>

- Anonymize the provided data set, ensuring the protection of personal details.
- Utilize anonymization techniques such as removing unnecessary columns, masking identifying information, and categorizing sensitive figures.
- Submit the anonymized data as a CSV file.

</div>

### <span style='color:#686dec'>Problem Statement</span>

> I have provided you with a data set named “mobile_customers.csv.” This contains information about customers that have signed up for the mobile app in the last three years.

> We need you to anonymise this data to hide personal details while preserving any useful information for the data scientists at InsightSpark.

> Here are some examples of how you may anonymise a data set:
> You could remove columns that don’t provide helpful information for analysis (e.g., names or credit card numbers).
> You could mask any columns that can identify an individual (e.g., passport numbers or mobile numbers).
> You could categorise personal figures (e.g., age and income) into a bracket rather than a specific number.

> First, research the different techniques available for anonymising a data set. Then, edit the data set to create an anonymised data set as a CSV file. When finished, please submit this CSV file for me to review before we share it with InsightSpark.


### <span style='color:#686dec'>Data Anonymisation approaches</span>

We don't want customer data to leak out, nevertheless there could be various applications of customer data usage; from analyses to modeling. Generally speaking, there is a standard of data privacy "criticalness", and a subsystem in a company operate with data corresponding to a specific level, so customer data doesn't tend to freeflow through all employees in a company. The process of data protection is probably done by a specific department in charge of data protection, or something along those lines. 

The data transfer target are **data scientists at InsightSpark**, so we should probably put some logic into our process of transformation, since the data science team will probably want to extract useful relations from the data, without really needing to know the exact details about the customers. So this means that ==**pure randomisation**== data replacement will be rather pointless, yet its a way we can anonymise the data. 

Other ways can be:

- ==**Binning**== (grouping into segments)
- ==**Masking**== (hiding only parts of the data)
- ==**Replacement**== (replace the unique values in data)
- ==**Perturbation**== (for numerical values)

### <span style='color:#686dec'>Data Column Explanations</span>

Lets read our dataset to get a glimpse of our dataset

```python
customers = pd.read_csv('/kaggle/input/commonwealth-bank-internship-data/mobile_customers.csv')
customers.head()
```

```
+---+--------------------------------------+-----------------+-----------------+-----------------+--------+---------------------------+----------------------------+------------+-------------------------------+----------------------------+-----------------------------+--------------------------+-----+--------+----------------------+---------------------+---------------------------+--------------------+
|   |             customer_id              | date_registered |    username     |      name       | gender |          address          |           email            | birthdate  |       current_location        |         residence          |          employer           |           job            | age | salary | credit_card_provider | credit_card_number  | credit_card_security_code | credit_card_expire |
+---+--------------------------------------+-----------------+-----------------+-----------------+--------+---------------------------+----------------------------+------------+-------------------------------+----------------------------+-----------------------------+--------------------------+-----+--------+----------------------+---------------------+---------------------------+--------------------+
| 0 | 24c9d2d0-d0d3-4a90-9a3a-e00e4aac99bd |   2021-09-29    |  robertsbryan   | Jonathan Snyder |   M    |    24675 Susan Valley     |    marcus58@hotmail.com    | 1978-03-11 |  ['78.937112', '71.260464']   |    195 Brandi Junctions    |    Byrd, Welch and Holt     | Chief Technology Officer | 49  | 53979  |    VISA 19 digit     |   38985874269846    |            994            |       10/27        |
|   |                                      |                 |                 |                 |        | North Dianabury, MO 02475 |                            |            |                               |  New Julieberg, NE 63410   |                             |                          |     |        |                      |                     |                           |                    |
| 1 | 7b2bc220-0296-4914-ba46-d6cc6a55a62a |   2019-08-17    |     egarcia     | Susan Dominguez |   F    |     4212 Cheryl Inlet     | alexanderkathy@hotmail.com | 1970-11-29 | ['-24.1692185', '100.746122'] | 58272 Brown Isle Apt. 698  |          Hurst PLC          |      Data scientist      | 43  | 81510  |       Discover       |  6525743622515979   |            163            |       07/30        |
|   |                                      |                 |                 |                 |        | Port Davidmouth, NC 54884 |                            |            |                               |   Port Michael, HI 04693   |                             |                          |     |        |                      |                     |                           |                    |
| 2 | 06febdf9-07fb-4a1b-87d7-a5f97d9a5faf |   2019-11-01    |   turnermegan   |  Corey Hebert   |   M    |   07388 Coleman Prairie   |      vwood@gmail.com       | 2009-04-23 |  ['8.019908', '-19.603269']   | 36848 Jones Lane Suite 282 | Mora, Caldwell and Guerrero | Chief Operating Officer  | 47  | 205345 |    VISA 16 digit     | 4010729915028682247 |            634            |       04/26        |
|   |                                      |                 |                 |                 |        |    Lake Amy, IA 78695     |                            |            |                               |   Marquezbury, ID 26822    |                             |                          |     |        |                      |                     |                           |                    |
| 3 | 23df88e5-5dd3-46af-ac0d-0c6bd92e4b96 |   2021-12-31    | richardcampbell | Latasha Griffin |   F    |    PSC 6217, Box 2610     |    kathleen36@gmail.com    | 1992-07-27 |   ['62.497506', '2.717198']   |   317 Lamb Cape Apt. 884   |          Patel PLC          | Counselling psychologist | 34  | 116095 |    VISA 16 digit     | 4854862659569207844 |           7957            |       10/31        |
|   |                                      |                 |                 |                 |        |       APO AA 53585        |                            |            |                               |     Lake Amy, DC 79074     |                             |                          |     |        |                      |                     |                           |                    |
| 4 | 6069c2d7-7905-4993-a155-64f6aba143b1 |   2020-08-09    | timothyjackson  | Colleen Wheeler |   F    |     0325 Potter Roads     |    johnbest@hotmail.com    | 1989-09-16 | ['73.7924695', '-80.314720']  |     21936 Mary Islands     |         Smith-Mejia         |     Mining engineer      | 57  | 107529 |     JCB 16 digit     |   213152724828217   |            72             |       05/28        |
|   |                                      |                 |                 |                 |        | Lake Lisashire, NM 77502  |                            |            |                               |   Mendozafort, TN 37124    |                             |                          |     |        |                      |                     |                           |                    |
+---+--------------------------------------+-----------------+-----------------+-----------------+--------+---------------------------+----------------------------+------------+-------------------------------+----------------------------+-----------------------------+--------------------------+-----+--------+----------------------+---------------------+---------------------------+--------------------+
```

Lets describe the columns that we have:

- `customer_id` : Identification assigned to customer
- `date_registered` : The data of registration ==perturbation==
- `username` : The username the customer uses 
- `name` : The customers' name
- `gender` : Gender of the customer
- `address` : Customers' address
- `email` : Customers' email ==mask==
- `birthday` : Date of bird of customer ==perturbation==
- `current_location` : Location of the customer  ==remove==
- `residence` : Residence of the customer
- `employer` : Employer of the customer ==replace==
- `age` : Age of the customer ==bin==
- `salary` : Customers' salary ==bin==
- `credit_card_provider` : Customers' credit card provider
- `credit_card_number` : The customers' credit card number
- `credit_card_security_code` : Customers' credit card security code
- `credit_card_expire` : Expiration data of the customers' credit card  

Let's start with something simple; binning. Both **<span style='color:#686dec'>Age</span>** and ****<span style='color:#686dec'>Salary</span>** can definitely be categorised, not providing the exact number, but still keeping enough information so some insights about the customers can be made.

```python
bins = [0, 18, 30, 45, 60, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '60+']
customers['age'] = pd.cut(customers['age'], bins=bins, labels=labels)
customers['age'].value_counts()
```

```
age
31-45    3122
46-60    3115
19-30    2504
60+      1070
0-18      189
Name: count, dtype: int64
```

```python
bins = [0,79000,119000,169000,200000, 300000]
labels = ['0-79K', '80-119K', '120-169K', '170-200K', '200K+']
customers['salary'] = pd.cut(customers['salary'], bins=bins, labels=labels)
customers['salary'].value_counts()
```

```
salary
0-79K       2551
120-169K    2230
200K+       2031
80-119K     1783
170-200K    1405
Name: count, dtype: int64
```

We will also remove column **<span style='color:#686dec'>current_location</span>**, since it doesn't really hold any value outside perhaps tracking and doesn't offer what **<span style='color:#686dec'>residence</span>** doesn't already offer.


**<span style='color:#686dec'>Employer</span>** can be a useful feature, we can see that many customers share the same employers, yet there is no real need to reveal the exact employer, so we can simply replace the unique column names with randomly generated text.

```python
customers['employer'].value_counts()
```

```
employer
Smith and Sons                  18
Smith Inc                       15
Smith PLC                       11
Johnson Group                   11
Johnson Ltd                     11
                                ..
Hicks, Ramirez and Taylor        1
Moreno, Mccarthy and Donovan     1
Klein-Foley                      1
Watson-Watson                    1
Warner, Munoz and Franklin       1
Name: count, Length: 8645, dtype: int64
```

So what we'll do is extract the unique values, create a mapping dictionary for it, generate random values and text, concatenate it and use the **map** method to replace the column values

```python
import random
import string

unique_employer = list(customers['employer'].unique())
unique_employer_mapper = {i:None for i in unique_employer}

def generate_random_name():
    digits = "".join( [random.choice(string.digits) for i in range(20)] )
    chars = "".join( [random.choice(string.ascii_letters) for i in range(20)] )
    return digits + chars

for employer in unique_employer:
    unique_employer_mapper[employer] = generate_random_name()

customers['employer'].map(unique_employer_mapper).value_counts()
```

```
employer
58604887536751648622FbvCGfRGBtnDGLWldyRF    18
18629626800688749605NmGJKOfhUhLunfJBxgOT    15
93505473119626534727ZBuGpghisKSeKttAWFBj    11
67338527509968294563wdJIoNdzKxpDAlhQYWqS    11
32329694609232288444HgflONofgmdPExwgiYPu    11
                                            ..
42630313409469792959ODugkraqZKMZhuvUzsHK     1
94222944269364035894tWTTBJxWfHBIdGfAOlED     1
55826701092383385383OpGYMhvZiWMmgZssUYmg     1
37074720113318535258XlwfPhjwwIDLFlNlkgJI     1
17949949387540722650PQkBoWfJTiCNVOSWzpAI     1
Name: count, Length: 8645, dtype: int64
```

**<span style='color:#686dec'>Email</span>** can also contain useful information; whilst the user name does contain important information to the customer, the provider can reveal some patterns between the users of the same provider, so what oftern is done is a mask is placed in between the second and secondlast letter of the username part of the email address; so that's what we'll do as well! We'll use the regular expression library, which should be fun!

The pattern that we'll utilise is as follows:

- The first character of the username (^[a-zA-Z0-9]{1})
- The middle part of the username that we want to mask (.*)
- The last character of the username followed by the domain ([a-zA-Z0-9]{1}@.*$)

```python
import re

def mask_email_username(email):
    
    # Regular expression to match the username part of the email
    pattern = r'(^[a-zA-Z0-9]{1})(.*)([a-zA-Z0-9]{1}@.*$)'
    
    # Substitute the middle part with '*' characters
    masked_email = re.sub(pattern, lambda m: f"{m.group(1)}{'*' * (len(m.group(2)))}{m.group(3)}", email)
    
    return masked_email
```

And with pandas apply & lambda functions we'll apply the function to all rows

```python
customers['email'].apply(lambda x: mask_email_username(x))
```

```
0             m******8@hotmail.com
1       a************y@hotmail.com
2                  v***d@gmail.com
3             k********6@gmail.com
4             j******t@hotmail.com
                   ...            
9995             j*****n@yahoo.com
9996      r************z@gmail.com
9997         b*********a@gmail.com
9998               w***d@gmail.com
9999             n*****s@yahoo.com
Name: email, Length: 10000, dtype: object
```

Another method we can also attempt to utilise is **perturbation**; we can apply it to values or dates as we'll do below. **<span style='color:#686dec'>date_registered</span>** seems to be quite harmless compared to other information about the users, but we might not want to give away the exact details, slight perturbation (addition of noise) probably will allow the data science team to analyse if there are some similarities between users that joined during similar period (month), but we won't be revealing the exact date (so dates probably will be noisy); so lets add/subtract a couple of days from the date. 

```python
import numpy as np

def random_days_adjustment(date):
    # Generate a random integer between 1 and 7
    days = np.random.randint(1, 8)
    # Randomly choose to add or subtract days
    adjustment = np.random.choice([-1, 1]) * days
    return date + pd.Timedelta(days=adjustment)

customers['date_registered'] = pd.to_datetime(customers['date_registered'])
customers['date_registered'].apply(random_days_adjustment)
```

```
0      2021-09-25
1      2019-08-15
2      2019-10-30
3      2021-12-30
4      2020-08-08
          ...    
9995   2021-09-09
9996   2020-08-09
9997   2019-10-29
9998   2020-08-18
9999   2020-03-30
Name: date_registered, Length: 10000, dtype: datetime64[ns]
```

Whilst the above approach may be quite misleading to the recipient, it probably is better to just retain the **year** and month of **birth**.

These are only a few ways, in fact there doesn't exist a right or wrong answer, without any formalisation or implemented standard that we must follow, we could say that we're done! :3 Of course there is still some more sensitive data left, but that's for another time!

The answers are provided below:

??? success

    The example answer is one way to approach this problem and not the only solution. To create the model answer, we built the following steps into a data privacy pipeline:

    - Remove the customer_id and current_location column. Removal of data (redaction) that does not have much informational value is a valid data privacy technique.
    - Mask the username column to hide the real username.
    - Replace the original name column with a fake name. Replacing real values for fake values is a valid data privacy technique.
    - Mask the email column to hide the real email address.
    - Add noise to the date_registered and birthdate columns to hide the real value. Adding noise protects the real value by adding random noise to the actual value.
    - Categorise the salary and age columns into bins. This categorisation hides the original values and preserves the distribution.
    - The credit_card_provider and credit_card_expire have been tokenised. This step converts the categorical value of the columns into a different random value while preserving the distribution.
    - The credit_card_number and credit_card_security_code have been masked.
    - The employer and job columns have also been tokenised to preserve the original distribution.
    - The residence and address have been replaced with fake values.


## **Concluding Remarks**

In this post, we covered a coupled of things; we conducted a simple data analysis of user transactions, in principle, the questions are quite straightforward, and more insight into the dataset may be quite interest, it would have been more interesting to receive data of more realistic transactions. Nevertheless, **pivot_tables** are quite useful to extract useful information from a dataset, so its good to know the useful arguments that the method offers; an example may be the addition of total using **margins** and so on. 

Secondly, we looked at another interesting part; data confidentiality. This is actually quite an important topic, since data needs to be carefully managed, especially when data is passed from the source of storage, to another party. In our case, we were taught that the data is to be provided to the data science team for analysis, so bearing that in mind we need to understand what is relevant to be passed on and what is of no use. 

The internship doesn't end here, there is also another section about unstructured data extraction and analysis, using the Twitter API, which is a fun NLP problem!

The intership problem was quite interesting and lightweight, definitely give it a go! [@Introduction to Data Science](https://www.theforage.com/simulations/commonwealth-bank/intro-data-science-sd7t)

## **Important Takeaways**

Lets recap some things which are worth remembering, as it can help you with your own projects!

### **:material-checkbox-multiple-marked-circle-outline: Pandas Column Binning**

We utilised the pandas method `pd.cut` in order to split the column data into segments, this is an important method, and it is often used in data analysis and machine learning pipelines.

```python
bins = [0,79000,119000,169000,200000, 300000]
labels = ['0-79K', '80-119K', '120-169K', '170-200K', '200K+']
customers['age'] = pd.cut(customers['age'], bins=bins, labels=labels)
```

What the above code will do is simply split the data into parts:
- values between **0-79000** will be assigned a label '0-79k' and so on

```python
pd.cut(
        data, # pandas dataframe column
        bins, # numerical value split location
        labels # labels for each segment
)
```

### **:material-checkbox-multiple-marked-circle-outline: Pandas Column Value Replacement**

We often do need to replace specific values/categories in the pandas column. In the above example, it was for customer data annonymisation, do replace values we can use the `pd.Series.map` method.

Next, we need to have a dictionary in the format:

```python
{'column value':'change it to this'}
```

So when we will call the following, **column value** will be replaced by **change it to this**

```python
customers['employer'].map(unique_employer_mapper)
```

### **:material-checkbox-multiple-marked-circle-outline: Pandas Pivot Table**

Pivoting data and rearranging it is also very important in data analysis, we can understand our data better if we know how to use the `pd.pivot_table`  as we did in the example below:

```python
pd.pivot_table(apple_transactions,index='store',columns='payment_method',values='quantity',aggfunc='sum',fill_value=0,margins=True).sort_values(by='cash',ascending=False)
```

```python
pd.pivot_table(
                data, # our dataframe
                index, # the unique categories in a column will be set as the index
                columns, # the unique categories in a column will be set as the columns
                values, # we will count/sum/... the data in this column
                aggfunc, # the aggregate function (sum,count,...)
                fill_value, # useful to fill in missing data for combinations that dont exist
                margins, # add a "total" column, so we know the row/column sums 
)
```

### **:material-checkbox-multiple-marked-circle-outline:Regular Expressions**

Another approach we used here are **regular expressions**, they are very useful when dealing with **string** data. We'll take a look at the pattern we used in this post; 

```python
pattern = r'(^[a-zA-Z0-9]{1})(.*)([a-zA-Z0-9]{1}@.*$)'
```

We need to focus on three parts:

- 1st Capturing Group (^[a-zA-Z0-9]{1}) : the first letter
- 2nd Capturing Group (.*) : everything in between the first and last letters of the username  
- 3rd Capturing Group ([a-zA-Z0-9]{1}@.*$) : The last letter, then @ and the rest after that 

We do this by using three capture groups. **[a-zA-Z0-9]** implies that the first letter can be either numeric or alphabetical and {1} ensures that we only select the first letter/value. The second capture group captures everything after the first letter (.*), but since we have a third capture group, we once again select a letter/number which is followed by the sign @, this is the capture requirement, and since we place the capture group in third, our second capture group captures everything upto that which is found in the third capture group. After the @ sign we select everything else using the dot (.) and star notation. 

A very helpful resource to play around with regular expressions; [regex101](https://regex101.com/)