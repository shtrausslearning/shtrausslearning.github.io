---
date: 2024-08-10
title: Retail Strategy and Analytics
authors: [andrey]
draft: false
categories:
     - rfm segmentation
tags:
     - financial analysis
     - rfm segmentation
     - internship
comments: true
---

# **Retail Strategy and Analytics**

<div class="grid cards" markdown>

- :simple-kaggle:{ .lg .middle }&nbsp; <b>[Kaggle Dataset](https://www.kaggle.com/datasets/shtrausslearning/forage-internship-data)</b>

- :fontawesome-regular-rectangle-list:{ .lg .middle }&nbsp; <b>[Quantium Internship](https://www.theforage.com/simulations/quantium/data-analytics-rqkb)</b>

</div>

<!-- more -->

The internship is split into a few segments:

- ==**Task 1**== : Data preparation and customer analytics
- ==**Task 2**== : Experimentation and uplift testing
- ==**Task 3**== : Analytics and commercial application

If you are interested in doing the internship, you can **[@Data Analytics](https://www.theforage.com/simulations/quantium/data-analytics-rqkb)**

## **Background**

As part of the internship we will focus on the following things for the first task:

- Analyse transaction and customer data to identify trends and inconsistencies. 
- Develop metrics and examine sales drivers to gain insights into overall sales performance. 
- Create visualizations and prepare findings to formulate a clear recommendation for the client's strategy.

And for the second task we will:

- Define metrics to select control stores.
- Analyse trial stores against controls.
- Use R/Python for data analysis and visualization and summarise findings and provide recommendations.


## **Quantium Dataset Preview**

Lets explore our dataset, to understand the dataset that we are going to be working with, and figure out if there are any preprocessing steps we need to take in order to get the data into a usable for us format.

The data that is provided to us:

- **`QVI_transaction_data`** : client transactional data
- **`QVI_purchase_behaviour`** : client segmentation features

```
+---+-------+-----------+----------------+--------+----------+------------------------------------------+----------+-----------+
|   | DATE  | STORE_NBR | LYLTY_CARD_NBR | TXN_ID | PROD_NBR |                PROD_NAME                 | PROD_QTY | TOT_SALES |
+---+-------+-----------+----------------+--------+----------+------------------------------------------+----------+-----------+
| 0 | 43390 |     1     |      1000      |   1    |    5     |  Natural Chip        Compny SeaSalt175g  |   2.0    |    6.0    |
| 1 | 43599 |     1     |      1307      |  348   |    66    |         CCs Nacho Cheese    175g         |   3.0    |    6.3    |
| 2 | 43605 |     1     |      1343      |  383   |    61    |  Smiths Crinkle Cut  Chips Chicken 170g  |   2.0    |    2.9    |
| 3 | 43329 |     2     |      2373      |  974   |    69    |  Smiths Chip Thinly  S/Cream&Onion 175g  |   5.0    |   15.0    |
| 4 | 43330 |     2     |      2426      |  1038  |   108    | Kettle Tortilla ChpsHny&Jlpno Chili 150g |   3.0    |   13.8    |
+---+-------+-----------+----------------+--------+----------+------------------------------------------+----------+-----------+
```

## **Feature Explanation**

Lets take some quick notes about the data that we have:

!!! note

	 - **`DATE`** : Date since 1899-12-30
     - **`STORE_NBR`** : The store identifier
     - **`LYLTY_CARD_NBR`** : Customer's loyalty identifier
     - **`PROD_NAME`** : Name of the product purchased
     - **`PROD_QTY`** : Products of type purchased
     - **`TOT_SALES`** : Sum of purchase
     -  **`LIFESTAGE`**: Customer attribute that identifies whether a customer has a family or not and what point in life they are at e.g. are their children in pre-school/primary/secondary school.
     - **`PREMIUM_CUSTOMER`** : Customer segmentation used to differentiate shoppers by the price point of products they buy and the types of products they buy. It is used to identify whether customers may spend more for quality or brand or whether they will purchase the cheapest options



## **Loading Dataset**

Time to load our dataset, having given a glimpse of the data, we can define our data types, similar to how you would do in SQL. Well load both of the files and merge them together on column **LYLTY_CARD_NBR**, we'll also make some minor adjustments before we start exploring the dataset.


```python
import pandas as pd

dtypes = {'DATE': int,
          'STORE_NBR':int,
          'LYLTY_CARD_NBR':int,
          'TXN_ID':int,
          'PROD_NBR':int,
          'PROD_NAME':str,
          'PROD_QTY':int,
          'TOT_SALES':float
          }

df_transaction = pd.read_csv(path,dtype=dtypes)
df_behaviour = pd.read_csv(path2)
df = df_transaction.merge(df_behaviour,on='LYLTY_CARD_NBR')
```

As we can see in the data below, we have a customer identifier column **LYLTY_CARD_NBR**, which we will need to do group by operations and determine aggregations for each unique customer that has made a purchase in our transactions dataset.

Our date column is in a rather odd format, what it represents is the number of days since "1899-12-30", so lets convert it to something we are more familiar with (datetime). However since it is a difference, we need it to be in the **time delta** format & not **datetime**, so lets use `pd.to_timedelta` setting the unit to days:

```python
# Convert days since "1899-12-30" to datetime
start_date = pd.to_datetime("1899-12-30")
df['DATETIME'] = start_date + pd.to_timedelta(df['DATE'], unit='d')
```

Lets also determine the first and last transaction date

```python
df['DATETIME'].min(),df['DATETIME'].max()
```

Looks like we have about a years worth of transactional data, the latest date being "2019-06-30", which is a little out of date for an RFM analysis. Let's assume that we received this data on "2019-07-01", and were asked to conduct the analysis

```
(Timestamp('2018-07-01 00:00:00'), Timestamp('2019-06-30 00:00:00'))
```

## **Data preparation and customer analytics**

### Problem Statement

> You are part of Quantium’s retail analytics team and have been approached by your client, the Category Manager for Chips, who wants to better understand the types of customers who purchase Chips and their purchasing behaviour within the region.

> The insights from your analysis will feed into the supermarket’s strategic plan for the chip category in the next half year.

> You have received the following email from your manager, Zilinka.

??? note "Email from Zilinka"

    'Hi, 

    Welcome again to the team, we love having new graduates join us! 

    I just wanted to send a quick follow up from our conversation earlier with a few pointers around the key areas of this task to make sure we set you up for success. 

    Below I have outlined your main tasks along with what we should be looking for in the data for each. 

    Examine transaction data – look for inconsistencies, missing data across the data set, outliers, correctly identified category items, numeric data across all tables. If you determine any anomalies make the necessary changes in the dataset and save it. Having clean data will help when it comes to your analysis. 

    Examine customer data – check for similar issues in the customer data, look for nulls and when you are happy merge the transaction and customer data together so it’s ready for the analysis ensuring you save your files along the way.

    Data analysis and customer segments – in your analysis make sure you define the metrics – look at total sales, drivers of sales, where the highest sales are coming from etc. Explore the data, create charts and graphs as well as noting any interesting trends and/or insights you find. These will all form part of our report to Julia. 

    Deep dive into customer segments – define your recommendation from your insights, determine which segments we should be targeting, if packet sizes are relative and form an overall conclusion based on your analysis. 

    Make sure you save your analysis in the CSV files and your visualisations – we will need them for our report. If you could work on this analysis and send me your initial findings by end of next week that would be great.  

    Looking forward to reviewing your work. 

    Thanks, 

    Zilinka'

> We need to present a strategic recommendation to Julia that is supported by data which she can then use for the upcoming category review. However, to do so, we need to analyse the data to understand the current purchasing trends and behaviours. The client is particularly interested in customer segments and their chip purchasing behaviour. Consider what metrics would help describe the customers’ purchasing behaviour.  


### Problem Workflow

What well do is conduct an analysis on our client's transaction dataset and identify customer purchasing behaviours to generate insights and provide commercial recommendations.

What well learn along the way:

<div class="grid cards" markdown>

- Understand how to examine and clean transaction and customer data.
- Learn to identify customer segments based on purchasing behavior.
- Gain experience in creating charts and graphs to present data insights.
- Learn how to derive commercial recommendations from data analysis

</div>

And we'll be doing the following:

<div class="grid cards" markdown>

- Analyse transaction and customer data to identify trends and inconsistencies. 
- Develop metrics and examine sales drivers to gain insights into overall sales performance. 
- Create visualizations and prepare findings to formulate a clear recommendation for the client's strategy.

</div>

### Cleaning Product Name Column

I would start by exploring the **`PROD_NAME`** column, it contains a several key information parts that we can extract and use in our customer segmentation task. We can also notice some text input abnormalities that we ought to fix along the way.


```python
def remove_mass(x):

    """

    Tokenise & Clean Product Name

    """
    
    string = x['PROD_NAME']
    replaced = re.sub(r'\s*\d+g$', '', string)
    lst_tokens = replaced.split(' ')
    
    lst_data = []
    for i in lst_tokens:
        if(i != ''):
            lst_data.append(i)
        
    return lst_data
    
df['TOKENS'] = df.apply(remove_mass,axis=1)  # cleaned tokenised 
df['TOKENS_STR'] = df['TOKENS'].apply(lambda x: " ".join(x))  # cleaned string name
```

An interest trick you may not have come across when working with python lists; the default library contains the module **`cmd`**, we can use it to neatly and more cleanly display the contents of a python list:

```python
import cmd
cli = cmd.Cmd()
cli.columnize(chips['TOKENS_STR'].unique().tolist(), displaywidth=120)
```

```
Natural Chip Compny SeaSalt           Smiths Crinkle Cut Tomato Salsa      Smiths Chip Thinly CutSalt/Vinegr  
CCs Nacho Cheese                      Kettle Mozzarella Basil & Pesto      Cheezels Cheese                    
Smiths Crinkle Cut Chips Chicken      Infuzions Thai SweetChili PotatoMix  Tostitos Lightly Salted            
Smiths Chip Thinly S/Cream&Onion      Kettle Sensations Camembert & Fig    Thins Chips Salt & Vinegar         
Kettle Tortilla ChpsHny&Jlpno Chili   Smith Crinkle Cut Mac N Cheese       Smiths Crinkle Cut Chips Barbecue  
Old El Paso Salsa Dip Tomato Mild     Kettle Honey Soy Chicken             Cheetos Puffs                      
Smiths Crinkle Chips Salt & Vinegar   Thins Chips Seasonedchicken          RRD Sweet Chilli & Sour Cream      
Grain Waves Sweet Chilli              Smiths Crinkle Cut Salt & Vinegar    WW Crinkle Cut Original            
Doritos Corn Chip Mexican Jalapeno    Infuzions BBQ Rib Prawn Crackers     Tostitos Splash Of Lime            
Grain Waves Sour Cream&Chives 210G    GrnWves Plus Btroot & Chilli Jam     Woolworths Medium Salsa            
Kettle Sensations Siracha Lime        Tyrrells Crisps Lightly Salted       Kettle Tortilla ChpsBtroot&Ricotta 
Twisties Cheese                       Kettle Sweet Chilli And Sour Cream   CCs Tasty Cheese                   
WW Crinkle Cut Chicken                Doritos Salsa Medium                 Woolworths Cheese Rings            
Thins Chips Light& Tangy              Kettle 135g Swt Pot Sea Salt         Tostitos Smoked Chipotle           
CCs Original                          Pringles SourCream Onion             Pringles Barbeque                  
Burger Rings                          Doritos Corn Chips Original          WW Supreme Cheese Corn Chips       
NCC Sour Cream & Garden Chives        Twisties Cheese Burger               Pringles Mystery Flavour           
Doritos Corn Chip Southern Chicken    Old El Paso Salsa Dip Chnky Tom Ht   Tyrrells Crisps Ched & Chives      
Cheezels Cheese Box                   Cobs Popd Swt/Chlli &Sr/Cream Chips  Snbts Whlgrn Crisps Cheddr&Mstrd   
Smiths Crinkle Original               Woolworths Mild Salsa                Cheetos Chs & Bacon Balls          
Infzns Crn Crnchers Tangy Gcamole     Natural Chip Co Tmato Hrb&Spce       Pringles Slt Vingar                
Kettle Sea Salt And Vinegar           Smiths Crinkle Cut Chips Original    Infuzions SourCream&Herbs Veg Strws
Smiths Chip Thinly Cut Original       Cobs Popd Sea Salt Chips             Kettle Tortilla ChpsFeta&Garlic    
Kettle Original                       Smiths Crinkle Cut Chips Chs&Onion   Infuzions Mango Chutny Papadums    
Red Rock Deli Thai Chilli&Lime        French Fries Potato Chips            RRD Steak & Chimuchurri            
Pringles Sthrn FriedChicken           Old El Paso Salsa Dip Tomato Med     RRD Honey Soy Chicken              
Pringles Sweet&Spcy BBQ               Doritos Corn Chips Cheese Supreme    Sunbites Whlegrn Crisps Frch/Onin  
Red Rock Deli SR Salsa & Mzzrlla      Pringles Original Crisps             RRD Salt & Vinegar                 
Thins Chips Originl saltd             RRD Chilli& Coconut                  Doritos Cheese Supreme             
Red Rock Deli Sp Salt & Truffle 150G  WW Original Corn Chips               Smiths Crinkle Cut Snag&Sauce      
Smiths Thinly Swt Chli&S/Cream175G    Thins Potato Chips Hot & Spicy       WW Sour Cream &OnionStacked Chips  
Kettle Chilli                         Cobs Popd Sour Crm &Chives Chips     RRD Lime & Pepper                  
Doritos Mexicana                      Smiths Crnkle Chip Orgnl Big Bag     Natural ChipCo Sea Salt & Vinegr   
Smiths Crinkle Cut French OnionDip    Doritos Corn Chips Nacho Cheese      Red Rock Deli Chikn&Garlic Aioli   
Natural ChipCo Hony Soy Chckn         Kettle Sensations BBQ&Maple          RRD SR Slow Rst Pork Belly         
Dorito Corn Chp Supreme               WW D/Style Chip Sea Salt             RRD Pc Sea Salt                    
Twisties Chicken                      Pringles Chicken Salt Crips          Smith Crinkle Cut Bolognese        
Smiths Thinly Cut Roast Chicken       WW Original Stacked Chips            Doritos Salsa Mild   
```

What we can notice upon inspecting the unique column values is that we have lots of **misspellings** as well as products related to **salsa** (which is a sauce) & we have some overlaps with chips that contain the word "salsa", eg. **Smiths Crinkle Cut Tomato Salsa**, my guess is that this was not intentionally done, so we need to segment these groups.

I have selected the salsas which are present in the column & we will filter out these products from the target products which are chips!

```python
# Salsa 
salasa = ["Doritos Salsa Mild",
        "Old El Paso Salsa Dip Tomato Med",
        "Woolworths Mild Salsa",
        "Old El Paso Salsa Dip Chnky Tom Ht",
        "Doritos Salsa Medium",
        "Woolworths Medium Salsa",
        "Old El Paso Salsa Dip Tomato Mild",
        "Smiths Crinkle Cut Tomato Salsa"]

salsas = df[df['TOKENS_STR'].isin(salasa)].copy()
chips = df[~df['TOKENS_STR'].isin(salasa)].copy()
```

### Parsing Product Name Column

The next step we can take is to identify the product producer parent companies. What I found was that there are products by 9 different parent companies `CATEGORY` shown below. Categorising them in this way will hopefully give us some more insights into customer purchasing behaviour and their market share of sales. We will also categorise our transactions into different brand names `BRAND`. And lastly we will extract the mass of packaging, all of which is extracted from the product name.


```python
# Segment chip/snack parents
woolworths = ['WW','Woolworths']
cobs = ['Cobs']
intersnack = ['Tyrrells']
snack_brands = ['NCC','Natural ChipCo','Natural Chip','CCs','Cheezels','Kettle','French Fries Potato Chips','Thins']
majans = ['Infuzions','Infzns']
red_rock_deli = ['Red Rock Deli','RRD']
pepsico = ['Smiths','Smith','Burger Rings','Dorito','Doritos','Grain Waves','Twisties','Tostitos','Cheetos','GrnWves']
kellanova = ['Pringles']
sunbites = ['Sunbites','Snbts']

lst_brands = pepsico + kellanova + sunbites +  store + cobs + intersnack + snack_brands + majans + red_rock_deli 

# Combine all lists into a dictionary for easy lookup
brand_categories = {
    'Woolworths': woolworths,
    'Cobs': cobs,
    'Intersnack': intersnack,
    'Snack Brands': snack_brands,
    'Majans': majans,
    'Red Rock Deli': red_rock_deli,
    'Pepsico': pepsico,
    'Kellanova': kellanova,
    'Sunbites': sunbites
}

# Function to categorize brands
def categorize_parent(brand):
    for category, brands in brand_categories.items():
        if any(part in brand for part in brands):
            return category
    return 'Other'  

# Function to find matches
def categorize_brand(text):
    for name in lst_brands:
        if name in text:
            return name
    return None

# Apply the function to the DataFrame
chips['PARENT'] = chips['TOKENS_STR'].apply(categorize_parent)                                      
chips['BRAND'] = chips['TOKENS_STR'].apply(categorize_brand)
chips['GRAMS'] = chips['PROD_NAME'].str.extract(r'(\d+)g')
```

### Customer Selection Share

We have some new information about the chips parent company, its brand name and the mass of the product, let's check the customer product choice distribution for these three features:

```python
# sales ammount distribution by parent company
chips['PARENT'].value_counts(normalize=True).round(4)*100
```

```
CATEGORY
Pepsico          34.95
Snack Brands     29.57
Kellanova        10.11
Red Rock Deli     7.16
Majans            5.72
Store             4.77
Cobs              3.91
Intersnack        2.60
Sunbites          1.21
Name: proportion, dtype: float64
```

What we can notice is that chip purchases/sales are quite heavility dominated by two key players **Pepsico** & **Snack Brands** (Australia). One is obviously international & the other is domestic (well if you take into account New Zealand as well perhaps not). 

```python
# sales ammount distribution by parent company
chips['BRAND'].value_counts(normalize=True).round(4)*100
```

```
BRAND
Kettle           16.73
Smiths           12.30
Dorito           10.22
Pringles         10.17
Red Rock Deli     7.20
Infuzions         5.75
Thins             5.70
Woolworths        4.80
Cobs              3.93
Tostitos          3.84
Twisties          3.83
Grain Waves       3.14
Natural Chip      3.03
Tyrrells          2.61
Cheezels          1.87
CCs               1.84
Sunbites          1.22
Cheetos           1.19
Burger Rings      0.63
Name: proportion, dtype: float64
```

The more common brands include **Kettle** (Snack Brands Australia), **Smiths** (PepsiCo), **Doritos** (PepsiCo). Woolworths also sold a fair share of products at 4.8%, so as we can see despite owning the larger share of products chosen by customers. We are not taking into account the quantity purchased at the moment.

```
GRAMS
175    26.82
150    16.59
134    10.37
110     9.25
170     8.25
165     6.32
330     5.18
380     2.65
270     2.60
200     1.85
135     1.35
250     1.31
210     1.31
90      1.24
190     1.24
160     1.23
220     0.65
70      0.62
180     0.61
125     0.60
Name: proportion, dtype: float64
```

When it comes to distribution of package size, we can note that **175** and **150** grams tend to be the most commonly selected products. However this could be purely due to the product preference itself, and we ought to look into the relation between product & size in more detail.

### Store Visits and Checkout Items

Lets also check two other columns, the store purchase statistics, we count the number of store visits for each store and get their stats:

```
count     271.000000
mean      915.867159
std       549.077129
min         1.000000
25%       491.500000
50%       647.000000
75%      1427.500000
max      1918.000000
Name: count, dtype: float64
```

and the item selection count per store visit:

```
PROD_QTY
2      221349
1       25650
5         418
3         408
4         373
200         2
Name: count, dtype: int64
```

From this information we know that there are **271 stores** in our data, with most averaging around **650-900 purchases** in our annual data on average and some stores going as high as 1400-1900. We ought to check the visit distribution for each store to understand where customers tend to go. Another thing we can notice is that most purchases are made for **1 or 2 items**, 3 and above tend to be quite rare. We also can notice a rather strange anomaly of 200 items bought, this probably is not a routine customer purchase, so lets not take them into account in our following analyses since they can skew about results.

```python
chips = chips[chips['PROD_QTY'] != 200]
```

### Customer segmented purchase share

Let's look at member distribution for two customer segmentation features `LIFESTAGE` and `PREMIUM_CUSTOMER`, these two feature will be important in determining customer purchase behaviour. 

```
LIFESTAGE
RETIREES                  20.38
OLDER SINGLES/COUPLES     20.11
YOUNG SINGLES/COUPLES     19.88
OLDER FAMILIES            13.46
YOUNG FAMILIES            12.64
MIDAGE SINGLES/COUPLES    10.02
NEW FAMILIES               3.51
Name: proportion, dtype: float64
```



Now let's look at two features that define a pre determined customer segmentation `LIFESTAGE` and `PREMIUM_CUSTOMER`, these two feature will be important in determining customer purchase behaviour. 

```
LIFESTAGE
OLDER SINGLES/COUPLES     20.58
RETIREES                  18.81
OLDER FAMILIES            18.32
YOUNG FAMILIES            16.42
YOUNG SINGLES/COUPLES     13.76
MIDAGE SINGLES/COUPLES     9.48
NEW FAMILIES               2.63
Name: proportion, dtype: float64
```

```
PREMIUM_CUSTOMER
Mainstream    38.51
Budget        35.16
Premium       26.33
Name: proportion, dtype: float64
```
