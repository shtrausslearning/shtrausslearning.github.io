---
date: 2024-08-02
title: Heuristic Customer Segmentation
authors: [andrey]
draft: true
categories:
     - rfm segmentation
tags:
     - financial analysis
     - rfm segmentation
     - internship
comments: true
---

# **Heuristic Customer Segmentation**

<div style="width: 100%; font-family: Trebuchet MS; font-weight: bold;">
    <div style="padding-top: 40%; position: relative; background-color: #000000; border-radius:10px;">
        <div style="background-image: url('https://cdn.theforage.com/vinternships/companyassets/32A6DqtsbF7LbKdcq/9F4wA4B5J6Kgmef3k/1693137922931/quantium_gradient_homepage_img_optimised.png'); background-size: cover; background-position: center; position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 1.0; border-radius:10px">
        </div>
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;">
            <div style="position: relative; display: table; height: 75%; width: 100%;">
            </div>
            <div style="position: absolute; bottom: 30px; left: 30px;">
            </div>
        </div>
    </div>
</div>

In today's post we will discuss heuristic segmentation, on a dataset of transactions provided as part of the [quantium internship](https://www.theforage.com/simulations/quantium/data-analytics-rqkb)

<div class="grid cards" markdown>

- :simple-kaggle:{ .lg .middle }&nbsp; <b>[Kaggle Dataset](https://www.kaggle.com/datasets/shtrausslearning/forage-internship-data)</b>

- :fontawesome-regular-rectangle-list:{ .lg .middle }&nbsp; <b>[Quantium Internship](https://www.theforage.com/simulations/commonwealth-bank/intro-data-science-sd7t)</b>

</div>

<!-- more -->

## **Background**

**Heuristic segmentation** aims to divide users/customers into different groups based on some empirical rules or models that have been deduced based on experience working with the group.

The positives of such methods are that they are quick to realise, interpretable 

## <b>Recency Frequency & Monetary Analysis</b>

One of the heuristic methods is the Recency, Frequency & Monetary (RFM) analysis. Its advantage over the previously mentioned approaches is that it takes into account the factor of time or **recency**. 


## <b>Quantium Dataset</b>

Lets explore our dataset, to understand the dataset that we are going to be working with, and figure out which columns we'll need for our RFM analysis.

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

Lets take some quick notes about the data that we have, we probably don't need anything else:

!!! note

     - DATE : Date since 1899-12-30
     - STORE_NBR : The store identifier
     - LYLTY_CARD_NBR : Customer's loyalty identifier
     - PROD_NAME : Name of the product purchased
     - PROD_QTY : Products of type purchased
     - TOT_SALES : Sum of purchase
     - LIFESTAGE: Customer attribute that identifies whether a customer has a family or not and what point in life they are at e.g. are their children in pre-school/primary/secondary school.
     - PREMIUM_CUSTOMER: Customer segmentation used to differentiate shoppers by the price point of products they buy and the types of products they buy. It is used to identify whether customers may spend more for quality or brand or whether they will purchase the cheapest options.


## **Loading Dataset**

Time to load our dataset, having given a glimpse of the data, we can define our data types, similar to how you would do in SQL. 


```python
import pandas as pd

# define data types
dtypes = {'DATE': int,
          'STORE_NBR':int,
          'LYLTY_CARD_NBR':int,
          'TXN_ID':int,
          'PROD_NBR':int,
          'PROD_NAME':str,
          'PROD_QTY':int,
          'TOT_SALES':float
          }

df = pd.read_csv('QVI_transaction_data.csv',dtype=dtypes)
```

As we can see in the data below, we have a customer identifier column ==LYLTY_CARD_NBR==, which we will need to do group by operations and determine aggregations for each unique customer that has made a purchase in our transactions dataset.

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











## :material-checkbox-multiple-marked-circle-outline: **Data preparation and customer analytics**

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

### Cleaning Product Name Column

I would start by exploring the **`PROD_NAME`** column, to determine whether the data content is relevant to our problem. Data is often noisy & need to clean it if there are any abnormalities present.


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

What we can notice upon inspecting the unique column values is that we have lots of **misspellings** as well as products related to **salsa** & since we have some overlaps witj chips that contain the word "salsa", eg. **Smiths Crinkle Cut Tomato Salsa**, my guess is that this was not intentional, so we need to segment these groups.

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

### Grouping products into producer

The next step we can take is to identify the product producer parent companies. What I found was that there are products by 9 different parent companies shown below, categorising them in this way will hopefull give us some more insights into customer purchasing behaviour

```python
# Segment chip/snack parents
store = ['WW','Woolworths']
cobs = ['Cobs']
intersnack = ['Tyrrells']
snack_brands = ['NCC','Natural ChipCo','Natural Chip','CCs','Cheezels','Kettle','French Fries Potato Chips','Thins']
majans = ['Infuzions','Infzns']
red_rock_deli = ['Red Rock Deli','RRD']
pepsico = ['Smiths','Smith','Burger Rings','Dorito','Doritos','Grain Waves','Twisties','Tostitos','Cheetos','GrnWves']
kellanova = ['Pringles']
sunbites = ['Sunbites','Snbts']

# Combine all lists into a dictionary for easy lookup
brand_categories = {
    'Store': store,
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
def categorize_brand(brand):
    for category, brands in brand_categories.items():
        if any(part in brand for part in brands):
            return category
    return 'Other'  # Return 'Other' if no match is found

# Apply the function to the DataFrame
chips['CATEGORY'] = chips['TOKENS_STR'].apply(categorize_brand)
```

```python
# sales ammount distribution by brand
chips['CATEGORY'].value_counts(normalize=True).round(4)*100
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

What we can notice is that chip purchases/sales are quite heavility dominated by two key players **Pepsico** & **Snack Brands** (Australia). One is obviously international & the other is domestic (well if you take into account New Zealand as well perhaps not). Notable is also the portion of **Woolworths**, which was 4.77% of all total sales revenue.









## **RFM Process**

### Determine the number of days since last transaction

First things first, we need to determine, for each customer when their last purchase occured, so we will need to determine for each transaction, when it last occured. We have also decided that the current date is "2019-07-01". 

To conduct subtraction, we need to create a datetime timestamp for this date, to do this we use the method `pd.to_datetime()` & to get the number of days we utilise `dt`

```python
today = pd.to_datetime('2019-07-01')

df['DT_LAST_TRANSACTION'] = (today - df['DATETIME']).dt.days
```

###