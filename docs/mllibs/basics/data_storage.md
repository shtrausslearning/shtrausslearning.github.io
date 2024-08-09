---
comments: true
---

Data Storage

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Colab-blue?logo=Google&logoColor=FDBA18)]()

## **Data Storage**

To utilise automisation with **mllibs**, data needs to be stored in **nlpi** instances. Input data is allocated a **name tag** (or key), allowing the user to reference the data source in a query so it can be used as an input into the selected **activation function**


### **<span style='color:#686dec'> Preset Datasets</span>**
    

Preset datasets are a quick way to load **load_sample_data()**

```python
def load_sample_data(self):
    self.store_data(sns.load_dataset('flights'),'flights')
    self.store_data(sns.load_dataset('penguins'),'penguins')
    self.store_data(sns.load_dataset('taxis'),'taxis')
    self.store_data(sns.load_dataset('titanic'),'titanic')
    self.store_data(sns.load_dataset('mpg'),'mpg')
```

once the **nlpi** instance has been created, you can store all the above data in **<span style='color:#eb92d0'>i.data** and reference to them by their allocated name, shown above

```python hl_lines="14"
c = nlpm()
c.load([
         eda_splot(),     # [eda] standard seaborn plots
         eda_scplot(),    # [eda] seaborn column plots
         stats_tests(),   # [stats] statistical tests for list data
         stats_plot(),    # [stats] ploty and compare statistical distributions
         libop_general(), # [library] mllibs related functionality
         pd_talktodata(), # [eda] pandas data exploration 
         fourier_all()    # [signal] fast fourier transformation related
        ])

c.setup()
i = nlpi(c)               # create an instance of nlpi
i.load_sample_data()      # load preset datasets
i.data.keys()
```

```
dict_keys(['flights', 'penguins', 'taxis', 'titanic', 'dmpg', 'stocks'])
```

Having loaded the data, you will have access to them when making some text requests!

### **<span style='color:#686dec'> Loading Own Data</span>**

To load your own data and reference your own data, you need to use **i.store_data** method. At present only two formats are used as storage types python **<span style='color:#eb92d0'>lists** and pandas **<span style='color:#eb92d0'>dataframes**. They can be imported directly

```py hl_lines="2 3" linenums="1"
nlpi.store_data(data:(list or pd.DataFrame),'name')
```

Or as part of a dictionary input:

```python linenums="1"
i.store_data(data:{'name1':list,'name2':pd.DataFrame})
```

#### **<span style='color:#8a8a8a '> :material-tag-check-outline: Example</span>**

For example, load **dataframe data** from the desired souce and name it something relevant:

```python

df = pd.read_csv('https://raw.githubusercontent.com/shtrausslearning/Data-Science-Portfolio/main/sources/stocks.csv',delimiter=',')
i.store_data({'stocks':df})
```

When wanting to use the data, simply use its reference name

```python
i['show the dataframe information for stocks']
```

Or some **python list** data, and give them relevant names which will be used to reference this data:

```python
store data
sample1 = list(np.random.normal(scale=1, size=1000))
sample2 = list(np.random.normal(scale=1, size=1000))
i.store_data({'distribution_A':sample1,
              'distribution_B':sample2})
```

An example when you want to compare both datasets:

```python
 i['comapare histograms of samples distribution_B distribution_A']
```

### **<span style='color:#686dec'> Active Columns**

When using **natural language** for automation, specifying a subset of a **dataframe** in a single query can make them them quite long. As a result, the use of **active columns** or simply put defined **subset column lists** is utilised in **mllibs** and can be defined by setting. 

An important distinguish to note is that **<span style='color:#eb92d0'>active columns** are not **data souces**, they are stored in the existing data dictionary under the key **ac** (see Data Extraction)

```python linenums="1"
i.store_ac('data_name','active_column name',['column A','column B'])
```

Its usage is quite standard:

> - First, specify for which data you want to store some subset of column names as active column names **"data_name"**
> - Give the active column some name, which will allow you to reference the particular columns
> - Specify a python list of strings which with the names of the columns of the dataframe

#### **<span style='color:#8a8a8a '> :material-tag-check-outline: Example</span>**

For example, we have dataset **penguins**, for which we want to reference two columns **bill_length_mm**,**bill_depth_mm** as **"selected_columns"**

We can do this by calling the **store_ac** method

```python
i.store_ac('penguins',                        # data name
           'selected_columns',                # active column reference name
           ['bill_length_mm','bill_depth_mm'] # column names that make up active column name
           ) 
```

Confirm, we have stored **selected_columns** into **penguins**:

```python
i.data['penguins']['ac']
{'selected_columns': ['bill_length_mm', 'bill_depth_mm']}
```

**Sample Requests**

An example, referencing the active column name in a request:

```python
i['using data penguins create a relplot using columns selected_columns set hue as island']
```

## **Data Extraction**

If you have the need to extract data related content, you can call **<span style='color:#eb92d0'>i.data**


#### **DataFrame Storage**

**nlpi** stores a variety of data related to **DataFrames**, the stored content changes depending on the implemented **activation functions**, here's an example:

```python linenums="1"
i.data['stocks']

{'data':            date      GOOG      AAPL      AMZN        FB      NFLX      MSFT
 0    2018-01-01  1.000000  1.000000  1.000000  1.000000  1.000000  1.000000
 1    2018-01-08  1.018172  1.011943  1.061881  0.959968  1.053526  1.015988
 2    2018-01-15  1.032008  1.019771  1.053240  0.970243  1.049860  1.020524
 3    2018-01-22  1.066783  0.980057  1.140676  1.016858  1.307681  1.066561
 4    2018-01-29  1.008773  0.917143  1.163374  1.018357  1.273537  1.040708
 ..          ...       ...       ...       ...       ...       ...       ...
 100  2019-12-02  1.216280  1.546914  1.425061  1.075997  1.463641  1.720717
 101  2019-12-09  1.222821  1.572286  1.432660  1.038855  1.421496  1.752239
 102  2019-12-16  1.224418  1.596800  1.453455  1.104094  1.604362  1.784896
 103  2019-12-23  1.226504  1.656000  1.521226  1.113728  1.567170  1.802472
 104  2019-12-30  1.213014  1.678000  1.503360  1.098475  1.540883  1.788185
 
 [105 rows x 7 columns],
 'subset': None,
 'splits': {},
 'splits_col': {},
 'features': ['date', 'GOOG', 'AAPL', 'AMZN', 'FB', 'NFLX', 'MSFT'],
 'target': None,
 'cat': ['date'],
 'num': ['GOOG', 'AAPL', 'AMZN', 'FB', 'NFLX', 'MSFT'],
 'miss': False,
 'size': 105,
 'dim': 7,
 'model_prediction': {},
 'model_correct': {},
 'model_error': {},
 'ac': {},
 'ft': None,
 'outliers': {},
 'dimred': {}}
```



