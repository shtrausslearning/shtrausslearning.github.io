---
comments: true
---

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Colab-blue?logo=Google&logoColor=FDBA18)]()

## **Data Storage**

To utilise automisation with **mllibs**, data needs to be stored in **nlpi** instances. Input data is allocated a **name tag**, allowing the user to reference the data source in a query so it can be used as an input into the selected **activation function**


### **Loading Preset Datasets**

Both [**Seaborn**](https://github.com/mwaskom/seaborn) and [**Plotly**](https://plotly.com/python-api-reference/generated/plotly.express.data.html) offer preset dataset which can be used for simple data analyses and testing. A mixture of different datasets from both have been selected to be loaded with the method **load_sample_data**

```python linenums="1"
def load_sample_data(self):
    self.store_data(px.data.stocks(),'stocks')
    self.store_data(px.data.tips(),'tips')
    self.store_data(px.data.iris(),'iris')
    self.store_data(px.data.carshare(),'carshare')
    self.store_data(px.data.experiment(),'experiment')
    self.store_data(px.data.wind(),'wind')
    self.store_data(sns.load_dataset('flights'),'flights')
    self.store_data(sns.load_dataset('penguins'),'penguins')
    self.store_data(sns.load_dataset('taxis'),'taxis')
    self.store_data(sns.load_dataset('titanic'),'titanic')
    self.store_data(sns.load_dataset('mpg'),'mpg')
```

once the **nlpi** instance has been created, you can store all the above data in **nlpi.data** and reference to them by their allocated name, shown above

```python linenums="1"
nlpi.load_sample_data()
```

### **Loading local data**

To load your own data, you need to use **nlpi.store_data**. At present only two formats are used as storage types python **lists** and pandas **dataframes**. They can be imported directly

```python linenums="1"
nlpi.store_data(data:(list or pd.DataFrame),'name')
```

Or as part of a dictionary input:

```python linenums="1"
nlpi.store_data(data:{'name1':list,'name2':pd.DataFrame})
```

### **Active Columns**

When using **natural language** for automation, specifying a subset of a **dataframe** in a single query can make them them quite long. As a result, the use of **active columns** or simply put defined **subset column lists** is utilised in **mllibs** and can be defined by setting. An important distinguish to note is that **active columns** are not **data souces**, they are stored in the existing data dictionary under the key **ac**

```python linenums="1"
nlpi.store_ac('data_name','active_column name',['column A','column B'])
```

Utilisation of **active columns** can be done via defining the name of the active column using **{}** quotations marks inside a query


## **Extracting Data Content**

If you have the need to extract data related content, you can call **nlpi.data['data_name']**

#### **DataFrame Storage**

**nlpi** stores a variety of data related to **DataFrames**, the stored content changes depending on the implemented **activation functions**, here's an example:

```python linenums="1"
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



