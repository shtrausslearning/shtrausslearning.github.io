## **Data Storage**

To utilise automisation with **mllibs**, data needs to be stored in **nlpi** instances (nlpi)


### **Loading Preset Datasets**

Both [**Seaborn**](https://github.com/mwaskom/seaborn) and [**Plotly**](https://plotly.com/python-api-reference/generated/plotly.express.data.html) offer preset dataset which can be used for simple data analyses and testing. A mixture of different datasets from both have been selected to be loaded with the method **load_sample_data**

```python
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

### **Loading local data**

To load your own data, you need to use **nlpi.store_data**. At present only two formats are used as storage types python **lists** and pandas **dataframes**. They can be imported directly

```python
nlpi.store_data(data:(list or pd.DataFrame),'name')
```

Or as part of a dictionary input:

```python
nlpi.store_data(data:{'name1':list,'name2':pd.DataFrame})
```

### **Active Columns**

When using **natural language** for automation, specifying a subset of a **dataframe** in a single query can make them them quite long. As a result, the use of **active columns** or simply put defined **subset column lists** is utilised in **mllibs** and can be defined by setting. An important distinguish to note is that **active columns** are not **data souces**, they are stored in the existing data dictionary under the key **ac**

```python
nlpi.store_ac('data_name','active_column name',['column A','column B'])
```

Utilisation of **active columns** can be done via defining the name of the active column using **{}** quotations marks inside a query


