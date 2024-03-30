## **Module Group**

src/pd[^1]

## **Project Stage ID**

[^1]: Reference to the sub folder in **src**

4[^2]

[^2]: Reference to the machine learning project phase identification defined [here](../../projects/mlproject.md)

## :material-frequently-asked-questions: **Purpose**

The purpose of this library is to allow the user to get to know the data stored in the dataframe using **natural language** 


## :fontawesome-solid-location-arrow: **Location**

Here are the locations of the relevant files associated with the module

<h4>module information:</h4>

/src/pd/mpd_talktodata.json

<h4>module activation functions:</h4>

/src/pd/mpd_talktodata.py

## :material-import: **Requirements**

Required module import information

```python
import numpy as np
import pandas as pd
from collections import OrderedDict
from mllibs.nlpi import nlpi
from mllibs.nlpm import parse_json
import pkg_resources
import json
```

## :material-selection-drag: **Selection**

Activation functions need to be assigned a unique label. Here's the process of **label** & activation function selection 

```python
def sel(self,args:dict):
    
    self.select = args['pred_task']
    self.args = args
    
    if(self.select == 'dfcolumninfo'):
        self.dfgroupby(self.args)
    if(self.select == 'dfsize'):
        self.dfsize(self.args)
    if(self.select == 'dfcolumn_distr'):
        self.dfcolumn_distr(self.args)
    if(self.select == 'dfcolumn_na'):
        self.dfcolumn_na(self.args)
    if(self.select == 'dfall_na'):
        self.dfall_na(self.args)
    if(self.select == 'show_stats'):
        self.show_statistics(args)
    if(self.select == 'show_info'):
        self.show_info(args)
    if(self.select == 'show_dtypes'):
        self.show_dtypes(args)
    if(self.select == 'show_feats'):
        self.show_features(args)   
    if(self.select == 'show_corr'):
        self.show_correlation(args)
```

## :octicons-code-16: **Activation Functions**

Here you will find the relevant activation functions available in class **mpd_talktodata**

### <b>:octicons-file-code-16: ==dfcolumninfo==</b>
 
<h4><b>data: <code>pd.DataFrame</code> targ:<code>None</code></b></h4>

The method is used to print the dataframe columns

<h4>code:</h4>

```python linenums="1"
def dfcolumninfo(self,args:dict):
    print(args['data'].columns)
```

### <b>:octicons-file-code-16: ==dfsize==</b>

<h4><b>data: <code>pd.DataFrame</code> targ:<code>None</code></b></h4>

The method is used to print the dataframe size

<h4>code:</h4>

```python linenums="1"
def dfsize(self,args:dict):
    print(args['data'].shape)
```

### :octicons-file-code-16: <b>==dfcolumn_distr==</b>

<h4><b>data: <code>pd.DataFrame</code> targ:<code>col</code>|<code>column</code></b></h4>

The method is used to print count the unique dataframe column values using **value_counts**

<h4>code:</h4>

```python
def dfcolumn_distr(self,args:dict):
    if(args['column'] != None):
        display(args['data'][args['column']].value_counts())
    elif(args['col'] != None):
        display(args['data'][args['col']].value_counts())
    else:
        print('[note] please specify the column name')
```

### :octicons-file-code-16: <b>==dfcolumn_na==</b> 

<h4><b>data: <code>pd.DataFrame</code> targ:<code>col</code>|<code>column</code></b></h4>

The method is used to store the missing data rows found in the dataframe column in **memory_output**

<h4>code:</h4>

```python linenums="1"
def dfcolumn_na(self,args:dict):

    if(args['column'] != None):
        ls = args['data'][args['column']]
    elif(args['col'] != None):
        ls = args['data'][args['col']]
    else:
        print('[note] please specify the column name')
        ls = None

    if(ls != None):

        # convert series to dataframe
        if(isinstance(ls,pd.DataFrame) == False):
            ls = ls.to_frame()

        print("[note] I've stored the missing rows")
        nlpi.memory_output.append({'data':ls[ls.isna().any(axis=1)]})     
```

### <b>:octicons-file-code-16: ==dfall_na==</b>

<h4><b>data: <code>pd.DataFrame</code> targ:<code>None</code></b></h4>

The method is used to print the statistics of the ammount of data missing in all columns & store the missing rows in **memory_output**

<h4>code:</h4>

```python linenums="1"
def dfall_na(self,args:dict):
    
    print(args['data'].isna().sum().sum(),'rows in total have missing data')
    print(args['data'].isna().sum())

    print("[note] I've stored the missing rows")
    ls = args['data']
    nlpi.memory_output.append({'data':ls[ls.isna().any(axis=1)]})  
```

### <b>:octicons-file-code-16: ==show_info==</b>

<h4><b>data: <code>pd.DataFrame</code> targ:<code>None</code></b></h4>

Method is used to print a concise summary of a pandas DataFrame. It provides information such as the number of rows and columns, the data types of each column, the memory usage, and the number of non-null values in each column. This method is useful for quickly understanding the structure and content of a DataFrame, especially when working with large datasets. Additionally, it can help identify missing or null values that may need to be addressed in data cleaning or preprocessing.

<h4>code:</h4>

```python linenums="1"
@staticmethod
def show_info(args:dict):
    print(args['data'].info())
```

### <b>:octicons-file-code-16: ==show_missing==</b>

<h4><b>data: <code>pd.DataFrame</code> targ:<code>None</code></b></h4>

Method is used to print a concise summary of a pandas DataFrame. It provides information such as the number of rows and columns, the data types of each column, the memory usage, and the number of non-null values in each column. This method is useful for quickly understanding the structure and content of a DataFrame, especially when working with large datasets. Additionally, it can help identify missing or null values that may need to be addressed in data cleaning or preprocessing.

<h4>code:</h4>

```python
@staticmethod
def show_missing(args:dict):
    print(args['data'].isna().sum(axis=0))
```

### <b>:octicons-file-code-16: ==show_stats==</b>

<h4><b>data: <code>pd.DataFrame</code> targ:<code>None</code></b></h4>

`pandas.DataFrame.describe()` is a method that provides a summary of the statistical properties of each column in a DataFrame. By default, it calculates the count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum for each numeric column. 

<h4>code:</h4>

```python
@staticmethod
def show_statistics(args:dict):
    display(args['data'].describe())
```

### <b>:octicons-file-code-16: ==show_dtypes==</b>

<h4><b>data: <code>pd.DataFrame</code> targ:<code>None</code></b></h4>

Attribute of a pandas DataFrame that returns the data types of each column in the DataFrame. This attribute is useful for understanding the data types of each column and can be used to convert columns to different data types if necessary.

<h4>code:</h4>

```python
@staticmethod
def show_dtypes(args:dict):
    print(args['data'].dtypes)
```

### <b>:octicons-file-code-16: ==show_corr==</b>

<h4><b>data: <code>pd.DataFrame</code> targ:<code>None</code></b></h4>

Method that calculates the correlation between columns in a DataFrame. Correlation is a statistical measure that indicates the degree to which two variables are related

<h4>code:</h4>

```python
@staticmethod
def show_correlation(args:dict):
    corr_mat = pd.DataFrame(np.round(args['data'].corr(),2),
                         index = list(args['data'].columns),
                         columns = list(args['data'].columns))
    corr_mat = corr_mat.dropna(how='all',axis=0)
    corr_mat = corr_mat.dropna(how='all',axis=1)
    display(corr_mat)
```
