
## **Information**

### :fontawesome-solid-layer-group: **Module Group**

src/stats[^1]

### :material-identifier: **Project Stage ID**

[^1]: Reference to the sub folder in `src`

4[^2]

[^2]: Reference to the machine learning project phase identification defined [here](../../projects/mlproject.md)

### :material-frequently-asked-questions: **Purpose**

The purpose of this module is to provide the user with a graphic visualisation of the statistical difference between two samples


### :fontawesome-solid-location-arrow: **Module Files**

Here are the locations of the relevant files associated with the module

<h4>module information</h4> 

/src/stats/mstats_plot.json

<h4>module activation functions</h4>

/src/stats/mstats_plot.py

### :material-import: **Requirements**

Module import information 

```python
from mllibs.nlpi import nlpi
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
```

### :material-selection-drag: **Selection**

Activation functions need to be assigned a unique label. Here's the process of **label** & activation function selection 

```python
# select activation function
def sel(self,args:dict):
    self.args = args
    select = args['pred_task']
    self.data_name = args['data_name']
    self.subset = args['subset']
    
    if(select == 'dp_hist'):
        self.dp_hist(args)
    if(select == 'dp_kde'):
        self.dp_kde(args)
```

## :octicons-code-16: **Activation Functions**

Here you will find the relevant **activation functions** available in **mstats_plot**

### <b>:octicons-file-code-16: ==dp_hist==</b>

<h4><b>data: [<code>list</code>,<code>list</code>] targ:<code>None</code></b></h4>

To visualise the univariate distribution variation between two samples, we can utilise the **histogram** distributions. Plotly express offers the ability to compare boxplot statistics for both datasets as well 

<h4>code:</h4>

```python
# plot Histogram of Two Samples (use plotly express)
# which don't necessarily have the same sample size 

def dp_hist(self,args:dict):

    sample1 = args['data'][0]
    sample2 = args['data'][1]

    data1 = pd.DataFrame(sample1,columns=['data'])
    data1['sample'] = 'one'
    data2 = pd.DataFrame(sample2,columns=['data'])
    data2['sample'] = 'two'
    names = ['one','two']
    combined = pd.concat([data1,data2])

    means_data = combined.groupby(by='sample').agg('mean')
    means = list(means_data['data'])

    floc = [0.55,0.65]
    fig = px.histogram(combined,x='data',color='sample',
                       marginal="box",
                       template='plotly_white',nbins=args['nbins'],
                       color_discrete_sequence=self.default_colors[0],
                       title='Comparing univariate distributions')

    fig.update_traces(opacity=0.8)
    fig.update_layout(barmode='group') # ['stack', 'group', 'overlay', 'relative']
    fig.update_layout(height=350,width=700)  
    # fig.update_traces(marker_line_width=1,marker_line_color="white") # outline

    fig.show()
```

<h4>sample request:</h4>

```python
sample1 = list(np.random.exponential(scale=1, size=1000))
sample2 = list(np.random.exponential(scale=1, size=1000))

interpreter.store_data({'distribution_A':sample1,
                        'distribution_B':sample2})

# request
req = "compare the histograms of two samples distribution_B distribution_A nbins 50"

# execution of request
interpreter[req]
```


### <b>:octicons-file-code-16: ==dp_kde==</b>

<h4><b>data: [<code>list</code>,<code>list</code>] targ:<code>None</code></b></h4>

To visualise the univariate distribution variation between two samples, we can also utilise a kernel density representation of the distributions. Seaborn offers a way to visualise this estimation in a static figure format.

<h4>code:</h4>

```python
# plot Kernel Density Plot of Two Samples

def dp_kde(self,args:dict):

    sample1 = args['data'][0]
    sample2 = args['data'][1]
    names = ['Sample 1','Sample 2']

    fig,ax = plt.subplots(1,1,figsize=(7,3.5))

    # Create a kernel density plot
    sns.kdeplot(data=[sample1, sample2],palette=self.default_colors[1],ax=ax,fill=True)
    sns.histplot(data=[sample1, sample2],palette=self.default_colors[1],ax=ax,alpha=0.01,stat='density',edgecolor=(0, 0, 0, 0.01))
    plt.legend(names)
    plt.xlabel('Values')
    plt.ylabel('Density')
    # plt.title('Distribution of Two Samples')
    plt.title('Distribution of Two Samples', loc='left', pad=10, fontdict={'horizontalalignment': 'left'})
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()
```

<h4>sample request:</h4>

```python
sample1 = list(np.random.exponential(scale=1, size=1000))
sample2 = list(np.random.exponential(scale=1, size=1000))

interpreter.store_data({'distribution_A':sample1,
                        'distribution_B':sample2})

# request
req = "compare kde plot of two samples distribution_B distribution_A"

# execution of request
interpreter[req]
```