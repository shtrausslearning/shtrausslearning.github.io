---
# icon: material/cube
---

## **Module Group**

src/eda[^1]

## **Project Stage ID**

[^1]: Reference to the sub folder in **src**

4[^2]

[^2]: Reference to the machine learning project phase identification defined [here](../../projects/mlproject.md)

## :material-frequently-asked-questions: **Purpose**

The purpose of this module is to provide the user with the ability to visualise each numerical columns in a pandas dataframe in a two dimensional figure relative to other numerical columns, the module revolves around the utilisation of **[plotly express](https://plotly.com/python/plotly-express/)**

## :fontawesome-solid-location-arrow: **Location**

Here are the locations of the relevant files associated with the module

<h4>module information:</h4>

/src/eda/meda_pplot.json

<h4>module activation functions:</h4>

/src/eda/meda_pplot.py

## :material-import: **Requirements**

Module import information

```python

```

## :material-selection-drag: **Selection**

Activation functions need to be assigned a unique label. Here's the process of **label** & activation function selection 

```python

```

## :octicons-code-16: **Activation Functions**

Here you will find the relevant activation functions available in class **meda_scplot**

### :octicons-file-code-16: ==plscatter==

<h4>description:</h4>

Used to create scatter plots. Scatter plots are used to visualize the relationship between two numerical variables. 

<h4>code:</h4>

```python linenums="1"
# plotly basic scatter plot(plscatter)
def plotly_scatterplot(self,args:dict):

    fig = px.scatter(args['data'],
                     x=args['x'],
                     y=args['y'],
                     color=args['hue'],
                     facet_col=args['col'],
                     facet_row=args['row'],
                     opacity=nlpi.pp['alpha'],
                     facet_col_wrap=args['col_wrap'],
                     template=nlpi.pp['template'],
                     width=nlpi.pp['figsize'][0],
                     height=nlpi.pp['figsize'][1],
                     title=nlpi.pp['title'])

    # Plot Adjustments

    fig.update_traces(marker={'size':nlpi.pp['s'],"line":{'width':nlpi.pp['mew'],'color':nlpi.pp['mec']}},
                      selector={'mode':'markers'})

    if(nlpi.pp['background'] is False):
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })

    fig.show()
```


### :octicons-file-code-16: ==plbox==

<h4>description:</h4>

Used to create **boxplots**. Box plots are used to visualize the distribution of numerical variables and display summary statistics such as the **median**, **quartiles**, and **outliers**

<h4>code:</h4>

```python linenums="1"
# plotly basic box plot (plbox)
def plotly_boxplot(self,args:dict):

    col_wrap = self.convert_str('col_wrap')
    nbins = self.convert_str('nbins')

    fig = px.box(args['data'],
                 x=args['x'],
                 y=args['y'],
                 color=args['hue'],
                 nbins=nbins,
                 facet_col=args['col'],
                 facet_row=args['row'],
                 facet_col_wrap=col_wrap,
                 template=nlpi.pp['template'],
                 width=nlpi.pp['figsize'][0],
                 height=nlpi.pp['figsize'][1],
                 title=nlpi.pp['title'])

    fig.show()
```

### :octicons-file-code-16: ==plhist==

<h4>description:</h4>

Used to create a histogram plot. A histogram is a graphical **representation of the distribution** of a dataset. It displays the frequency of occurrence of data points within specified intervals, called bins

<h4>code:</h4>

```python linenums="1"
# plotly basic histogram plot (plhist)
def plotly_histogram(self,args:dict):

    col_wrap = self.convert_str('col_wrap')
    nbins = self.convert_str('nbins')

    fig = px.histogram(args['data'],
                       x=args['x'],
                       y=args['y'],
                       color=args['hue'],
                       facet_col=args['col'],
                       facet_row=args['row'],
                       facet_col_wrap=col_wrap,
                       nbins=nbins,
                       template=nlpi.pp['template'],
                       width=nlpi.pp['figsize'][0],
                       height=nlpi.pp['figsize'][1],
                       title=nlpi.pp['title'])

    fig.show()
```

### :octicons-file-code-16: ==plline==

<h4>description:</h4>

Used to create a histogram plot. A histogram is a graphical **representation of the distribution** of a dataset. It displays the frequency of occurrence of data points within specified intervals, called bins

<h4>code:</h4>

```python linenums="1"
# plotly basic histogram plot (plline)
def plotly_line(self,args:dict):

    col_wrap = self.convert_str('col_wrap')

    fig = px.line(args['data'],
                   x=args['x'],
                   y=args['y'],
                   color=args['hue'],
                   facet_col=args['col'],
                   facet_row=args['row'],
                   facet_col_wrap=col_wrap,
                   template=nlpi.pp['template'],
                   width=nlpi.pp['figsize'][0],
                   height=nlpi.pp['figsize'][1],
                   title=nlpi.pp['title'])

    fig.show()
```

### :octicons-file-code-16: ==plviolin==

<h4>description:</h4>

A violin plot displays the **distribution of a continuous variable** across different categories or groups. It consists of a series of vertical or horizontal violin-shaped curves, where the width of each curve represents the density or frequency of data points at different values. The wider parts of the curve indicate areas with higher density, while the narrower parts represent areas with lower density.

<h4>code:</h4>

```python linenums="1"
# [plotly] Violin plot (plviolin)

def plotly_violin(self,args:dict):

    col_wrap = self.convert_str('col_wrap')

    fig = px.violin(args['data'],
                   x=args['x'],
                   y=args['y'],
                   color=args['hue'],
                   facet_col=args['col'],
                   facet_row=args['row'],
                   facet_col_wrap=col_wrap,
                   box=True,
                   template=nlpi.pp['template'],
                   width=nlpi.pp['figsize'][0],
                   height=nlpi.pp['figsize'][1],
                   title=nlpi.pp['title'])

    fig.show()
```

### :octicons-file-code-16: ==plbarplot==

<h4>description:</h4>

A **bar chart** displays the distribution or comparison of data across different categories or groups. Each category is represented by a bar, where the length or height of the bar corresponds to the value or frequency of the data in that category

<h4>code:</h4>

```python linenums="1"
# [plotly] Bar Plot (plbarplot)

def plotly_bar(self,args:dict):

    fig = px.bar(args['data'],
                 x=args['x'],
                 y=args['y'],
                 color=args['hue'],
                 facet_col=args['col'],
                 facet_row=args['row'],
                 facet_col_wrap=col_wrap,
                 template=nlpi.pp['template'],
                 width=nlpi.pp['figsize'][0],
                 height=nlpi.pp['figsize'][1],
                 title=nlpi.pp['title'])

    fig.show()
```

### :octicons-file-code-16: ==plbarplot==

<h4>description:</h4>

A **density heatmap** uses color to represent the density or frequency of data points in different regions of the 2D space. The intensity of the color corresponds to the density of data points, with darker colors indicating higher densities.

<h4>code:</h4>

```python linenums="1"
# [plotly] Heatmap (plheatmap)

def plotly_heatmap(self,args:dict):

    col_wrap = self.convert_str('col_wrap')

    fig = px.density_heatmap(args['data'],
                             x=args['x'],
                             y=args['y'],
                             facet_col=args['col'],
                             facet_row=args['row'],
                             facet_col_wrap=col_wrap,
                             template=nlpi.pp['template'],
                             width=nlpi.pp['figsize'][0],
                             height=nlpi.pp['figsize'][1],
                             title=nlpi.pp['title'])

    fig.show()
```
