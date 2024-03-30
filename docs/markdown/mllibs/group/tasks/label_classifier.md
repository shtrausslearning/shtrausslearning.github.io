---
tags:
  - interpreter classifier
  - iterative process
---

## :material-tag-check: **Module Function Label Classifier**

Linking the input user requests to desired activation functions in integrated modules can be achieved using multiclass classification. 

### Required Knowledge

There is some required knowledge of the internals of the library

#### Corpus

The dataset for each added module is located in **[`src/corpus`](https://github.com/mllibs/mllibs/tree/main/src/mllibs/corpus)**

??? "Module corpus & Module class relation"

	By standard convention, each module functions class and its corpus are related:

	* module ==classes== are stored in `src` and are named `m` + **module name**
	* module ==corpus== are stored in `src/corpus` and are named **module name**

	Example:

	Module class (==eda_simple==) is stored in `src/eda/meda_simple.py` & its corpus is stored in `src/corpus/eda_simple.json`

#### nlpm

The ==**nlpm**== class has several functions, one of these is to **assemble corpuses** into a format used by the classifier. Each functionality module needs to be added into a single group & given a corresponding label.

#### Grouping Added Modules

* A ==**collection**== (group of added modules) instance is first created
* Module instances are loaded are loaded via ==**collection.load()**==
* Classification model(s) are trained using ==**collection.train()**==

``` python
# link all modules into one collection
collection = nlpm()
collection.load([
                 eda_simple(),    # [eda] simple pandas EDA
                 eda_splot(),     # [eda] standard seaborn plots
                 eda_scplot(),    # [eda] seaborn column plots
                ])
collection.train()
```

More detailed information about this process can be found on the **`nlpi`** documentation page

**

	