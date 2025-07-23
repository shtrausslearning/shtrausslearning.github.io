---
title: PySpark Basics
comments: true
---

## **:simple-apachespark: PySpark Basics**

Some simple and brief collection of pyspark operations 


<div class="grid cards" markdown>

-   :material-notebook-multiple: __[Select, Drop, Rename Columns](../blog/posts/2025-07-14-spark4.md)__

    ---

    - Column selection
    - Dropping columns
    - Renaming of columns


-   :material-notebook-multiple: __[Pipelines](../blog/posts/2025-07-14-spark3.md)__

    ---

    - Missing data treatment classification pipeline
    - Feature scaling using ScandardScaler classification pipeline
    - TF-IDF corpus classification pipeline
    - PCA dimensionality reduction classification pipeline


-   :material-notebook-multiple: __[Data Filtration](../blog/posts/2025-07-18-spark5.md)__

    ---

    - Filtration by column value
    - String related filtration using like / contains
    - Missing data filtration
    - List based filtration using isin
    - General data clearning operations


-   :material-notebook-multiple: __[Aggregations](../blog/posts/2025-07-19-spark6.md)__

    ---

    - Aggregations without grouping
    - Aggregations with grouping 
    - Filtering after grouping


-   :material-notebook-multiple: __[Pivoting](../blog/posts/2025-07-21-spark7.md)__

    ---

	- Basic pivot operation
	- Pivot with multiple aggregations
	- Conditional pivoting
	- Pivoting with specified column values


-   :material-notebook-multiple: __[Time Series Pipelines](../blog/posts/2025-07-23-spark8.md)__

    ---

    - Basic pipeline conversion of timestamp to unix time
    - Lag feature combination pipelines 
    - Aggregation based statistics pipelines


</div>


## **:simple-apachespark: Basic Classification Project**

A classification project for beginners, shows how one can utilise pyspark in a machine learning project

<div class="grid cards" markdown>

-   :material-notebook-multiple: __[Data Preprocessing with PySpark](../blog/posts/2023-08-20-spark-preprocess.md)__

    ---

    - We'll look at how to start a spark_session
    - Setting up data types for the dataset using StructType
    - Focuses on data preparation in the preprocessing state


-   :material-notebook-multiple: __[Training ML Models with PySpark](../blog/posts/2023-08-21-spark-modeling.md)__

    ---

    - Using spark.ml.classification to train binary classification models
    - Introduction to StringIndexer, VectorAssembler
    - Splitting dataset into subsets using .randomSplit
    - Saving & loading models


-   :material-notebook-multiple: __[Hyperparameter Tuning with Pipelines](../blog/posts/2025-08-22-hyperparameter-tuning-with-pipelines.md)__

    ---

    - Using spark.ml.pipeline introduce a compact training approach
    - Saving & loading pipelines
    - Model evaluation using MulticlassClassificationEvaluator
    - pyspark.ml.tuning for hyperparameter optimisation

</div>

