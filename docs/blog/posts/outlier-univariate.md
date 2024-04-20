---
date:
  created: 2024-03-17
  updated: 2024-03-17
category:
  - eda
description: Learn how to handle outlier using various univariate methods.
title: Handle Outliers - Univariate
slug: outlier-univariate
draft: true
---

# :calendar:{ title="Thursday, 20 July 2023 10:38 AM" } Handle Outliers - Univariate

Handling outlier is a big task for data scientist. To handle the outliers we have many different methods to handle them **i.e. IQR, Z-score, Mean-Median Imputation, Winsorization, etc**. We are going to discuss only univariate methods to handle outliers.

> :calendar: I have written this page as notes very time ago; so if there is any mistake please let me know I'll fix it. Thanks 🤗

<!-- more -->

## Deletion Based Approach

### IQR

In this method by using Inter Quartile Range(IQR), we detect outliers. IQR tells us the variation in the data set. Any value, which is beyond the range of $-1.5 \ast IQR$ to $1.5 \ast IQR$ treated as outliers.

The concept of quartiles and IQR can best be visualized from the boxplot. It has the minimum and maximum point defined as $Q1 - 1.5 \ast IQR$ and $Q3 + 1.5 \ast IQR$ respectively. Any point outside this range is outlier.

!!! failure "Cons"

    It delete your many data point because even if there is only one data point in a row is act as outlier for their respective column then the row is being removed which means to remove one outlier you removed many essential data point from your dataset.

```python
import pandas as pd

def apply_iqr_deletion(df: pd.DataFrame, columns: list[str], *tiles: tuple[float, float]):
    """
    Deletes outliers from given columns which are out of range of minimum and maximum percentile values.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be cleaned.
    columns: list[str]
        List of columns to be cleaned.
    *tiles: tuple[float, float]
        Tuples of (minimum percentile, maximum percentile) for each column.

    Returns
    -------
    DataFrame with deleted outliers data points.

    Raises
    ------
    ValueError: If `len(columns) != len(tiles)`.
    """
	if len(columns) != len(tiles):
		raise ValueError('len(columns) != len(tiles)')

	for col, tile in zip(columns, tiles):
		mini, maxi = df[col].quantile[tile]
		df = df[(df[col]>mini) & (df[col]<maxi)]
	return df
```

!!! summary

    If you have less number of outliers in your data then apply `apply_iqr_deletion` function but if you have many outliers than a **threshold value** then use `apply_iqr_capping` function to cap the outliers within a range.

### Z-Score

This method assumes that **the variable has a Gaussian distribution**. It represents the number of standard deviations an observation is away from the mean.

In this method we calculate the z-score with $Z = \frac{(x_i - \bar{x})}{\sigma}$ of the feature then set a threshold (generally as ±3) then remove the data point which are $\ge 3$ and $\le -3$.

!!! tip

    You can also **calculate absolute value of every z-score** then just one constraint is required as $\ge 3$.

!!! failure "Cons"

    - It deletes the rows which contains outlier which leads to data loss. And generally, losing the data is not good because it creates bias in the model and you doesn't inference well.

```python
import numpy as np
import pandas as pd
from scipy import stats

def apply_zscore_deletion(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """
    Deletes outliers from given columns using z-score method.

    Args:
        df: pd.DataFrame
            DataFrame to be cleaned.
        columns: List[str]
            List of columns to be cleaned.
        threshold: float
            Threshold for z-score.

    Returns:
        pd.DataFrame
            DataFrame with deleted outliers data points.
    """

    if not isinstance(threshold, (float, int)):
        raise TypeError("Threshold must be a float or integer value.")

    z_scores = np.abs(stats.zscore(df[columns]))
    df = df[columns][z_scores <= threshold]
    return df
```

!!! summary

    - It uses _mean and standard deviation_ of the population data which is generally not available so we need to **apply hypothesis testing** to ensure that sample mean and sample standard deviation is being used instead of population parameters.

!!! warning "Doubt"

    - Why do we calculate Z-Score because it requires population standard deviation which is not available for every for every datasets.
    - We should use T-Score instead.

## Capping Based Approach

### Winsorization

It is a way to minimise the influence of outliers in your data by either:

- Assigning the outlier a lower weight.
- Changing the value so that it is close to other values in the set.

!!! success "Pros"

    - It doesn't delete the rows where outliers lie instead it clip those outliers with your defined percentile values for each column.

!!! failure "Cons"

    - If there is many outlier values in the column/feature then after clipping the distribution of column/feature will change.

```python
import pandas as pd

def apply_winsorization(df: pd.DataFrame, columns: list[str], *tiles: tuple[float, float]):
    """
    Caps outliers in given columns to the minimum and maximum percentile values.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be cleaned.
    columns: list[str]
        List of columns to be cleaned.
    *tiles: tuple[float, float]
        Tuples of (minimum percentile, maximum percentile) for each column.

    Returns
    -------
    DataFrame with capped outliers data points.

    Raises
    ------
    ValueError: If `len(columns) != len(tiles)`.
    """
	if len(columns) != len(tiles):
		raise ValueError('len(columns) != len(tiles)')

	for col, tile in zip(columns, tiles):
		mini, maxi = df[col].quantile[tile]
		df[col] = df[col].clip(mini, maxi)
	return df
```

!!! summary

    Use this method because it uses capping technique to handle outliers.

???+ abstract "Important Links"

    - :simple-medium:&nbsp; [Detecting and Treating Outliers | How to Handle Outliers](https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/)
    - :simple-medium:&nbsp; [Detect and Remove the Outliers in a Dataset | by Dilip Valeti | Medium](https://medium.com/@dilip.voleti/detect-and-remove-the-outliers-in-a-dataset-1398f4cc7b44)
