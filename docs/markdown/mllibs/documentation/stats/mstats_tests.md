
## **Module Group**

src/stats[^1]

## **Project Stage ID**

[^1]: Reference to the sub folder in `src`

4[^2]

[^2]: Reference to the machine learning project phase identification defined [here](../../projects/mlproject.md)

## :material-frequently-asked-questions: **Purpose**

The purpose of this module is to provide the user with the ability to do **data sample comparison tests** that are available in **statsmodels** and **scipy.stats** libraries. The module requires **list or parameter value** for sample comparison to test statistical hypotheses and is not aimed at dataframe based data

## :fontawesome-solid-location-arrow: **Module Files**

Here are the locations of the relevant files associated with the module

<h4>module information</h4> 

/src/stats/mstats_tests.json

<h4>module activation functions </h4>

/src/stats/mstats_tests.py

## :material-import: **Requirements**

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
from scipy import stats
from scipy.stats import kstest, shapiro, chisquare, jarque_bera, f_oneway
from statsmodels.stats.diagnostic import lilliefors
import pingouin as pg
```

## :material-selection-drag: **Selection**

Activation functions need to be assigned a unique label. Here's the process of `label` & activation function selection 

```python
    # select activation function
    def sel(self,args:dict):
                
        self.args = args
        select = args['pred_task']
        self.data_name = args['data_name']
        self.subset = args['subset']
        print('subset',self.subset)

        # [t-tests]
        
        if(select == 'its_ttest'):
            self.its_ttest(args)
        if(select == 'p_ttest'):
            self.paired_ttest(args)
        if(select == 'os_ttest'):
            self.os_ttest(args)

        # [u-test] [anova]

        if(select == 'utest'):
            self.utest(args)
        if(select == 'two_sample_anova'):
            self.two_sample_anova(args)

        # [check] Kolmogorov Smirnov Tests

        if(select == 'ks_sample_normal'):
            self.kstest_onesample_normal(args)
        if(select == 'ks_sample_uniform'):
            self.kstest_onesample_uniform(args)
        if(select == 'ks_sample_exponential'):
            self.kstest_onesample_exponential(args)

        # [check] Normality distribution checks

        if(select == 'lilliefors_normal'):
            self.lilliefors_normal(args)
        if(select == 'shapirowilk_normal'):
            self.shapirowilk_normal(args)
        if(select == 'jarque_bera_norma'):
            self.jarquebera_normal(args)

        # [check] chi2 tests

        if(select == 'chi2_test'):
            self.chi2_test(args)
        if(select == 'chi2_peng'):
            self.chi2_test_peng(args)
```

## :octicons-code-16: **Activation Functions**

Here you will find the relevant **activation functions** available in class `mstats_tests`

### :octicons-file-code-16: ==its_ttest==

<h4>subgroup:</h4>

T-Test

<h4>description:</h4>

Independent two sample **Student's t-test**: This test is used to compare the **means of two independent samples**. It assumes that the data is (normally distributed) and that the (variances of the 
two groups are equal)

<h4>code:</h4>

```python linenums="1"
# [independent two sample t-test]

# Student's t-test: This test is used to compare the means of (two independent samples) 
# It assumes that the data is (normally distributed) and that the (variances of the 
# two groups are equal)

def its_ttest(self,args:dict):

    statistic, p_value = stats.ttest_ind(args['data'][0], args['data'][1])

    print("T-statistic:", statistic)
    print("P-value:", p_value)

    # Compare p-value with alpha
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")
```

### :octicons-file-code-16: ==paired_ttest==

<h4>subgroup:</h4>

T-Test

<h4>description:</h4>

A paired **Student's t-test** is a statistical test used to determine if there is a significant difference between the **means of two related samples**. It is used when the data sets are paired or matched in some way, such as when the same group of subjects is measured before and after a treatment or intervention.

<h4>code:</h4>

```python linenums="1"
# [paired t-test]

# This test is used when you have paired or matched observations.
# It is used to determine if there is a significant difference between 
# the means of two related groups or conditions.

def paired_ttest(self,args:dict):

    print('[note] perform a paired two-sample t-test is used to compare the means of (two related groups)!')

    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(args['data'][0], args['data'][1])

    print("T-statistic:", statistic)
    print("P-value:", p_value)

    # Compare p-value with alpha
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")
```

### :octicons-file-code-16: ==os_ttest==

<h4>subgroup:</h4>

T-Test

<h4>description:</h4>

A one sample **Student's t-test** is a statistical test used to determine if there is a significant **difference between the mean of a sample and a known or hypothesized population mean**. It is used when you have one sample of data and want to compare its mean to a specific value.

<h4>code:</h4>

```python linenums="1"
# [one sample t-test]

# This test is used when you want to compare the mean of a single group to a known population mean or a specific value.

def os_ttest(self,args:dict):

    if(args['popmean'] != None):

        # Perform one-sample t-test
        t_statistic, p_value = stats.ttest_1samp(args['data'], popmean=args['popmean'])

        print("t-statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis")

    else:

        print('[note] please specify the population mean using popmean')
```

### :octicons-file-code-16: ==utest==

<h4>subgroup:</h4>

distribution check

<h4>description:</h4>

The **Mann-Whitney test**, also known as the **Wilcoxon rank-sum test**, is a nonparametric statistical test used to determine whether there is a significant **difference between the distributions of two independent samples**. It is often used when the data does not meet the assumptions of parametric tests like the **t-test**

<h4>code:</h4>

```python linenums="1"
# determine if there is a significant difference between the distributions

# A : [u-test]

# The [Mann-Whitney test], also known as the [Wilcoxon rank-sum test], 
# is a nonparametric statistical test used to determine whether there 
# is a significant difference between the distributions of two independent samples. 
# It is often used when the data does not meet the assumptions of parametric tests 
# like the t-test.

def utest(self,args:dict):

    # Perform Mann-Whitney U-test
    statistic, p_value = stats.mannwhitneyu(args['data'][0], args['data'][1])

    print("U-statistic:", statistic)
    print("P-value:", p_value)

    # Compare p-value with alpha
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")
```

### :octicons-file-code-16: ==kstest_twosample==

<h4>subgroup:</h4>

distribution check

<h4>description:</h4>

The **Kolmogorov-Smirnov test** is a nonparametric statistical test that **determines whether a sample comes from a specific distribution**. It compares the empirical cumulative distribution function (ECDF) of the sample to the cumulative distribution function (CDF) of the specified distribution

<h4>code:</h4>

```python linenums="1"
# [GENERAL] Kolmogorov Smirnov Test Two Sample Test for distribution

def kstest_twosample(self,args:dict):

    # Perform the KS test
    statistic, p_value = kstest(args['data'][0], args['data'][1])

    print('[KS] test two samples from sample distribution')
    print("KS statistic:", statistic)
    print("P-value:", p_value)
```

### :octicons-file-code-16: ==kstest_onesample_normal==

<h4>subgroup:</h4>

distribution check

<h4>description:</h4>

The **Kolmogorov-Smirnov** test for a **normal distribution** is a statistical test that determines whether a sample of data comes from a normal distribution

<h4>code:</h4>

```python linenums="1"
# Perform Kolmogorov-Smirnov test for [normal] distribution

def kstest_onesample_normal(self,args:dict):

    statistic, p_value = kstest(args['data'], 'norm')

    print('[KS] test sample from (normal) distribution')
    print("KS statistic:", statistic)
    print("P-value:", p_value)

    # Compare p-value with alpha (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")
```

### :octicons-file-code-16: ==kstest_onesample_uniform==

<h4>subgroup:</h4>

distribution check

<h4>description:</h4>

The **Kolmogorov-Smirnov** test for a **uniform distribution** is a statistical test that determines whether a sample of data comes from a uniform distribution

<h4>code:</h4>

```python linenums="1"
# Perform Kolmogorov-Smirnov test for [Uniform] distribution

def kstest_onesample_uniform(self,args:dict):

    statistic, p_value = kstest(args['data'], 'uniform')

    print('[KS] test sample from (uniform) distribution')
    print("KS statistic:", statistic)
    print("P-value:", p_value)

    # Compare p-value with alpha (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")
```

### :octicons-file-code-16: ==kstest_onesample_exponential==

<h4>subgroup:</h4>

distribution check

<h4>description:</h4>

The **Kolmogorov-Smirnov** test for a **exponential distribution** is a statistical test that determines whether a sample of data comes from a exponential distribution

<h4>code:</h4>

```python linenums="1"
# Perform Kolmogorov-Smirnov test for [Exponential] distribution

def kstest_onesample_exponential(self,args:dict):

    statistic, p_value = kstest(args['data'], 'expon')

    print('[KS] test sample from (exponential) distribution')
    print("KS statistic:", statistic)
    print("P-value:", p_value)

    # Compare p-value with alpha (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")
```

### :octicons-file-code-16: ==lilliefors_normal==

<h4>subgroup:</h4>

distribution check

<h4>description:</h4>

The **Lilliefors test**, also known as the **Kolmogorov-Smirnov test for normality**, is a statistical test used to determine whether a sample of data comes from a **normal distribution**. It is similar to the Kolmogorov-Smirnov test, but it is specifically designed for testing against a normal distribution.

<h4>code:</h4>

```python linenums="1"
# Lilliefors Test to check if distribution is normal distribution

def lilliefors_normal(self,args:dict):

    # Perform the Lilliefors test
    statistic, p_value = lilliefors(args['data'])

    print("Lilliefors test statistic:", statistic)
    print("Lilliefors p-value:", p_value)
        
    # Compare p-value with alpha (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis") 
```

### :octicons-file-code-16: ==shapirowilk_normal==

<h4>subgroup:</h4>

distribution check

<h4>description:</h4>

The **Shapiro-Wilk** test is another statistical test used to determine whether a sample of data comes from a **normal distribution**

<h4>code:</h4>

```python linenums="1"
# Shapiro-Wilk Test to check if distribution is normal

def shapirowilk_normal(self,args:dict):

    # Perform Shapiro-Wilk test
    statistic, p_value = shapiro(args['data'])

    # Print the test statistic and p-value
    print("Test Statistic:", statistic)
    print("P-value:", p_value)

    # Compare p-value with alpha (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis") 
```

### :octicons-file-code-16: ==chi2_test==

<h4>subgroup:</h4>

-

<h4>description:</h4>

The **chi-square test** is a statistical test used to determine if there is a significant association between two **categorical variables**

<h4>code:</h4>

```python linenums="1"
# [Chi2 statistical test]

# Calculate a one-way chi-square test
# The chi-square test is a statistical test used to determine 
# if there is a significant association between two categorical variables.

# chi-square statistic measures how much the observed frequencies deviate 
# from the expected frequencies. A higher value indicates a greater discrepancy.

def chi2_test(self,args:dict):

    # perform the chi-squared test
    statistic, p_value = chisquare(args['data'][0], f_exp=args['data'][1])

    print("Chi-squared statistic:", statistic)
    print("P-value:", p_value)

    # Compare p-value with alpha (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis") 
```

### :octicons-file-code-16: ==jarquebera_normal==

<h4>subgroup:</h4>

distribution check

<h4>description:</h4>

The **Jarque-Bera test** is a statistical test used to determine whether a given dataset follows a normal distribution. It is based on the **skewness** and **kurtosis** of the data

<h4>code:</h4>

```python linenums="1"
# [ Jarque-Bera test ]

# The Jarque-Bera test is a statistical test used to determine whether 
# a given dataset follows a normal distribution. It is based on the 
# skewness and kurtosis of the data. 

def jarquebera_normal(self,args:dict):

    # Perform the Jarque-Bera test
    statistic, p_value = stats.jarque_bera(args['data'])

    print('Statistic:", statistic')
    print("P-value:", p_value)

    # Compare p-value with alpha (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis") 
```

### :octicons-file-code-16: ==two_sample_anova==

<h4>subgroup:</h4>

-

<h4>description:</h4>

The **ANOVA (Analysis of Variance) test** is used to determine if there are any statistically significant **differences between the means** of two or more groups

<h4>code:</h4>

```python linenums="1"
# [ ANOVA test ] (limited to two samples)

# ANOVA (Analysis of Variance) test is used to determine if there are any statistically significant differences between the (means) of two or more groups

def two_sample_anova(self,args:dict):

    # Perform one-way ANOVA test
    statistic, p_value = stats.f_oneway(args['data'][0], args['data'][1])

    # Print the results
    print("Statistic:", statistic)
    print("p-value:", p_value)

    # Compare p-value with alpha (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis") 
```
