
## <b>Module Information</b>

### :fontawesome-solid-layer-group: **<span style='color:#586bc9'>Module Group</span>**

Statistical Tests

### :material-frequently-asked-questions: **<span style='color:#586bc9'>Purpose</span>**

The purpose of this module is to provide the user with the ability to do **data sample comparison tests** that are available in **statsmodels** and **scipy.stats** libraries. The module requires **list or parameter value** for sample comparison to test statistical hypotheses and is not aimed at dataframe based data

A list of all available activation functions in the module :octicons-file-code-16: **mstats_tests** 


<div class="grid cards" markdown>

- :octicons-file-code-16:{ .lg .middle }&nbsp; [__its_ttest__](mstats_tests.html#its_ttest)

    --- 

    <b><span style='color:#586bc9'>data:</span> [<code>list</code>,<code>list</code>] <span style='color:#586bc9'>targ:</span><code>None</code></b>

    Independent two sample **Student's t-test**: This test is used to compare the **means of two independent samples**. It assumes that the data is (normally distributed) and that the (variances of the two groups are equal)

- :octicons-file-code-16:{ .lg .middle }&nbsp; [__paired_ttest__](mstats_tests.html#paired_ttest)

    --- 

    <b><span style='color:#586bc9'>data:</span> [<code>list</code>,<code>list</code>] <span style='color:#586bc9'>targ:</span><code>None</code></b>

    A paired **Student's t-test** is a statistical test used to determine if there is a significant difference between the **means of two related samples**. It is used when the data sets are paired or matched in some way, such as when the same group of subjects is measured before and after a treatment or intervention


- :octicons-file-code-16:{ .lg .middle }&nbsp; [__os_ttest__](mstats_tests.html#os_ttest)

    --- 

    <b><span style='color:#586bc9'>data:</span> <code>list</code> <span style='color:#586bc9'>targ:</span><code>popmean</code></b>

    A one sample **Student's t-test** is a statistical test used to determine if there is a significant **difference between the mean of a sample and a known or hypothesized population mean**. It is used when you have one sample of data and want to compare its mean to a specific value.

- :octicons-file-code-16:{ .lg .middle }&nbsp; [__utest__](mstats_tests.html#utest)

    --- 

    <b><span style='color:#586bc9'>data:</span> [<code>list</code>,<code>list</code>] <b><span style='color:#586bc9'>targ:</span></b><code>None</code></b>

    The **Mann-Whitney test**, also known as the **Wilcoxon rank-sum test**, is a nonparametric statistical test used to determine whether there is a significant **difference between the distributions of two independent samples**. It is often used when the data does not meet the assumptions of parametric tests like the **t-test**

- :octicons-file-code-16:{ .lg .middle }&nbsp; [__kstest_twosample__](mstats_tests.html#kstest_twosample)

    --- 

    <b><span style='color:#586bc9'>data:</span> [<code>list</code>,<code>list</code>] <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **Kolmogorov-Smirnov test** is a nonparametric statistical test that **determines whether a sample comes from a specific distribution**. It compares the empirical cumulative distribution function (ECDF) of the sample to the cumulative distribution function (CDF) of the specified distribution


- :octicons-file-code-16:{ .lg .middle }&nbsp; [__kstest_onesample_normal__](mstats_tests.html#kstest_onesample_normal)

    --- 

    <b><span style='color:#586bc9'>data:</span> <code>list</code> <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **Kolmogorov-Smirnov** test for a **normal distribution** is a statistical test that determines whether a sample of data comes from a normal distribution


- :octicons-file-code-16:{ .lg .middle }&nbsp; [__kstest_onesample_uniform__](mstats_tests.html#kstest_onesample_uniform)

    --- 

    <b><span style='color:#586bc9'>data:</span> <code>list</code> <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **Kolmogorov-Smirnov** test for a **uniform distribution** is a statistical test that determines whether a sample of data comes from a uniform distribution


- :octicons-file-code-16:{ .lg .middle }&nbsp; [__kstest_onesample_exponential__](mstats_tests.html#kstest_onesample_exponential)

    --- 

    <b><span style='color:#586bc9'>data:</span> <code>list</code> <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **Kolmogorov-Smirnov** test for a **exponential distribution** is a statistical test that determines whether a sample of data comes from a exponential distribution

- :octicons-file-code-16:{ .lg .middle }&nbsp; [__lilliefors_normal__](mstats_tests.html#lilliefors_normal)

    --- 

    <b><span style='color:#586bc9'>data:</span> <code>list</code> <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **Lilliefors test**, also known as the **Kolmogorov-Smirnov test for normality**, is a statistical test used to determine whether a sample of data comes from a **normal distribution**. It is similar to the Kolmogorov-Smirnov test, but it is specifically designed for testing against a normal distribution.


- :octicons-file-code-16:{ .lg .middle }&nbsp; [__shapirowilk_normal__](mstats_tests.html#shapirowilk_normal)

    --- 

    <b><span style='color:#586bc9'>data:</span> <code>list</code> <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **Shapiro-Wilk** test is another statistical test used to determine whether a sample of data comes from a **normal distribution**


- :octicons-file-code-16:{ .lg .middle }&nbsp; [__chi2_test__](mstats_tests.html#chi2_test)

    --- 

    <b><span style='color:#586bc9'>data:</span> [<code>list</code>,<code>list</code>] <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **chi-square test** is a statistical test used to determine if there is a significant association between two **categorical variables**


- :octicons-file-code-16:{ .lg .middle }&nbsp; [__jarquebera_normal__](mstats_tests.html#jarquebera_normal¶)

    --- 

    <b><span style='color:#586bc9'>data:</span> <code>list</code> <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **Jarque-Bera test** is a statistical test used to determine whether a given dataset follows a normal distribution. It is based on the **skewness** and **kurtosis** of the data

- :octicons-file-code-16:{ .lg .middle }&nbsp; [__two_sample_anova¶__](mstats_tests.html#two_sample_anova¶)

    --- 

    <b><span style='color:#586bc9'>data:</span> [<code>list</code>,<code>list</code>] <span style='color:#586bc9'>targ:</span><code>None</code></b>

    The **ANOVA (Analysis of Variance) test** is used to determine if there are any statistically significant **differences between the means** of two or more groups

</div>