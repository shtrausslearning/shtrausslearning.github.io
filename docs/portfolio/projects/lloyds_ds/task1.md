---
title: Task 1 - Data gathering and exploratory analysis
hide_comments: false
---

# Let's get started

As you step into the role of a data science graduate at Lloyds Banking Group, you're immediately thrust into a real-world scenario with significant implications for our business. The Data Science & Analytics team, under the leadership of Li, a seasoned senior data scientist, is currently grappling with a critical project: predicting customer churn to enhance retention strategies.

Li briefs you on the situation, "We've observed a worrying trend of customers, particularly young professionals and small business owners, leaving for competitors. This project aims to reverse that trend by identifying at-risk customers and implementing targeted interventions."

Your task is crucial. You'll begin by gathering and analysing data to understand the factors contributing to customer churn to uncover actionable insights that can inform strategic decisions. The pressure is on, as SmartBank, a key subsidiary, has reported a decline in retention rates, and there's mounting pressure from senior management to deliver solutions swiftly.

Li emphasises the importance of this task, noting, "Our findings will directly impact the strategies we deploy to retain our customers. We need accurate, insightful analysis to inform these strategies."

You're not alone in this; the team is here to guide and support you. This is your opportunity to apply your skills, contributing to a project that could shape the future of customer engagement at Lloyds.

<br>

# Techniques for identifying and collecting data

#### **Understanding the data landscape**

The first step in data collection is to understand the landscape of available data. In a corporate environment, data can come from a variety of sources, including internal databases, customer relationship management (CRM) systems, financial records, web analytics, and external data sets. Knowing where data resides and how it can be accessed is crucial for gathering relevant information efficiently.

#### **Defining data requirements**

Clearly define what you need to know before you begin. To predict customer churn, identify key variables such as customer demographics, transaction history, customer service interactions, and usage patterns. This will help focus your efforts on collecting the most relevant and useful data for your analysis.

#### **Data collection methods**

Data can be collected through several methods:

- **Primary data collection:** This involves gathering data directly from the source. For example, conducting surveys and interviews, or direct observation. While primary data is specific and relevant, it can be time-consuming and costly to obtain.
- **Secondary data collection:** Involves using existing data collected for another purpose. This could include historical data, third-party data sets, or data obtained from public sources. Secondary data is often easier and quicker to access but may require more careful consideration to ensure its relevance and accuracy.
- **Automated data collection:** Using tools such as web scraping, application programming interfaces (APIs), or data integration platforms to automatically gather data from online sources or databases. This method is efficient for collecting large volumes of data and keeping it up-to-date.

#### **Evaluating data quality**

Once data is collected, it's essential to evaluate its quality. Consider the following aspects:

- **Accuracy:** Ensure that the data accurately reflects the real-world scenarios you are studying.
- **Completeness:** Check for missing values or incomplete records that might skew your analysis.
- **Consistency:** Ensure that data is consistent across different sources and time periods.
- **Timeliness:** Data should be current and relevant to the time frame of your analysis.

#### **Data integration and storage**

After collecting data from various sources, the next step is to integrate it into a cohesive data set. This may involve merging data sets, transforming data formats, or cleaning data to ensure uniformity. Once integrated, store the data securely, ensuring that it is organised and easily accessible for analysis.

By mastering these techniques, you'll be well-prepared to gather and utilise data effectively, setting the stage for a successful data analysis project.

<br>

# Methods for performing exploratory data analysis to uncover patterns and insights

Exploratory data analysis (EDA) is a crucial stage in the data science process. It allows you to understand the underlying configuration of your data and identify key patterns and relationships. EDA involves using statistical techniques and data visualisation to summarise the main characteristics of the data set.

#### **Descriptive statistics**

Begin your EDA by calculating descriptive statistics, which provide a summary of the basic features of your data. Key metrics include:

- **Mean, median, and mode:** These measures of central tendency help you understand the typical value in your data set.
- **Standard deviation and variance:** These metrics indicate the spread or dispersion of your data, showing how much variation exists from the average.
- **Min, max, and range:** These values provide insights into the bounds of your data, highlighting any potential outliers.

#### **Data visualisation**

Visualisation is a powerful tool in EDA, helping you to quickly grasp complex data distributions and relationships. Common visualisation techniques include:

- **Histograms and density plots:** Useful for understanding the distribution of a single variable, including its central tendency and spread.
- **Box plots:** Help identify the spread and outliers in your data by showing the quartiles and median.
- **Scatter plots:** Ideal for examining relationships between two continuous variables, helping you identify potential correlations or trends.
- **Bar charts and heatmaps:** Useful for categorical data, showing the frequency or proportion of categories and the relationships between categorical variables.

#### **Correlation analysis**

To uncover relationships between variables, perform a correlation analysis. This involves calculating correlation coefficients, such as Pearson’s or Spearman’s, which quantify the strength and direction of the relationship between variables. Understanding these correlations can help you identify key predictors of customer churn.

#### **Data profiling and anomaly detection**

Data profiling involves examining data for anomalies, missing values, or inconsistencies. This phase is essential for ensuring data quality before proceeding to model building. Look for patterns in missing data, which could indicate underlying data collection issues or important missing information.

#### **Hypothesis generation**

Based on the insights gained from descriptive statistics and visualisations, formulate hypotheses about your data. For example, you might hypothesise that high customer service interaction frequency is related to increased churn rates. These hypotheses can guide your subsequent analysis and model-building efforts.

Employing these methods can help you uncover critical insights from your data, guiding your understanding and informing your decision-making process as you prepare for more advanced data analysis stages.

<br>

# Best practices for cleaning and preparing data for machine learning models

Properly cleaning and preparing data is essential for building reliable and accurate machine learning models. This process ensures that the data used in model training is of high quality, which directly impacts the model's performance.

#### **Handling missing data**

Missing data is a typical problem that can significantly affect your model's accuracy. Here are a few strategies to address this problem:

- **Imputation:** Replace missing values with a statistical measure such as the mean, median, or mode of the column. For categorical variables, the most frequent category can be used.
- **Deletion:** Remove rows or columns with missing values, particularly if the proportion of missing data is small. However, this should be done cautiously to avoid losing valuable information.
- **Flagging:** Create a new binary column that flags whether data was missing in the original data set. This can help your model learn if the absence of data is itself informative.

#### **Outlier detection and treatment**

Outliers can skew the results of your machine-learning model. Use visualisation techniques like box plots or statistical methods to detect outliers. Once identified, you can:

- **Remove outliers:** If they are caused by data entry errors or are not relevant to the analysis.
- **Cap outliers:** Set a threshold beyond which data is capped. This technique minimises the influence of extreme values without removing data points entirely.

#### **Normalisation and standardisation**

Data normalisation and standardisation are techniques used to ensure that numerical features contribute equally to the model's learning process. These processes involve:

- **Normalisation:** Rescaling the values of numeric features to a common scale, typically [0, 1]. This is useful when features have different units or scales.
- **Standardisation:** Transforming data to have a mean of zero and a standard deviation of one. This process is particularly useful when the data follows a Gaussian distribution.

#### **Encoding categorical variables**

Machine learning models require numerical input, making it necessary to convert categorical data into numerical form. Common methods include:

- **One-hot encoding:** Creating binary columns for each category in a categorical feature. This method prevents the model from assuming any ordinal relationship between categories.
- **Label encoding:** Converting each category to a numerical value. This method is simpler but should be used with caution as it can imply an ordinal relationship where none exists.

#### **Feature engineering and selection**

Creating new features from the existing data (feature engineering) and selecting the most relevant features (feature selection) can significantly improve model performance. Techniques include:

- **Creating interaction features:** Combining two or more features to capture interactions.
- **Feature scaling:** Adjusting the range of features to ensure they contribute equally to the model.
- **Dimensionality reduction:** Using methods such as principal component analysis (PCA) to reduce the number of features, which can improve model performance and reduce overfitting.

By adhering to these best practices, you'll ensure that your data set is clean, well-prepared, and optimised for building effective machine learning models. This foundational work is crucial for achieving accurate and reliable predictions in your project.

<br>

# Data gathering and exploratory analysis

Now that you've been introduced to the scenario and the importance of this project, it's time to roll up your sleeves and get started. This task will challenge you to apply your data science skills in a real-world context, helping you connect theoretical knowledge and practical application.

#### **Data collection: start with relevance**

Your first step is to identify and gather relevant data that will provide insights into customer churn. Focus on data that can help you understand customer behaviour, such as demographics, transaction history, and customer service interactions. Remember, the goal is to collect data that is pertinent to the problem at hand. Approach this step with a critical eye, considering how each piece of data might contribute to understanding why customers are leaving.

#### **EDA: discover patterns and insights**

Once you've collected the data, the next step is to perform EDA. This phase is crucial as it helps you uncover patterns, identify anomalies, and understand the data's structure. Use visual tools such as histograms, scatter plots, and heat maps to explore relationships between variables. Pay close attention to trends that might indicate early signs of churn, such as decreased usage frequency or increased interaction with customer service.

EDA is not just a technical exercise; it's an opportunity to hypothesise the underlying causes of churn. For example, if you notice that customers who engage less with your digital services are more likely to churn, this insight could inform targeted retention strategies. Your analysis should be thorough and well-documented, providing a clear narrative that connects your findings to potential business actions.

#### **Data cleaning and preparation: ensuring quality**

The quality of your data affects the reliability of your predictive model. In this stage, focus on cleaning and preparing your data set. Handle missing values appropriately, either through imputation or removal, and ensure that all variables are in a consistent format. Normalising or standardising your data may be necessary, especially if the data includes variables with different scales. This step is about precision and care; small errors can lead to significant inaccuracies in your model.

As you work through these activities, keep the broader project goals in mind. Your findings from EDA and your cleaned data set will form the foundation for building a robust predictive model. This model will help SmartBank not only understand current churn rates but also anticipate and mitigate future risks, directly influencing customer retention strategies.

#### **Why accuracy and thorough understanding matter**

Your work in this task is not just about completing an assignment; it's about developing a deep, practical understanding of data science in action. Accuracy and thorough understanding are crucial, as the insights you derive will inform strategic decisions at SmartBank. The quality of your analysis could mean the difference between retaining valuable customers and losing them to competitors.

Approach each step with diligence and an analytical mindset. This is your opportunity to make a tangible impact on a real-world problem, honing your skills in the process. Embrace the challenge, knowing that your contributions are vital to the project's success.

<br>

# Task instructions

**Introduction**

In this task, you will take the first critical steps toward building a predictive model for customer churn. Your work will involve gathering relevant data, conducting EDA, and preparing the data set for model development. These activities are foundational for ensuring the accuracy and reliability of your subsequent analysis and predictions.

**Instructions**

**Identify and gather data:**

- Review the provided data sources and select those most relevant for predicting customer churn. Focus on key areas such as customer demographics, transaction history, and customer service interactions.
- Document your selection criteria and rationale for choosing each data set, ensuring that the data will provide meaningful insights into customer behaviour.

**Perform EDA:**

- Use statistical techniques and data visualisation tools to explore the data sets. Create visualisations such as histograms, scatter plots, and box plots to understand distributions, trends, and relationships between variables.
- Identify key features that may influence customer churn, paying special attention to patterns or anomalies that could be significant.

**Clean and preprocess the data:**

- Handle missing values by choosing appropriate methods such as imputation, removal, or flagging. Justify your chosen method based on the data and context.
- Detect and address outliers that could skew the analysis or predictions. Decide whether to cap, transform, or remove outliers based on their nature and potential impact.
- Standardise or normalise numerical features to ensure consistent scales across variables. This step is crucial for preparing the data for machine learning algorithms.
- Encode categorical variables using techniques like one-hot encoding to transform them into a numerical form appropriate for analysis.

**Deliverable:**

- **File submission:** Submit a comprehensive report detailing your data gathering, EDA, and data cleaning processes. The report should include:
    - A summary of the data sets selected and the rationale for their inclusion
    - Visualisations and statistical summaries from the EDA
    - A description of the data cleaning and preprocessing steps taken
    - The cleaned and preprocessed data set ready for model building

Ensure that your report is clear, concise, and well-organised, as it will be a key component of the project's success, guiding future analysis and model development.

---

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**
