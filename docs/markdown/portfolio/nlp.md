## :fontawesome-solid-language:{ .language } <b>Natural Language Processing</b> 

Natural language processing (NLP) is a branch of artificial intelligence (AI) that deals with the interaction between computers and humans using natural language. It involves the development of algorithms and computational models that can understand, analyze, and generate human language. 

### :material-label-variant-outline: <b><span style='color:#FFCA58;text-align:center'></span>Banking Consumer Complaint Analysis</b>

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://www.kaggle.com/code/shtrausslearning/customer-transaction-predictive-analytics)

In this study, we aim to create an **automated ticket classification model** for incoming text based complaints, which is a **multiclass classification problem**. Such a model is useful for a company in order to automate the process of sorting financial product reviews & subsequently pass the review to an experient in the relevant field. We explore traditional ML methods, which utilise hidden-state BERT embedding for features, as well as fine-tune DistilBert for our classification problem & compare the two approaches

![](images/banking_consumer_1.png){ width="300" } ![](images/banking_consumer_2.png){ width="300" }

### :material-label-variant-outline: <b><span style='color:#FFCA58;text-align:center'></span>Twitter Emotion Classification</b>

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://www.kaggle.com/code/shtrausslearning/twitter-emotion-classification)

In this study, we fine-tune a transformer model so it can classify the `sentiment` of user tweets for 6 different emotions (multiclass classification). We first create a baseline by utilising traditional ML methods that use extracted `BERT` embeddings for features, then we will turn to a more complex transformer encoder, `DistilBert` & `fine-tune` its model weights for our classification problem

<center>
![](images/sentiment_tsne.png)
</center>

### :material-label-variant-outline: <b><span style='color:#FFCA58;text-align:center'></span>edX Course Recommendations</b> 

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://www.kaggle.com/code/shtrausslearning/nlp-edx-course-recommendations)

In this study, we create an **NLP based recommendation system** which informs a user about possible courses they make like, based on a couse they have jusy added. We will utilise **[scrapped edX](https://www.kaggle.com/datasets/khusheekapoor/edx-courses-dataset-2021)** course description data, clean the text data and then convert document into vector form using two different approaches BoW based **TF-IDF** and **word2vec**, then calculate the **consine similarity**, from which we will be able to extract a list of courses which are most similar and so can be recommended.

<center>
	![](images/embedding.png)
</center>

### :material-label-multiple-outline: <b><span style='color:#FFCA58;text-align:center'></span>Banking User Review Analysis & Modeling</b>

#### (1) Parsing Dataset

![name](https://img.shields.io/badge/bf4-parsing-e589e5) ![name](https://img.shields.io/badge/py-requests-e589e5) [![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/shtrausslearning/otus_nlp_course/blob/main/hw/1_parsing.ipynb)

In this study we look at the **parsing/scraping** side of data. Its no secret that a lot text important information is stored on websites, as a result, for us to utilise this data in our of analyses and modeling, we need a way to extract this information, this process is referred to website parsing. For this example, we'll look at a user service review website, which stored user reviews on a variety of services and objects. We'll be parsing a common banking service & follow up with an exploratory data analysis, which should tell us about the contents of our extracted text data.

#### (2) Banking Product Review Sentiment Modeling

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/shtrausslearning/otus_nlp_course/blob/main/hw/3_product-reviews.ipynb) ![name](https://img.shields.io/badge/sklearn-parsing-e589e5) ![name](https://img.shields.io/badge/catboost-requests-e589e5)

In this notebook, we look at creating a **sentiment model** based on traditional NLP machine learning approaches. We will be using the parsed dataset about **bank service** reviews, which consists of ratings as well as recommend/don't recommend type labels. We'll be using `TF-IDF` & `Word2Vec` methods to encode text data & use typical shallow and deep tree based enseble models. Once we have found the best performing approaches, we'll be doing a brute force based `GridSearchCV` hyperparameter optimisation in order to tune our model. After selecting the best model, we'll make some conclusions about our predicts & make some future work comments.

### :material-home-group: MLLIBS

[![name](https://img.shields.io/badge/Repository-Project-blue?logo=GitHub)](https://github.com/shtrausslearning/Data-Science-Portfolio/tree/main/ANZ_internship)

mllibs is a project aimed to automate various processes using text commands. 

---

**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**

