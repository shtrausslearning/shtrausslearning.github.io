---
comments: true
---

**Natural language processing (NLP)** is a branch of artificial intelligence (AI) that deals with the interaction between computers and humans using natural language. It involves the development of algorithms and computational models that can understand, analyze, and generate human language. 

<br>


<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Banking Consumer Complaint Analysis](https://www.kaggle.com/code/shtrausslearning/banking-consumer-complaint-analysis)</b>

	---

	In this study, we aim to create an **automated ticket classification model** for incoming text based complaints, which is a **multiclass classification problem**. 
	
	- Such a model is useful for a company in order to automate the process of sorting financial product reviews & subsequently pass the review to an experient in the relevant field. 
	- We explore traditional ML methods, which utilise hidden-state **BERT embedding** for features, as well as fine-tune **DistilBert** for our classification problem & compare the two approaches, comparing the model accuracies of both approaches

</div>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[News sentiment based trading strategy](https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy)</b>

	---

	In this project, we'll apply **NLP** to financial stock movement prediction. Using **NLP**, we can ask ourselves questions such as, how positive or negative a **news article** (related to financial markets is). It provides a way to monitor **financial market sentiments** by utilising any text based source so we can determine whether the text based source posted on specific day has a positive or negative sentiment score. By combining **historic market data** with **news sources related to financial markets**, we can create a trading strategy that utilises NLP. The whole project revolved around generating accurate **sentiment labels** that would correlate to **event returns**

	Based on historic data, we can calculate the **return** of a specific event, however one of challenges to utilise NLP for such application are the **target labels** and the **ground truths** would be set as the even return direction. We first need to create a model that is able to accurately define the sentiment of the **news source**, to do this we try a couple of different approaches: 

	- The first method, we **manually define labels** and evaluate the performance of the model. The manual approach utilised three strategies combined into one (percentage value extraction, **TextBlob** & Beat/Misses). For encoding, we utilised static **spacy word embeddings** & investigated how the dimensionality of the vectors affected the model accuracy.
	- We also utilised an expertly labeled dataset & tested the resulting model on the dataset, however there wasn't a too significant increase in accuracy.

	The best performance boost came from the utilisation of Deep Learning **LSTM** with a trainable **embedding laber** architecture, which showed much better generalisation performance than classical machine learning models, including **ensemble methods**

	The last approach we tried as **VADER**, which allows us to utilise a **custom lexicon**, which we can change to something more related: **[**financial markets**](https://www.sciencedirect.com/science/article/abs/pii/S0167923616300240)**. It was interesting to note that the VADER approach resulted in a high postive correlation to **event return**

</div>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Twitter Emotion Classification](https://www.kaggle.com/code/shtrausslearning/twitter-emotion-classification)</b>

	---

	In this study, we fine-tune a transformer model so it can classify the **sentiment** of user tweets for **6 different emotions** (multiclass classification). 
	
	- We first create a baseline by utilising traditional ML methods, which for features use extracted **BERT** embeddings for each sentence. 
	- Once we have our baseline model, we then turn to more complex transformer models, **DistilBert** & **fine-tune** its model weights for our classification problem


</div>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[edX Course Recommendations](https://www.kaggle.com/code/shtrausslearning/nlp-edx-course-recommendations)</b>

	---

	In this study, we create an **NLP based recommendation system** which informs a user about possible courses they make like, based on a couse they have jusy added
	
	-  We will utilise **[scrapped edX](https://www.kaggle.com/datasets/khusheekapoor/edx-courses-dataset-2021)** course description data, clean the text data and then convert document into vector form using two different approaches BoW based **TF-IDF** and **word2vec**,
	- Then utilising **consine similarity** we calculate the most relevant courses which we can recommendand

</div>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Creating a Transformer Attention Encoder](https://www.kaggle.com/code/shtrausslearning/creating-a-transformer-attention-encoder?scriptVersionId=149696179n)</b>

	---

	In this study, we look at some of the basics of a **transformer** architecture model, the **encoder**, by writing and utilising custom **pytorch** classes. Encoder simply put: Converts a **series tokens** into a **series of embedding vectors** (hidden state) & consists of **multiple layers** (**blocks**) constructed together 

	The **encoder structure**:

	- Composed of **multiple encoder layers (blocks)** stacked next to each other (similar to CNN layer stacks)
	- Each encoder block contains **multi-head self attention** & **fully connected feed forward layer** (for each input embedding)

	Purpose of the Encoder:

	- Input tokens are encoded & modified into a form that **stores some contextual information** in the sequence

	The basis of the encoder can be utilised for a number of different applications, as is common in **HuggingFace**, we'll create a simple tail end classification class, so the model can be utilised for **classification**

</div>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Banking User Review Analysis & Modeling]()</b>

	---

	<h4> 1 | Parsing Dataset | :simple-github:{ .lg .middle }&nbsp; <b>[Github Notebook](https://github.com/shtrausslearning/otus_nlp_course/blob/main/hw/1_parsing.ipynb)</b></h4>

	In this study we look at the **parsing/scraping** side of data, extracting the customer review data 
	
	- Its no secret that a lot text important information is stored on websites, as a result, for us to utilise this data in our of analyses and modeling, we need a way to extract this information, this process is referred to website parsing. 
	- For our study we need to extract customer user reviews from **[irecommend](https://irecommend.ru/content/sberbank?new=50)**. 
	- We'll be parsing a common **banking coorporation** that offers a variety of services so the reviews aren't too specific to a particular product. 
	- Having parsed our dataset, we'll follow up this with a rather basic **exploratory data analysis** based on **ngram** word combinations, so we can very quickly understand the content of the entire corpus.

	<h4> 2 | Banking Product Review Sentiment Modeling | :simple-github:{ .lg .middle }&nbsp; <b>[Github Notebook](https://github.com/shtrausslearning/otus_nlp_course/blob/main/hw/3_product-reviews.ipynb)</b></h4>

	Once we have parsed and created our dataset, we look at creating a **sentiment model** based on traditional NLP machine learning approaches. 
	
	- We will be using the parsed dataset about **bank service** reviews, which consists of ratings as well as recommend/don't recommend type labels.
	- We'll be using **TF-IDF** & **Word2Vec** methods to encode text data & use typical shallow and deep tree based enseble models. 
	- Once we have found the best performing approaches, we'll be doing a brute force based **GridSearchCV** hyperparameter optimisation in order to tune our model. After selecting the best model, we'll make some conclusions about our predicts & make some future work comments.

</div>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[mllibs](ttps://github.com/mllibs)</b>

	***

	**mllibs** is a project aimed to automate various processes using text commands
	
	- Development of such helper modules are motivated by the fact that everyones understanding of coding & subject matter (ML in this case) may be different
	- Often we see people create functions and classes to simplify the process of **code automation** (which is good practice)
	- Likewise, NLP based interpreters follow this trend as well, except, in this case our only inputs for activating certain code is natural language
	- Using python, we can interpret natural language in the form of string type data, using natural langauge interpreters
	mllibs aims to provide an automated way to do machine learning using **natural language**

</div>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>OTUS NLP Course Work</b>

	Natural language course related work on a variety of **Natural Language Processing** topics

	#### :octicons-bookmark-16: <b><span style='color:#FFCA58;text-align:center'></span>[NER with preset tools (re,natasha)](https://github.com/shtrausslearning/otus_nlp_course/blob/main/3_%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20NLP/11_%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%20NER/preset_NER.ipynb)</b>

	In this notebook, we look at how to utilise the **natasha** & **re** libraries to do predefined **NER**  tagging 
	
	- **natasha** comes with already predefined set of classification labels, whilst **re** can be used to identify capitalised words using regular expressions. 
	- These tools, together with lemmatisers from **`pymorphy2`** allow us to very easily utilise ready instruments for named entity recognition in documents without any model training.

	---

	#### :octicons-bookmark-16: <b><span style='color:#FFCA58;text-align:center'></span>[Training a NER model with GRU](https://github.com/shtrausslearning/otus_nlp_course/blob/main/3_%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20NLP/11_%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%20NER/gru_NER.ipynb)</b>

	In this project, we train a neural network **NER** model based on **GRU** architecture, which can recognise named entities using **BIO tags** based on car user review data. 
	
	- Unlike the previous notebook, the concept of **NER** is used a little more abstractly, we are interested in any markups for word(s) that we create in the text, not just names.
	- For markups we use tags that describe the quality of the car (eg. appearance, comfort, costs, etc.). 
	- The model learns to classify tokens in the text that belong to one of the tag classes. 
	- Recognition of such labels is convenient for quick understanding of the content of the review.

	#### :octicons-bookmark-16: <b><span style='color:#FFCA58;text-align:center'></span>[Sentiment Analysis of Kazakh News](https://github.com/shtrausslearning/otus_nlp_course/blob/main/3_%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20NLP/9_%D0%9F%D1%80%D0%B5%D0%B4%D0%BE%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B0%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D0%B8%20%D0%BF%D0%BE%D0%BD%D1%8F%D1%82%D0%B8%D0%B5%20%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BD%D1%8B%D1%85%20%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D1%82%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B9%20%D1%81%D0%BB%D0%BE%D0%B2/khazah-news-sentiment.ipynb)</b>

	In this project we create create a model that is used for sentiment analysis using classical NLP + machine learning approaches 
	
	- We utilise standard baseline approaches such as **`TF-IDF`** and **`BoW`** together with **`RandomForestClassifier`** and standard **`train_test_split`** to train and evaluate the generalisation performance of the model using **`f1_score`** metric since we end up having slightly disbalanced sentiment classes

	---

	#### :octicons-bookmark-16: <b><span style='color:#FFCA58;text-align:center'></span>[Fine Tuning BERT for Multilabel Toxic Comment Classification](https://github.com/shtrausslearning/otus_nlp_course/blob/main/4_Нейросетевые%20языковые%20модели/16_Transfer%20learning%3B%20BERT%20model/multilabel-text-classification.ipynb)</b>

	In this project we will be creating a **multilabel model** for toxic comment classification using transfomer architecture **BERT**. 
	
	- This main difference between multilabel and multiclass classification is that we are treating this as a binary classification problem, but checking for multiple labels for whether the text belongs to the class or not.

	#### :octicons-bookmark-16: <b><span style='color:#FFCA58;text-align:center'></span>[Fine Tuning BERT for Linguistic Acceptability](https://github.com/shtrausslearning/otus_nlp_course/blob/main/4_%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D1%81%D0%B5%D1%82%D0%B5%D0%B2%D1%8B%D0%B5%20%D1%8F%D0%B7%D1%8B%D0%BA%D0%BE%D0%B2%D1%8B%D0%B5%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8/17_Pretrained%20Language%20Model%20Example/binary-text-classification-comment.ipynb)</b>

	In this project we will be fine-tuning a transformer model for the **language acceptability problem (CoLa)** problem. 
	
	- The **CoLa** dataset itself is a benchmark dataset for evaluating natural language understanding models. 
	- CoLa stands for "Corpus of Linguistic Acceptability" and consists of sentences from various sources, such as news articles and fiction, that have been labeled as either grammatically correct or incorrect. 
	- The dataset is commonly used to evaluate models' ability to understand and interpret the grammatical structure of sentences. 
	- For this task we'll be utilising the **bert-base-uncased** model and utilise **huggingface's** convenient downstream task task adaptation variation for **binary classification** using **BertForSequenceClassification** 

	---

	#### :octicons-bookmark-16: <b><span style='color:#FFCA58;text-align:center'></span>[Customer Service Dialogue System for GrabrFi](pdf/chatbot_grabrfi.pdf)</b>

	As part of the final project of the **[nlp course](https://otus.ru/lessons/nlp/)**, the aim of the project was to create a dialogue system for a banking service business **GrabrFi**, focusing on combining various NLP methods that can be utilised in chatbots. 
	
	- Combining a **Telegram** structure that utilises **TF-IDF** with **cosine_similarity**, **multiclass classification** based approach, **Question Answering** (BERT), **generative** (DialoGPT). 
	- The task of answering user questions and queries was split up into different subgroups found in the **[help section](https://help.grabrfi.com)** so that each model would be in charge of its own section, as a result of experimenting with different method activation thresholds, a dialogue system that utilised all of the above methods was created, and all methods were able to work together. 
	- This allowed for an understanding of the different approaches that can be utilised in the creation of a dialogue system. 

</div>


---

**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**

