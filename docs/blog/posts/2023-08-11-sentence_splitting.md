# Named Entity Recognition for Sentence Splitting

![](images/booktable.jpg)

---
date: 2023-08-11 12:00:00 +0000 <br>
categories: [NLP] <br>
tags: [sklearn, NER, CountVectorizer, sentence splitting]
---

> In the last post, we talked about how to use **NER** for tagging named entities using transformers. In this sections, we'll try something a little more simpler, utilising traditional encoding & ML methods. One advantage of using such models is the cost of training. We'll also look at a less common example for **NER** tagging, which I've implemented in my project **[mllibs](https://github.com/shtrausslearning/mllibs)**

## :fontawesome-solid-book: <b>Background</b>

### <b><span style='color:#6A5ACD;text-align:center'>❯❯ </span>What is NER?</b> 

So what exactly is `NER`?

- `NER` is a natural language processing technique which identifies and extracts **named entities** from unstructured text
- Named entities refer to words or combination of words that represent specific objects, places etc, in principle **it can be any word/words we define it to be**
- `NER` algorithms use Machine or Deep Learning algorithms to analyse text and recognise pattens that indicate the presence of **a named entity**


> In this notebook, we define **named entities** to be word(s) that can be defined **as sentence splitters**, which differs from how `NER` tends to be used for extraction of place names and so on
> {: .prompt-info }

### <b><span style='color:#6A5ACD;text-align:center'>❯❯ </span>Sentence splitting</b> 

What can be thought to be quite straightforward and non trivial isn't quite so. **Sentence splitting** or sentence simplification is the task of taking a sentence that is usually too long and breaking it up into two or more simpler sentences. In the next section, we'll mention one example, where it can be useful. 

Consider an example:

> The quick red fox jumped over the lazy brown dog. The dog didn't see it coming. <br>
<kbd>The quick red fox jumped over the lazy brown dog.</kbd> <kbd>The dog didn't see it coming.</kbd>

Something more difficult now:

> The quick red fox jumped over the lazy brown dog. The dog didn't see it coming. <br>
<kbd>The quick red fox jumped over the lazy brown dog</kbd> `and` `then` <kbd>ran away</kbd>

We can split the sentence based on two occuring events, which is separated by the words **and then**. By splitting the sentence into two parts we understand that there are two actions that are associated with the object `fox`. Simple tokenisers such as **[sent_tokenize](https://www.nltk.org/api/nltk.tokenize.sent_tokenize.html)** from `NLTK` or any other sentence tokeniser are not suited for tokenising the second example, which is where **NER** comes in.

### <b><span style='color:#6A5ACD;text-align:center'>❯❯ </span>Motivation</b> 

#### Classifying user requests

Let's say we have a problem in which we need to **classify a user request** into two groups; <kbd>class 0</kbd> (impute data) & <kbd>class 1</kbd> (standardise columns) using text input data, a **binary classification problem**. The classification model will activate specific activation functions, so incorrectly classified models will activate the wrong function! 

Hopefully you are familiar with text preprocessing & training using sklearn, because that's what we'll be using in this post. We train a `LogisticRegression` binary classification model and the model itself is complex enough to get a perfect score on the dataset it was trained on. 

```python

# commands corresponding to activation function
train = ["impute missing, mean value",
         "impute missing, average value",
         "fill missing, mean value",
         "standardise columns",
         "normalise columns",
         "standardise"
         ]

# target labels
y_train = ['imp-mean','imp-mean','imp-mean','std','std','std']

# encoder data
def tokenizer(text):
    return word_tokenize(text)

# Bag of Words Encoding
encoder = CountVectorizer()
X_train = encoder.fit_transform(train)

# Train model to predict between two classes
model = LogisticRegression()
model.fit(X_train,y_train)
```

#### Using Classifier

Now let's assume we deploy this model, we have two user requests:
-  <kbd>First of all, impute all missing data with column mean values, then standardise all columns</kbd>
-  <kbd>impute missing data with column mean values</kbd>

In the first request, we clearly have too much information for the model to understand what to do, we could actually split the request into two parts:
- First of all, <kbd>impute all missing data with column mean values</kbd>, then <kbd>standardise all columns</kbd>.

The second request contains only one request:
- <kbd>impute missing data with column mean values</kbd>

Let's check how both models would perform:

```python

# user request
request_a = "First of all, impute all missing data with column mean values and then standardise all columns."
request_b = "impute missing data with column mean values"

# test model on new data
X_test_a = encoder.transform([request_a])
X_test_b = encoder.transform([request_b])
y_pred_a = model.predict_proba(X_test_a)
y_pred_b = model.predict_proba(X_test_b)

print('text_a:',y_pred_a)
print('text_b',y_pred_b)

```

`predict_proba` gives us the following probability distribution, we are using a threshold of 0.5 which is the default for `predict_proba`, ie. for **text_a**, the model has predicted `class 1` (standardise columns) ✘, & for **text_b**, `class 0` (impute data) ✔

```
text_a: [[0.49158457 0.50841543]]  
text_b [[0.68768998 0.31231002]]
```

So we can see that the closer the content of the user request is to one of the text it was trained on, the more accurate the model. Intuitively, there clearly is a need to split the first user request into parts because the request doesn't fit into either of the two categories, so if we can tag tokens `,` & `then`, we can split the sentence into two. Having two separate sentences, we then can repredict on both of these sentences to get the correct predictions.

> Similar to if you took `cosine_similarity` between the input user request and different documents that belong to a class. If the content contains too much additional text, the similarity will start to fall.
{: .prompt-info }

#### Using NER to identify sentence split locations

We could use `regex` in an attempt to **find split locations**, and then **cut the document** at `regex` matched pattern locations, however if we go down the **NER** route, we actually can tag words in the sentence however we want, which is very convenient. Aside from sentence splitting tags such as `and then`, we would also identify words which don't add value and tag them as well (eg. `first of all`).


## :material-database-check: <b>The Dataset</b>

### <b><span style='color:#6A5ACD;text-align:center'>❯❯ </span>Tagging split tag words</b> 

The dataset will be created by us, and is by not means perfect. I used this **[useful reference](https://www.miamioh.edu/hcwe/handouts/sentence-combination/index.html#:~:text=You%20have%20four%20options%20for,moreover%2C%22%20or%20%22thus%22)** in order to try and add various writing variations. As opposed to the **grouped document token** approach I did in **[this notebook](https://www.kaggle.com/code/shtrausslearning/mllibs-ner-based-classification-sentence-split)**, I won't be adding other OOV words to **class (0)**.

I've created about 20 general sentences and annotated word/token locations at which I would like to identify a **splitting token**. We'll be using the standard `BIO` tagging system, so we can tag segments of tokens, identifying **beginning & inner tags**. 

Our dataset contains two columns `text` & its annotation `annotated` variant.

|text|annotated                    |
|--------|-----------------------------|
|Please do A and then do B using C|Please do A [split : and then] do B using C|
|Do C, then do D|Do C[split : , then] do D    |
|Do D, and then do F|Do D[split : , and then] do F|
|please do A followed by P|please do A [split : followed by] P|
|please do A, once done do C|please do A[split : , once done] do C|
|do A then do B|do A [split : then do] B     |
|please do A, do B|please do A[split : , do] B  |
|please do A moreover do B|please do A [split : moreover do] B|
|please do A: please do B|please do A[split : :] please do B|
|please do A, please do C|please do A[split : ,] please do C|
|do A; do B|do A[split : ;] do B         |
|please do A, then do B and eventually do C|please do A[split : , then] do B [split : and eventually] do C|
|do A, do B, finally do C|do A, do B[split : , finally] do C|
|Please do A. Please do B|Please do A[split : .] Please do B|
|first of all, please do A, then do B|[split : first of all,] please do A[split : , then do] B|
|firstly, do A and then do B|[split : firstly,] do A [split : and then] do B|
|first off, do A, then do B|[split : first off,] do A[split : , then] do B|
|first, do A then do B|[split : first,] do A [split : then] do B|
|first do A, then do B|[split : first] do A[split : , then] do B|
|foremost do A, then do B|[split : foremost] do A[split : , then] do B|
|please do A, next do B|please do A[split : , next] do B|
|please do A next do B|please do A [split : next] do B|

### <b><span style='color:#6A5ACD;text-align:center'>❯❯ </span>NER Tagging</b> 

I've created a simple **NER** tagging class `ner_annotator` which we can use manually tag each row in the dataframe, it works by iteratively going through all rows in a dataframe, looping on only rows which have not been annotated yet (`active_df`) and tagging parts of the sentence using the format <kbd>words</kbd>`-`<kbd>tag</kbd>, which is used in the **[massive 1.1](https://huggingface.co/datasets/AmazonScience/massive)** dataset on huggingface

```python
#from IPython.display import clear_output in jupyter
import pandas as pd
import numpy as np
import re
import warnings;warnings.filterwarnings('ignore')

# NER Annotation Class

class ner_annotator:
    
    def __init__(self,df:pd.DataFrame):
        self.df = df
        self.word2tag = {}
        self.LABEL_PATTERN = r"\[(.*?)\]"
        self.deactive_df = None
        self.active_df = None
        
        self.__initialise()
        
    def __initialise(self):
        
        '''
        
        [1] ANNOTATION COLUMN RELATED OPERATIONS
        
        '''
        
        # if annotaion column is all empty
        
        if('annotated' in self.df.columns):
            
            if(self.df['annotated'].isna().sum() == self.df.shape[0]):
                self.df['annotated'] = None
                
            # if annotation column is not empty
                
            elif(self.df['annotated'].isna().sum() != self.df.shape[0]):
                
                # Store Tags
                for idx,row_data in self.df.iterrows():
                    if(type(row_data['annotated']) == str):
                        matches = re.findall(self.LABEL_PATTERN, row_data['annotated'] )
                        for match in matches:
                            tag, phrase = match.split(" : ")
                            self.word2tag[phrase] = tag
                            
        # if annotation column is not present
                            
        else:
            word2tag = {}
            self.df['annotated'] = None    
            
        # active_df -> NaN are present
        # deactive_df -> already has annotations
            
        self.active_df = self.df[self.df['annotated'].isna()]
        self.deactive_df = self.df[~self.df['annotated'].isna()]
        
    '''
    
    REVIEW ANNOTATIONS
    
    '''
    # nullify rows which are not NaN, but don't have 
        
    def review_annotations(self):
        idx = list(self.deactive_df[~self.deactive_df["annotated"].str.contains(self.LABEL_PATTERN)]['annotated'].index)
        annot = list(self.deactive_df[~self.deactive_df["annotated"].str.contains(self.LABEL_PATTERN)]['annotated'].values)
        
        for i,j in zip(idx,annot):
            print(i,j)
            
    # drop annotations (from deactive_df)
            
    def drop_annotations(self,idx:list):
        remove_df = self.deactive_df.iloc[idx]
        remove_df['annotated'] = None
        self.active_df = pd.concat([self.active_df,remove_df])
        self.deactive_df = self.deactive_df.drop(list(idx),axis=0)
        self.deactive_df.sort_index()
        print('dopped annotations saving >> annot.csv')
        pd.to_csv('annot.csv',pd.concat([self.active_df,self.deactive_df]))
        
    '''
    
    ANNOTATE ON ACTIVE ONLY
    
    '''
        
    def ner_annotate(self):
        
        for idx,row_data in self.active_df.iterrows():
            
            q = row_data['question'] # question
            t = q                    # annotated question holder
            
            annotate_row = True
            while annotate_row is True:
                
                print('Current Annotations:')
                print(t,'\n')
                
                # user input
                user = input('tag (word-tag) format >> ')
                
                # [1] end of annotation (go to next row)
                
                if(user in ['quit','q']):
                    
                    annotate_row = False
                    row_data['annotated'] = t
                    
                    # Store Tags
                    matches = re.findall(self.LABEL_PATTERN, t)
                    for match in matches:
                        tag, phrase = match.split(" : ")
                        self.word2tag[phrase] = tag
                        
                        # clean up output
                        #               clear_output(wait=True)
                        
                        # [2] stop annotation loop
                        
                elif(user in 'stop'):
                    
                    ldf = pd.concat([self.deactive_df,self.active_df],axis=0)
                    ldf.to_csv('annot.csv',index=False)
                    return 
                
                # [3] Reset current Row Tags
                
                elif(user in ['reset','r']):
                    
                    t=q 
                    print(t,'\n')
                    #           clear_output(wait=True)
                    user = input('tag (word-tag) format >> ')
                    
                # [4] Show current word2tag mapping dictionary
                    
                elif(user == 'show'):
                    print(self.word2tag)

                # [5] Automatically set tags from existing tags in word2tag 
                    
                elif(user == 'dict'):
                    
                    # use dictionary to automatically set tags
                    for word,tag in self.word2tag.items():
                        if(word in t):
                            express = f'[{tag} : {word}]' 
                            t = t.replace(word,express)            
                            
                            # Tags Specified

                # [6] We are actually annotating document
                            
                elif('-' in user):
                    
                    # parse input
                    word,tag = user.split('-')
                    
                    if(word == ''):
                        word = input('please add word >> ')
                    if(tag == ''):
                        tag = input('please add tag >> ')
                        
                    if(word in t):
                        express = f'[{tag} : {word}]' 
                        t = t.replace(word,express)
                    else:
                        print('not found in sentence')
                        
                else:
                    print('please use (word-tag format)')
                    
        # finished annotation
        ldf = pd.concat([self.deactive_df,self.active_df],axis=0)
        ldf.to_csv('annot.csv',index=False)
```

We can annotate the dataset using the folowing commands:

```python

# read dataset
df_annot = pd.read_csv('annot.csv')   # read dataframe

temp = ner_annotator(df_annot)  # instantiate annotation class
temp.ner_annotate()             # start annotating
```

## :material-package-variant-closed-check: <b>Parsing annotations</b>

Now that we have an annotated dataset, we will need a parser which interprets the resulting annotations, and creates tags for words in a document. `Parser` will be called using the special class `__call__`, when we will iterate over our documents. 

```python

from typing import List
import regex as re
from nltk.tokenize import word_tokenize
import numpy as np

'''

PARSER FOR THE DATASET NER TAG FORMAT

'''

class Parser:
    
    # RE patterns for tag extraction
    LABEL_PATTERN = r"\[(.*?)\]"
    PUNCTUATION_PATTERN = r"([.,\/#!$%\^&\*;:{}=\-_`~()'\"’¿])"
    
    # initialise, first word/id tag is O (outside)
    def __init__(self):
        self.tag_to_id = {
            "O": 0
        }
        self.id_to_tag = {
            0: "O"
        }
        
    ''' CREATE TAGS '''
        
    # input : sentence, tagged sentence
        
    def __call__(self, sentence: str, annotated: str) -> List[str]:
        
        ''' Create Dictionary of Identified Tags'''
        
        # 1. set label B or I    
        
        matches = re.findall(self.LABEL_PATTERN, annotated)
        word_to_tag = {}
        for match in matches:
            tag, phrase = match.split(" : ")
            words = phrase.split(" ") 
            word_to_tag[words[0]] = f"B-{tag.upper()}"
            for w in words[1:]:
                word_to_tag[w] = f"I-{tag.upper()}"
                
        ''' Tokenise Sentence & add tags to not tagged words (O)'''
                
        # 2. add token tag to main tag dictionary

        tags = []
        sentence = re.sub(self.PUNCTUATION_PATTERN, r" \1 ", sentence)
        for w in sentence.split():
            if w not in word_to_tag:
                tags.append("O")
            else:
                tags.append(word_to_tag[w])
                self.__add_tag(word_to_tag[w])
                
        return tags
    
    ''' TAG CONVERSION '''
    
    # to word2id (tag_to_id)
    # to id2word (id_to_tag)

    def __add_tag(self, tag: str):
        if tag in self.tag_to_id:
            return
        id_ = len(self.tag_to_id)
        self.tag_to_id[tag] = id_
        self.id_to_tag[id_] = tag
        
        ''' Get Tag Number ID '''
        # or just number id for token
        
    def get_id(self, tag: str):
        return self.tag_to_id[tag]
    
    ''' Get Tag Token from Number ID'''
    # given id get its token
    
    def get_label(self, id_: int):
        return self.get_tag_label(id_)
```

## :octicons-file-code-16: <b>NER Models</b>

### <b><span style='color:#6A5ACD;text-align:center'>❯❯ </span>Training NER Model</b> 

We have an annotation parser `Parser`, now let's prepare the training loop. We read our dataset `df` and iterate over all rows in the dataset, tokenising the document & storing it in `lst_data` and parse the annotated `annotated` column, creating `BIO` tags for words in the document `text`, which is stored in `lst_tags`

The output of preprocessed data `lst_data` will be a list of tokens in the whole corpus, and `lst_tags` will be its corresponding `BIO` tag.

```python
# import relevant libraries
import pandas as pd    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# import our NER annotated documents
parser = Parser()
df = pd.read_csv('annot.csv')   # read dataframe

# parse our NER tag data & tokenise our text
lst_data = []; lst_tags = []
for ii,row in df.iterrows():
    lst_data.extend(word_tokenize(row['question']))
    lst_tags.extend(parser(row["question"], row["annotated"]))
```

```
['Please', 'do', 'A', 'and', 'then', 'do', 'B', 'using', 'C', 'Do'] # lst_data
['O', 'O', 'O', 'B-SPLIT', 'I-SPLIT', 'O', 'O', 'O', 'O', 'O']  # lst_tags
```

The tokens are then converted into `BoW` vectors using `CountVectorizer`, which is then fed into the model `LogisticRegression`. `accuracy` will be used as an evaluation metric in order for us to compare the models. We'll also look at the `confusion matrix` to see which classes were actually miss predicted.

```python
# define encoder
encoder = CountVectorizer()

# fit the encoder on our corpus
X = encoder.fit_transform(lst_data)
y = np.array(lst_tags)

# try our different models
#model = LogisticRegression()
model = RandomForestClassifier()

# train model
model.fit(X,y)
y_pred = model.predict(X)
print(f'accuracy: {round(accuracy_score(y_pred,y),3)}')
```

Our trained model performance:

- `LogisticRegression` accuracy: 0.77
- `RandomForestClassifier` accuracy: 0.808

`accuracy` doesn't really tell us the whole picture, since a model might be simply confusing `B` and `I` tags; for our problem is not that critical since we don't really need to distinguish them, let's review the **classification_report**

```
              precision    recall  f1-score   support

     B-SPLIT       0.77      0.33      0.47        30
     I-SPLIT       0.88      0.60      0.71        25
           O       0.80      0.99      0.89       106

    accuracy                           0.81       161
   macro avg       0.82      0.64      0.69       161
weighted avg       0.81      0.81      0.78       161
```

and **confusion matrix**

  
```
[[ 10   2  18]  # class B-SPLIT
 [  2  15   8]  # class I-SPLIT
 [  1   0 105]] # class O
```

Looks like we have a large ammount of misspredictions for tag `B-SPLIT`, with the recall being quite low (ie quite a few `B-SPLIT` are misspredicted), which is to be expected. classification accuracy is much higher when we have tags with multiple splitting words, these tend to be cases for **transitional adverbs**, so its quite nice that the model can identify such splitting locations quite well.


### <b><span style='color:#6A5ACD;text-align:center'>❯❯ </span>Testing NER model</b> 

Different models can show different performance upon actual usage, so let's compare how both of them perform

Our test example:

> 'please do this and then do that, once done do that' <br>

Intuitively, we would want to split the sentence into three sections, let's see how our models perform:

```python
# test our model
def ner_model(inputs):

    tokens = word_tokenize(inputs)
    y_pred_test = model.predict(encoder.transform(tokens))    
    return pd.DataFrame({'word':tokens,'tag':y_pred_test})

# test model on some samples
to_tag = 'please do this and then do that, once done do that'

# predict tags 
pred_tags = ner_model(to_tag)
print(pred_tags)
```

`LogisticRegression` prediction tags:

```
      word      tag
0   please        O
1       do        O
2     this        O
3      and  B-SPLIT
4     then  I-SPLIT
5       do        O
6     that        O
7        ,        O
8     once        O
9     done        O
10      do        O
11    that        O
```

`RandomForest` prediction tags:

```
      word      tag
0   please        O
1       do        O
2     this        O
3      and  B-SPLIT
4     then  I-SPLIT
5       do        O
6     that        O
7        ,        O
8     once  I-SPLIT
9     done  I-SPLIT
10      do        O
11    that        O
```

We can see that **model complexity** indeed does play a role, as the linear model `LogisticRegression` wasn't able to identify the the second splitting tokens `once` `done`. `RandomForest` works better as a sentence splitting `ner` tagging model. Having identified the indicies of `ner` tags, it is quite straightforward to split the document into `word` & divide it based on the model prediction tag condition 

## :fontawesome-solid-person-walking-dashed-line-arrow-right: <b>Conclusion</b>

In this post we created a `NER` tagger, which identifies key words that we defined as **sentence splitting words**. Identificaiton of such words allows us to split a long paragraph into parts & analyse or conduct further downstream tasks upon performing splits. We showed that **sentence splitting** can be defined as more than just splitting by punctuation like `.`, and since people often connect sentences using **transitional adverbs** and alike, `NER` tagging comes in handy to identify such "weaker" sentence splitters.

We defined a class that allows us to create our own annotations (`ner_annotator`), a annotaion parser that allows us to interpret the annotations made by `ner_annotator`. We created a dataset that contained `text` and the relevant `ner` annotations `annotations`, which we used to train our `ner` tag model.

As a result of our model testing, we saw that `RandomForest` performs quite well in comparison to simpler linear models. Overall, the created model can be utilised for weak sentence splitting

***

**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**
