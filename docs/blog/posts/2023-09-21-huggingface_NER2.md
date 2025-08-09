---
date: 2023-08-19
title: Named Entity Recognition with Huggingface Trainer
authors: [andrey]
categories:
    - nlp
tags:
    - huggingface
    - ner
comments: false
---

# **Named Entity Recognition with Huggingface Trainer**

In a **[previous post](https://shtrausslearning.github.io/posts/huggingface_NER/)** we looked at how we can utilise Huggingface together with PyTorch in order to create a NER tagging classifier. We did this by loading a preset encoder model & defined our own tail end model for our NER classification task. This required us to utilise Torch`, ie create more lower end code, which isn't the most beginner friendly, especially if you don't know Torch. In this post, we'll look at utilising only Huggingface, which simplifies the **training** & **inference** steps quite a lot. We'll be using the **trainer** & **pipeline** methods of the Huggingface library and will use a dataset used in **[mllibs](https://pypi.org/project/mllibs/)**, which includes tags for different words that can be identified as keywords to finding data source tokens, plot parameter tokens and function input parameter tokens.

<!-- more -->

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/19dwCH-iTdnYgUJM2AVdTsvBirEMJLjXA?usp=sharing)

## Background

### Huggingface Trainer

A Huggingface `Trainer` is a **high-level API** provided by the Huggingface Transformers library that makes it easier to train, fine-tune, and evaluate various Natural Language Processing (NLP) models. It provides a simple and consistent interface for training and evaluating models using various techniques such as text classification, named entity recognition, question answering, and more. The Trainer API abstracts away many of the low-level details of model training and evaluation, like we did in the **[previous post](https://shtrausslearning.github.io/posts/huggingface_NER/)**, allowing users to focus on their specific NLP tasks.

### Why do we need NER in NLP?

> Named entity recognition (NER) is important because it helps to identify and extract important information from unstructured text data. This can be useful in a variety of applications such as information retrieval, sentiment analysis, and text summarization. NER can also help to improve the accuracy of machine learning models by providing additional features for classification tasks. Additionally, NER is important for natural language processing `NLP` tasks such as machine translation, speech recognition, and question answering systems.

**NER** can be quite a helpful tool in a variety of **NLP** applications. 

### Our NER application

Our **NER** application in this example:

We'll be using **token classification** in order to identify tokens that can be identified as function `input parameters`, `data sources` and `plot parameters`. Identifying them helps us extract information from input text and allow us to store & use relevant data that is defined in a `string`

Let's look at an example:

**Input Text:**

: visualise column kdeplot for hf hue author fill True alpha 0.1 mew 1 mec 1 s 10 stheme viridis bw 10

**NER Tagged:**

: visualise column kdeplot `[source : for]` hf `[param : hue]` author `[pp : fill]` True `[pp : alpha]` 0.1 `[pp : mew]` 1 `[pp : mec]` 1 `[pp : s]` 10 `[pp : stheme]` viridis `[pp : bw]` 10

## The Dataset

We can define a tag parser that supports an input format: `[tag : name]` (the **[massive dataset](https://huggingface.co/datasets/AmazonScience/massive) format)**. First, lets initialise the parser; `parser`, which reads the above format every time it is called. To parse all data in our dataset, we loop through all data and store relevant mapping dictionaries, returning the **tokenised text** & **tokenised tags**. 

Let's look at the input data format:

```python
# for demonstration only
ldf = pd.read_csv('ner_mp.csv')
text,annot = list(ldf['text'].values),list(ldf['annot'].values)

text[:4]
# ['visualise column kdeplot for dbscan_labels hue outlier_dbscan',
# 'visualise column kdeplot for hf hue author fill True alpha 0.1 mew 1 mec 1 s 10 stheme viridis bw 10',
# 'create relplot using hf x pressure y mass_flux col geometry hue author',
# 'pca dimensionality reduction using data mpg subset numerical columns only']

annot[:4]
# ['visualise column kdeplot [source : for] dbscan_labels [param : hue] outlier_dbscan',
# 'visualise column kdeplot [source : for] hf [param : hue] author [pp : fill] True [pp : alpha] 0.1 [pp : mew] 1 [pp : mec] 1 [pp : s] 10 [pp : stheme] viridis [pp : bw] 10',
# 'create relplot using hf [param : x] pressure [param : y] mass_flux [param : col] geometry [param : hue] author',
# 'pca dimensionality reduction [source : using data] mpg [source : subset] numerical columns only']
```

### Annotation Parser

The above format can be parsed using `Parser`, we first need to initialise it

```python
from typing import List
import regex as re

# ner tag parser
class Parser:

    LABEL_PATTERN = r"\[(.*?)\]"
    PUNCTUATION_PATTERN = r"([,\/#!$%\^&\*;:{}=\-`~()'\"’¿])"

    def tokenise(self,text:str):
        sentence = re.sub(self.PUNCTUATION_PATTERN, r" \1 ", text)
        tokens = []
        for w in sentence.split():
            tokens.append(w)
            
        return tokens
    
    def __init__(self):
        self.tag_to_id = {
            "O": 0
        }
        self.id_to_tag = {
            0: "O"
        }
    
    def __call__(self, sentence: str, annotated: str) -> List[str]:
        matches = re.findall(self.LABEL_PATTERN, annotated)
        word_to_tag = {}
        for match in matches:
            tag, phrase = match.split(" : ")
            words = phrase.split(" ") 
            word_to_tag[words[0]] = f"B-{tag.upper()}"
            for w in words[1:]:
                word_to_tag[w] = f"I-{tag.upper()}"

        tags = []; txt = []
        sentence = re.sub(self.PUNCTUATION_PATTERN, r" \1 ", sentence)
        for w in sentence.split():

            txt.append(w)
            if w not in word_to_tag:
                tags.append("O")
            else:
                tags.append(word_to_tag[w])   # return tag for current [text,annot]
                self.__add_tag(word_to_tag[w]) # add [tag] to tag_to_id/id_to_tag
    
        # convert tags to numeric representation (if needed)

        ttags = []
        for tag in tags:
            ttags.append(self.tag_to_id[tag])
        tags = ttags
            
        return tags

    def __add_tag(self, tag: str):
        if tag in self.tag_to_id:
            return
        id_ = len(self.tag_to_id)
        self.tag_to_id[tag] = id_
        self.id_to_tag[id_] = tag
        
    def get_id(self, tag: str):
        return self.tag_to_id[tag]
    
# initialise parser
parser = Parser()
```

### Parse Dataset

To use the parser on a dataset, we loop through all our rows, calling the parser, which fills out `tag_to_id`, `id_to_tag` and returns the tokenised `tags` when it is called

```python

# loop through dataset
lst_tokens = []; lst_annot = []
for i,j in zip(text,annot):
    tags = parser(i,j)                      # parse input ner format 
    lst_tokens.append(parser.tokenise(i))   # tokenise the input text
    lst_annot.append(tags)             
```

We'll be using **Huggingface** to train our model, so we need our two lists `lst_tokens`,`lst_annot` to be converted into a huggingface `dataset`. To do this we create a `dataframe` and then use `from_pandas` method to convert the `dataframe` to a `dataset`. Each row in the `dataset` contains a list, which by default will not be registered correctly as the default datatype, so we need to define feature types (`features` argument) when calling `from_pandas`. A list is equivalent to a `Sequence`, whilst `ner_tags` are our labels, for which we need to define `ClassLabel`.

```python
from datasets import Dataset,DatasetDict,Features,Value,ClassLabel,Sequence

# create dataframe from token/tag pairs
df_raw_dataset = pd.DataFrame({'tokens':lst_tokens,
                               'ner_tags':lst_annot})

# custom data type
class_names = list(parser.tag_to_id.keys())
ft = Features({'tokens': Sequence(Value("string")),
               'ner_tags':Sequence(ClassLabel(names=class_names))})


# dataset = Dataset.from_pandas(df_raw_dataset,features=ft) # convert dataframe to dataset
dataset = Dataset.from_pandas(df_raw_dataset,
                              features=ft) # convert dataframe to dataset

# visualise a sample from the dataset
raw_datasets = DatasetDict() # create dataset diction
raw_datasets['train'] = dataset # register dataset training data
raw_datasets['train'][0]
```

```
{'tokens': ['visualise',
  'column',
  'kdeplot',
  'for',
  'dbscan_labels',
  'hue',
  'outlier_dbscan'],
 'ner_tags': [0, 0, 0, 1, 0, 2, 0]}
```

Let's look at what data we have as a result of parsing input data `ldf` with `parser` 

```python
# visualise tags
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
# visualise column kdeplot for      dbscan_labels hue     outlier_dbscan 
# O         O      O       B-SOURCE O             B-PARAM O              
```

## Transformer Tokeniser

The above format of **word/tag** pair works fine if we use the pair for classification as it is. Our approach involves using a **transformer encoder** model (`bert-base-cased`), which has it own tokenisation approach, which means we need to create token tags for each **subtoken** that the tokeniser creates, this means we need to make a little adjustment to our input data

First let's load our tokeniser using `form_pretrained`

```python
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

To align the tags to correpond to **subword tokens** we can use a helper function `align_labels_with_tokens`; the function inputs the **original labels** (those with tags for words) and **word_ids**, which the `tokeniser` outputs representing the **token-word** association. As the tokeniser can split a word into parts, we can use this information to define the **subtoken** tags

```python
# helper function; convert word token labels to subword token labels
# which will be fed into the model w/ input_ids

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
```

### Tokeniser Sample

And let's look at an example of how the `tokeniser` actually splits the input text data, as well as the **token-word** association list; this information allows us to define subtoken tags based on their word tag from `word_ids` data

```python
# the model tokeniser format
inputs = tokenizer(raw_datasets["train"][10]["tokens"], is_split_into_words=True)

print('bert tokens')
print(inputs.tokens())

# the tokeniser also keeps track of which tokens belong to which word
print('\nbert word_ids')
print(inputs.word_ids())

# bert tokens
# ['[CLS]', 'create', 'sea', '##born', 'kernel', 'density', 'plot', 'using', 'h', '##f', 'x', 'x', '_', 'e', '_', 'out', '[SEP]']

# bert word_id
# [None, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 8, 8, 8, None]
```

### Subtoken NER tag adjustment

Having all relevant pieces, let's return to our original dataset `raw_datasets` which we created in **[parse dataset](https://shtrausslearning.github.io/posts/huggingface_NER2/#-parse-dataset)**

```python

# Tokenise input text
def tokenize_and_align_labels(examples):

    # tokenise 
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)                        # subtoken word identifier
        new_labels.append(align_labels_with_tokens(labels, word_ids))  # new ner labels for subtokens

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(tokenize_and_align_labels,
                                      batched=True,
                                      remove_columns=raw_datasets["train"].column_names)

tokenized_datasets['train']
# Dataset({
#    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
#    num_rows: 51
# })
```

### Data Collator Adjustment

The final preprocessing step is the `data_collator`

```python
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# let's check what it does
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
batch["labels"]
#tensor([[-100,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,
#           0,    0,    2,    2,    0,    0,    0,    0,    0,    0, -100, -100,
#        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
#        -100, -100],
#       [-100,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    2,
#           2,    0,    3,    0,    3,    0,    0,    0,    3,    4,    0,    3,
#           4,    0,    3,    0,    3,    4,    4,    0,    0,    0,    3,    4,
#           0, -100]])
```

## Prepare for Training

### Evaluation Metrics

The trainer allows us to add an evaluation function which when added to `compute_metrics` in the trainer return the model prediction, which are the `logits` and `labels`. Using this data we can then write an **evaluation metric function** which returns a dictionary of metric and its value pairs

```python
import numpy as np

def compute_metrics(eval_preds):

    model prediction
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
```

### Define Model

Huggingface allows us to easily adjust the **base model** (`bert-base-cased`) for different tasks by loading the task type from the main library. For NER, we need to load `AutoModelForTokenClassification`, for which we then need to define two mapping dictionaries `id2label` & `label2id`

```python
# define subtoken-tag mapping dictionary
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
                                                        id2label=id2label,
                                                        label2id=label2id)
```

### Train Model

Time to train our model, we'll train the model for 40 epochs, without an evaluation strategy (validation dataset), using a learning rate in our optimiser is set to 2e-5, which by default is the AdamW optimiser.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner2",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=40,
    weight_decay=0.01)
```

```python
from transformers import Trainer

trainer = Trainer(
    model=model,                       # define the model
    args=args,                         # define the trainer parameters
    train_dataset=tokenized_datasets["train"],  # define the adjusted ner tag dataset
    data_collator=data_collator,      # define the data collator
    compute_metrics=compute_metrics,  # define the evaluation metrics function
    tokenizer=tokenizer,              # define the tokeniser
)
trainer.train()

# TrainOutput(global_step=280, training_loss=0.009931504726409912, metrics={'train_runtime': 24.8186, 'train_samples_per_second': 82.197, 'train_steps_per_second': 11.282, 'total_flos': 27046414867968.0, 'train_loss': 0.009931504726409912, 'epoch': 40.0})
```

If we use `evaluation_strategy="steps"`, `eval_steps=50` & `eval_dataset` (just setting to train data) we get the following **metrics** so the model has memorised the inputs in our training set 

```
Step 	Training Loss 	Validation Loss 	Precision 	Recall 	F1 	Accuracy
50 	No log 	0.030486 	0.994949 	0.989950 	0.992443 	0.995763
100 	No log 	0.006724 	0.990000 	0.994975 	0.992481 	0.995763
150 	No log 	0.007272 	0.994949 	0.989950 	0.992443 	0.995763
200 	No log 	0.007903 	0.994949 	0.989950 	0.992443 	0.995763
250 	No log 	0.006493 	0.990000 	0.994975 	0.992481 	0.995763
```

### Save Huggingface Trainer

We can save our `trainer` using `.save_model` method, which saves all relevant needed to utilise the `pipeline` method.

```python
trainer.save_model("bert-finetuned-ner2")
```

### Load a Huggingface Pipeline

Having saved our `trainer`, we simply call the relevant model checkpoint which we used in `.save_model` and set the `task` argument, which for our **NER** problem is `token-classification`, we also need to set an **aggregation_strategy**:

The `aggregation_strategy` is a parameter in the Hugging Face Trainer API that specifies how to aggregate the predictions of multiple batches during evaluation. This is particularly useful when evaluating large datasets that cannot fit into memory all at once. The aggregation_strategy parameter can be set to one of several options, including "mean", "max", "min", and "median", which determine how the predictions should be combined. 

```python
from transformers import pipeline
# Replace this with your own checkpoint
model_checkpoint = "bert-finetuned-ner2"
token_classifier = pipeline("token-classification",
                            model=model_checkpoint)
```

Let's try finding the relevant tokens in the following request

```python
token_classifier("create a seaborn scatterplot using A, set x=B y: C (mew: 10, mec:20)")
```
```
[{'entity_group': 'SOURCE',
  'score': 0.9986413,
  'word': 'using',
  'start': 29,
  'end': 34},
 {'entity_group': 'PARAM',
  'score': 0.9031896,
  'word': 'x',
  'start': 42,
  'end': 43},
 {'entity_group': 'PARAM',
  'score': 0.99833447,
  'word': 'y',
  'start': 46,
  'end': 47},
 {'entity_group': 'PP',
  'score': 0.74444526,
  'word': 'mew',
  'start': 52,
  'end': 55},
 {'entity_group': 'SOURCE',
  'score': 0.4806772,
  'word': ',',
  'start': 59,
  'end': 60}]
```

All in all, we can see that the model predicts the relevant tags quite well, there are some issues with `mew` and `mec` tags probably because of the aggregation strategy doesn't work well with the subword tokens that the tokeniser creates for such a short word as it is likely present in other parts but has a different subword token tags, a problem for another day

## Summary

In this post we looked at how we can use huggingface to do **named entity recognition**. Let's look at the step that we took:

- [x] NER Task
  + [x] load **[massive format ner annotations](https://shtrausslearning.github.io/posts/huggingface_NER2/#the-dataset)**
  + [x] parse the annotations & create word/tag pairs `ldf` **[parse dataset](https://shtrausslearning.github.io/posts/huggingface_NER2/#-parse-dataset)**
  + [x] convert parsed dataset to `dataframe` then `dataset` format `raw_dataset` **[parse dataset](https://shtrausslearning.github.io/posts/huggingface_NER2/#-parse-dataset)**
  + [x] tokenise the input `text` from input dataset & correct `raw_dataset` ner tags to coincide with subtokens **[subtoken ner tag adjustment](https://shtrausslearning.github.io/posts/huggingface_NER2/#-subtoken-ner-tag-adjustment)**
  + [x] defined a **[model](https://shtrausslearning.github.io/posts/huggingface_NER2/#-define-model)** & data collator for **token classification**
  + [x] trained our token classifier which achieved an **[accuracy of 0.99+](https://shtrausslearning.github.io/posts/huggingface_NER2/#-train-model)** on the training dataset
  + [x] defined a **[pipeline](https://shtrausslearning.github.io/posts/huggingface_NER2/#-load-a-huggingface-pipeline)** and tested our model on some input data, the pipeline returns `entity_group` & relevant `score` for each tag it has identified
     
So this wraps up our post. Such an approach definitely helps simplify things, a custom training loop like in a **[previous post](https://shtrausslearning.github.io/posts/huggingface_NER/)** is not needed because of the `trainer`, which is very convenient because we can save the trainer upon fine tuning & easily do inference on new data using the **pipeline** method. Having used the **[massive dataset](https://huggingface.co/datasets/AmazonScience/massive) format)** & storing it in a `csv`, created some additional preprocessing steps, in addition to the adjustments we needed to make to each word token, so we would have data for **subword tokens** added some complexity. We also didn't experiment with the `aggregation_strategy` in **pipeline** since our results were more or less good, however its also something to consider when creating a **NER** tagger, you can find the different approaches **[here](https://huggingface.co/transformers/v4.7.0/_modules/transformers/pipelines/token_classification.html)**

***

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**

