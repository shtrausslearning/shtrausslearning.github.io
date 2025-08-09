---
date: 2023-10-21
title: Gene Classification
authors: [andrey]
categories:
     - science
tags:
     - classification
     - bioinformatics
     - machine learning
comments: true
---

# **Gene Classification**

In this notebook, we look at how to work with biological sequence data, by venturing into a **classification problem**, in which we will be classifying between seven different **genes groups** common to three different species (human,chimpanzee & dog)

<div class="grid cards" markdown>

  - :simple-google:{ .lg .middle }&nbsp; <b>[Open Colab Notebook](https://colab.research.google.com/drive/1TU9_w1eWTsnObTKPKWNcR265EVUHmNWW?usp=sharing)</b>
  - :simple-kaggle:{ .lg .middle }&nbsp; <b>[Kaggle Dataset](https://www.kaggle.com/datasets/nageshsingh/dna-sequence-dataset)</b>

</div>

<!-- more -->

## Background

### Gene Classification Problem

**Gene classification** using machine learning is the process of using algorithms and statistical models to analyze large datasets of genetic information and predict the function or characteristics of different genes. Machine learning techniques can be used to identify patterns in gene expression data, classify genes into different functional categories, or predict the likelihood of a gene being associated with a particular disease or phenotype. This approach can help researchers to better understand the complex relationships between genes, their functions, and their interactions with other biological systems.

### Genes

The dataset that we'll be using contains the following **gene classes** for three different species, **human**, **chimpanzee** & **dog**, it is added more as a reference for those interested in what each class represents in our data

**Gene families** are groups of related genes that share a common ancestor. Members of gene families may be paralogs or orthologs. Gene paralogs are genes with similar sequences from within the same species while gene orthologs are genes with similar sequences in different species.

!!! abstract "<b>Gene Families</b>"

    "G Protein-Coupled Receptors"

     The **G protein-coupled receptors** (GPCRs) gene family is a large and diverse group of genes that encode proteins involved in cellular signaling pathways. These receptors are located on the surface of cells and are activated by a wide range of ligands, including hormones, neurotransmitters, and environmental stimuli. GPCRs play critical roles in regulating many physiological processes, such as sensory perception, hormone secretion, and immune response. Dysregulation of GPCR signaling has been implicated in a variety of diseases, including cancer, diabetes, and cardiovascular disorders.

    "Tyrosine Kinase"

     The **tyrosine kinase** gene family is a group of genes that encode proteins involved in cellular signaling pathways. These proteins are enzymes that add phosphate groups to specific tyrosine residues on target proteins, thereby regulating their activity. Tyrosine kinases play critical roles in many physiological processes, such as cell growth, differentiation, and survival. Dysregulation of tyrosine kinase signaling has been implicated in a variety of diseases, including cancer, autoimmune disorders, and developmental disorders. Examples of tyrosine kinase genes include EGFR, HER2, and BCR-ABL.

    "Tyrosine Phosphatase"

     The **tyrosine phosphatase** gene family is a group of genes that encode proteins involved in cellular signaling pathways. These proteins are enzymes that remove phosphate groups from specific tyrosine residues on target proteins, thereby regulating their activity. Tyrosine phosphatases play critical roles in many physiological processes, such as cell growth, differentiation, and survival. Dysregulation of tyrosine phosphatase signaling has also been implicated in a variety of diseases, including cancer, autoimmune disorders, and developmental disorders. Examples of tyrosine phosphatase genes include PTPN1, PTPN6, and PTPN11.

    "Synthetase"

     The **synthetase** gene family is a group of genes that encode for enzymes called aminoacyl-tRNA synthetases. These enzymes are responsible for attaching specific amino acids to their corresponding tRNA molecules during protein synthesis. There are 20 different aminoacyl-tRNA synthetases, one for each amino acid, and each enzyme recognizes and binds to its specific amino acid and tRNA molecule. The synthetase gene family is highly conserved across all living organisms and mutations in these genes can lead to various genetic disorders.

    "Synthase"

     The **synthase** gene family is a group of genes that encode enzymes responsible for synthesizing various molecules within cells. These enzymes are involved in a wide range of biological processes, including the synthesis of lipids, nucleotides, and amino acids. Different members of the synthase gene family may be involved in different aspects of these processes, and mutations in these genes can lead to a variety of diseases and disorders. Examples of synthase genes include fatty acid synthase, which is involved in the synthesis of fatty acids, and adenylyl cyclase, which synthesizes the signaling molecule cyclic AMP.

    "Ion Channel"

     The **ion channel** gene family is a group of genes that encode proteins responsible for the transport of ions across cell membranes. These proteins are integral membrane proteins that form pores or channels in the lipid bilayer of the cell membrane, allowing the selective movement of ions such as sodium, potassium, calcium, and chloride. Ion channels are critical for many physiological processes, including muscle contraction, nerve signaling, and hormone secretion. Dysregulation of ion channel activity has been implicated in a variety of diseases, including epilepsy, cardiac arrhythmias, and cystic fibrosis. Examples of ion channel genes include SCN1A, KCNQ1, and CFTR.

    "Transcription Factors"

     The **transcription factor** gene family is a group of genes that encode proteins responsible for regulating the expression of other genes. These proteins bind to DNA and control the rate at which genes are transcribed into mRNA, which is then translated into proteins. Transcription factors are involved in a wide range of biological processes, including development, differentiation, and response to environmental stimuli. Different members of the transcription factor gene family may have different target genes and regulatory mechanisms, allowing for precise control of gene expression. Mutations in these genes can lead to a variety of diseases and disorders, including cancer and developmental disorders. Examples of transcription factor genes include homeobox genes, which regulate embryonic development, and p53, which regulates cell cycle progression and DNA repair

## Dataset

We're loading three DNA datasets, our dataset is in the form of a `sequence` & subsequent **gene family** label, `class`

```python
import pandas as pd

human_dna = pd.read_table('../input/dna-sequence-dataset/human.txt')
chimp_dna = pd.read_table('../input/dna-sequence-dataset/chimpanzee.txt')
dog_dna = pd.read_table('../input/dna-sequence-dataset/dog.txt')
```

## :octicons-star-16: DNA Encoding

Biological sequences come in the format:

> GTGCCCAGGTTCAGTGAGTGACACAGGCAG

This mimics a standard **NLP** based problem, in which we need to convert text into numerical representation before we can feed this data into our models

There are 3 general approaches to encode biological sequence data:

1. Ordinal encoding DNA Sequence
2. One-Hot encoding DNA Sequence
3. DNA sequence as a “language”, known as `k-mer` counting

So let us implement each of them and see which gives us the perfect input features.

### Encoding Sequences

#### (1) Ordinal Encoding


```python
# encode list of strings
def ordinal_encoder(my_array:list[str]):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25 # A
    float_encoded[float_encoded == 1] = 0.50 # C
    float_encoded[float_encoded == 2] = 0.75 # G
    float_encoded[float_encoded == 3] = 1.00 # T
    float_encoded[float_encoded == 4] = 0.00 # anything else
    return float_encoded

# let’s try it out with a simple sequence
seq_test = 'TTCAGCCAGTG'
ordinal_encoder(lst_string(seq_test))
```

```
array([1.  , 1.  , 0.5 , 0.25, 0.75, 0.5 , 0.5 , 0.25, 0.75, 1.  , 0.75])
```

One slight issue with such an approach is that if we have biological sequences of different length, we won't be able to concatenate them together without **truncation** or **padding**

#### (2) One-Hot Encoding

Another approach is to use one-hot encoding to represent the DNA sequence. For example, “ATGC” would become [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0] vectors & these one-hot encoded vectors are then concatenated into 2-dimensional arrays. Ie. each vector represents the presence or absence of a particular nucleotides in the sequence, the total length then becomes the total number of nucleotides x nucleotide absence/present vector.

```python
from sklearn.preprocessing import OneHotEncoder

def ohe(seq_string:str):
    seq_string = lst_string(seq_string)
    int_encoded = label_encoder.transform(seq_string)
    onehot_encoder = OneHotEncoder(sparse_output=True,dtype=int)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded[:,None])
    return onehot_encoded.toarray()

# let’s try it out with a simple sequence
seq_test = 'GAATTCTCGAA'
ohe(seq_test)
```
```
array([[0, 0, 1, 0],
       [1, 0, 0, 0],
       [1, 0, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [1, 0, 0, 0],
       [1, 0, 0, 0]])
```

The size of these matrices will be directly proptional to their total nucleotide count. If we have sequences of different length, we would need to resort to either **padding** or **truncation** again. If we had used **unitigs** as in **[this notebooks](https://www.kaggle.com/code/shtrausslearning/transcription-factor-binding-location-prediction)**, this problem would not exist

#### (3) K-MER Counting

DNA and protein sequences can be seen as the language of life. The language encodes instructions as well as functions for the molecules that are found in all life forms. The sequence language resemblance continues with the genome as the book, subsequences (genes and gene families) are sentences and chapters, **k-mers** and **peptides** are words, and nucleotide bases and amino acids are the alphabets. Since the relationship seems so likely, it stands to reason that the natural language processing(NLP) should also implement the natural language of DNA and protein sequences.

The method we use here is manageable and easy. We first take the long biological sequence and break it down into k-mer length overlapping “words”. For example, if we use “words” of length 6 (hexamers), “ATGCATGCA” becomes: ‘ATGCAT’, ‘TGCATG’, ‘GCATGC’, ‘CATGCA’. Hence our example sequence is broken down into 4 hexamer words.

```python
def kmers_count(seq:str, size:int) -> list:
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

# let’s try it out with a simple sequence
mySeq = 'GTGCCCAGGTTCAGTGAGTGACACAGGCAG'
kmers_count(mySeq, size=7)
```

```
['gtgccc',
 'tgccca',
 'gcccag',
 'cccagg',
 'ccaggt',
 'caggtt',
 'aggttc',
 'ggttca',
 'gttcag',
 'ttcagt',
 'tcagtg',
 'cagtga',
 'agtgag',
 'gtgagt',
 'tgagtg',
 'gagtga',
 'agtgac',
 'gtgaca',
 'tgacac',
 'gacaca',
 'acacag',
 'cacagg',
 'acaggc',
 'caggca',
 'aggcag']
```

`kmers_count` returns a list of sequences (`k-mer` words), which then can be joined together into a single `string`. Once we have this split, we can use **NLP** encoding/embedding methods (eg. `CountVectorizer`) to generate numerical representations of these `k-mer` words

```python
joined_sentence = ' '.join(words)
joined_sentence
```

```
'gtgccc tgccca gcccag cccagg ccaggt caggtt aggttc ggttca gttcag ttcagt tcagtg cagtga agtgag gtgagt tgagtg gagtga agtgac gtgaca tgacac gacaca acacag cacagg acaggc caggca aggcag'
```

Having a "corpus" of sequences & their labels, we need to merge them together into a single array, for example if we only have two sequences, **even of different length**:

```python
mySeq1 = 'TCTCACACATGTGCCAATCACTGTCACCC'
mySeq2 = 'GTGCCCAGGTTCAGTGAGTGACACAGGCAG'
sentence1 = ' '.join(kmers_count(mySeq1, size=6))
sentence2 = ' '.join(kmers_count(mySeq2, size=6))
```

Fitting a **Bag of Words** model, we generate a fixed dictionary size of `kmers`, thus all data will have a dimensionality proportional to the dictionary count. Similar to **OHE**, the content will be (1/0), corresponding to either being present in the string or not.

```python
# Creating the Bag of Words model:
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform([sentence1, sentence2]).toarray()
```

```
array([[1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
        1, 0, 1, 1, 0],
       [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,
        0, 1, 0, 0, 1]])
```

#### Choice of Encoding

For **DNA** sequence classification methods, its more logical to utilise the `kmers` approach since `kmers` are commonly used in sequence analysis and genome assembly, as they can provide information about the composition and structure of the sequence. Which is on top of the non uniform sequence length issue addessed above.

## Objective

Our objective is to train a **classification model** that is trained on the human DNA sequence and can predict a gene family based on the DNA sequence of the coding sequence. To test the model, we will use the DNA sequence of humans, dogs, and chimpanzees and compare model accuracies.

## Preprocessing

Having already loaded our dataset & define our problem statement, lets create prepare our dataset for training, we'll simply using the `apply` method & store the list of **kmers** in our dataframe

```python
def kmers_count(seq:str, size:int) -> list:
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

human_dna['kmers'] = human_dna.apply(lambda x: kmers_count(x['sequence']), axis=1)
human_dna = human_dna.drop('sequence', axis=1)

chimp_dna['kmers'] = chimp_dna.apply(lambda x: kmers_count(x['sequence']), axis=1)
chimp_dna = chimp_dna.drop('sequence', axis=1)

dog_dna['kmers'] = dog_dna.apply(lambda x: kmers_count(x['sequence']), axis=1)
dog_dna = dog_dna.drop('sequence', axis=1)
```
```
     class     words
0    4    [atgccc, tgcccc, gcccca, ccccaa, cccaac, ccaac...
1    4    [atgaac, tgaacg, gaacga, aacgaa, acgaaa, cgaaa...
2    3    [atgtgt, tgtgtg, gtgtgg, tgtggc, gtggca, tggca...
3    3    [atgtgt, tgtgtg, gtgtgg, tgtggc, gtggca, tggca...
4    3    [atgcaa, tgcaac, gcaaca, caacag, aacagc, acagc...
```

Let's create a list containg the **kmers** string for each row in the dataset (which can simply be using with `fit` in `CountVectorizer`) like we did in the example & its related label:

```python
human_texts = list(human_dna['kmers'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
#separate labels
y_human = human_dna.iloc[:, 0].values # y_human for human_dna

chimp_texts = list(chimp_dna['kmers'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
#separate labels
y_chim = chimp_dna.iloc[:, 0].values # y_chim for chimp_dna

dog_texts = list(dog_dna['kmers'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
#separate labels
y_dog = dog_dna.iloc[:, 0].values  # y_dog for dog_dna
```


```python
human_texts[0]
```

```
'atgccc tgcccc gcccca ccccaa cccaac ccaact caacta aactaa actaaa ctaaat taaata aaatac aatact atacta tactac actacc ctaccg taccgt accgta ccgtat cgtatg gtatgg tatggc atggcc tggccc ggccca gcccac cccacc ccacca caccat accata ccataa cataat ataatt taatta aattac attacc ttaccc tacccc accccc ccccca ccccat cccata ccatac catact atactc tactcc actcct ctcctt tcctta ccttac cttaca ttacac tacact acacta cactat actatt ctattc tattcc attcct ttcctc tcctca cctcat ctcatc tcatca catcac atcacc tcaccc caccca acccaa cccaac ccaact caacta aactaa actaaa ctaaaa taaaaa aaaaat aaaata aaatat aatatt atatta tattaa attaaa ttaaac taaaca aaacac aacaca acacaa cacaaa acaaac caaact aaacta aactac actacc ctacca taccac accacc ccacct caccta acctac cctacc ctacct tacctc acctcc cctccc ctccct tccctc ccctca cctcac ctcacc tcacca caccaa accaaa ccaaag caaagc aaagcc aagccc agccca gcccat cccata ccataa cataaa ataaaa taaaaa aaaaat aaaata aaataa aataaa ataaaa taaaaa aaaaaa aaaaat aaaatt aaatta aattat attata ttataa tataac ataaca taacaa aacaaa acaaac caaacc aaaccc aaccct accctg ccctga cctgag ctgaga tgagaa gagaac agaacc gaacca aaccaa accaaa ccaaaa caaaat aaaatg aaatga aatgaa atgaac tgaacg gaacga aacgaa acgaaa cgaaaa gaaaat aaaatc aaatct aatctg atctgt tctgtt ctgttc tgttcg gttcgc ttcgct tcgctt cgcttc gcttca cttcat ttcatt tcattc cattca attcat ttcatt tcattg cattgc attgcc ttgccc tgcccc gccccc ccccca ccccac cccaca ccacaa cacaat acaatc caatcc aatcct atccta tcctag'
```

`CountVectorizer` allows us to control groupings of these kmers, lets use `ngram_range` of 4. We'll call `fit` on the human dataset & `transform` all datasets:

```python
from sklearn.feature_extraction.text import CountVectorizer
vectoriser = CountVectorizer(ngram_range=(4,4)) #The n-gram size of 4 is previously determined 

# fit & transform
X = vectoriser.fit_transform(human_texts)
X_chimp = vectoriser.transform(chimp_texts)
X_dog = vectoriser.transform(dog_texts)
```

This will give us the following dataset size, our dictionary for `ngram=4` gives us 232414 features for our model:

```python
print(X.shape)
print(X_chimp.shape)
print(X_dog.shape)
```

```
(4380, 232414)
(1682, 232414)
(820, 232414)
```


So, for humans we have 4380 genes converted into **uniform length feature vectors of 4-gram k-mer** (length 6) counts. For chimpanzee and dogs, we have the same number of features with 1682 and 820 genes respectively since we `fit` on the human dataset.

## Training Model

So now that we know how to transform our DNA sequences into uniform length numerical vectors in the form of k-mer counts and ngrams, we can now go ahead and build a classification model that can predict the DNA sequence function based only on the sequence itself.

Here we will use the human data to train the model, **holding out 20%** of the human data to test/evaluation the model. Then we can challenge the model’s generalizability by trying to predict sequence function in other species (the chimpanzee and dog).

Next, train/test split human dataset and build simple multinomial **naive Bayes classifier**

```python
# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y_human, 
                                                    test_size = 0.20, 
                                                    random_state=42)
```

```python
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
```

## Evaluation

For evaluation, we'll be checking the **confusion matrix** as well as some other metrics like **f1_score** for three different subsets of generalisation data:

### Human Dataset Prediction

Let's check how well our model performs on the hold out human dataset:

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
```

```
Confusion matrix for predictions on human test DNA sequence

Predicted   0    1   2    3    4   5    6
Actual                                   
0          99    0   0    0    1   0    2
1           0  104   0    0    0   0    2
2           0    0  78    0    0   0    0
3           0    0   0  124    0   0    1
4           1    0   0    0  143   0    5
5           0    0   0    0    0  51    0
6           1    0   0    1    0   0  263
accuracy = 0.984 
precision = 0.984 
recall = 0.984 
f1 = 0.984
```

### Chimpanzee Dataset Prediction

Let's check how well the model performs on the chimpanzee dataset:

```python
print(pd.crosstab(pd.Series(y_chim, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_chim, y_pred_chimp)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
```

```
Confusion matrix for predictions on Chimpanzee test DNA sequence

Predicted    0    1    2    3    4    5    6
Actual                                      
0          232    0    0    0    0    0    2
1            0  184    0    0    0    0    1
2            0    0  144    0    0    0    0
3            0    0    0  227    0    0    1
4            2    0    0    0  254    0    5
5            0    0    0    0    0  109    0
6            0    0    0    0    0    0  521
accuracy = 0.993 
precision = 0.994 
recall = 0.993 
f1 = 0.993
```

### Dog Dataset Prediction

Let's check the classification performance on the dog test dataset:

```python
print(pd.crosstab(pd.Series(y_dog, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
```

```
Predicted    0   1   2   3    4   5    6
Actual                                  
0          127   0   0   0    0   0    4
1            0  63   0   0    1   0   11
2            0   0  49   0    1   0   14
3            1   0   0  81    2   0   11
4            4   0   0   1  126   0    4
5            4   0   0   0    1  53    2
6            0   0   0   0    0   0  260
accuracy = 0.926 
precision = 0.934 
recall = 0.926 
f1 = 0.925
```

### Conclusion

For all gene family data, the model is able to produce good results. It also does on Chimpanzee which is because the chimpanzee and humans share the same genetic hierarchy structure. However, the performance on the dog dataset (in comparison) was not quite as good, probably because dogs and human share less common genes.

## Concluding remarks

In this post we looked at an interesting machine learning applicaiton in the field of bioinformatics. We started with a marked dataset for three difference species, containing labelled gene classes of DNA segments extracted from the genome of these species. 

Our goal was to create a machine learning model that was able to classify input DNA segments into one of the specified classes. For this we utilised a **kmer** preprocessing approach. This preprocessing step probably was the most difficult part of the entire project:

* Our gene classes contain biological sequences of various lengths, as a result, we needed to pay attention to how we would go about encoding the text data into numerical format, so we could train a classifier. 
* For this reason we resorted to **kmer subset** groupings & created a dictionary of these subset, thus the dataset ended up representing data that specified whether this **kmer** subset of the data was present in the input DNA sequence or not.

 The resulting model was able to very convinsingly carry out this task without any significant problems, even on non human datasets (chimpanzee & dogs), which is very promising.