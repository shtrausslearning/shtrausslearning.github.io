---
date: 2023-10-21
title: Gene Classification
authors: [andrey]
categories:
     - science
tags:
     - classification
     - bioinformatics
---

# **Gene Classification**

In this notebook, we look at how to work with biological sequence data, by venturing into a classification problem, in which we will be classifying between six different genes groups common to three different species (human,chimpanzee & dog)

<!-- more -->

## Background

### Gene Classification Problem

**Gene classification** using machine learning is the process of using algorithms and statistical models to analyze large datasets of genetic information and predict the function or characteristics of different genes. Machine learning techniques can be used to identify patterns in gene expression data, classify genes into different functional categories, or predict the likelihood of a gene being associated with a particular disease or phenotype. This approach can help researchers to better understand the complex relationships between genes, their functions, and their interactions with other biological systems.

### Genes

The dataset that we'll be using contains the following gene classes for three different species, **human**, **chimpanzee** & **dog**, it is added more as a reference for those interested in what each class represents in our data

**Gene families** are groups of related genes that share a common ancestor. Members of gene families may be paralogs or orthologs. Gene paralogs are genes with similar sequences from within the same species while gene orthologs are genes with similar sequences in different species.


<h4>G Protein-Coupled Receptors</h4>

The **G protein-coupled receptors** (GPCRs) gene family is a large and diverse group of genes that encode proteins involved in cellular signaling pathways. These receptors are located on the surface of cells and are activated by a wide range of ligands, including hormones, neurotransmitters, and environmental stimuli. GPCRs play critical roles in regulating many physiological processes, such as sensory perception, hormone secretion, and immune response. Dysregulation of GPCR signaling has been implicated in a variety of diseases, including cancer, diabetes, and cardiovascular disorders.

<h4>Tyrosine Kinase</h4>

The **tyrosine kinase** gene family is a group of genes that encode proteins involved in cellular signaling pathways. These proteins are enzymes that add phosphate groups to specific tyrosine residues on target proteins, thereby regulating their activity. Tyrosine kinases play critical roles in many physiological processes, such as cell growth, differentiation, and survival. Dysregulation of tyrosine kinase signaling has been implicated in a variety of diseases, including cancer, autoimmune disorders, and developmental disorders. Examples of tyrosine kinase genes include EGFR, HER2, and BCR-ABL.

<h4>Tyrosine Phosphatase</h4>

The **tyrosine phosphatase** gene family is a group of genes that encode proteins involved in cellular signaling pathways. These proteins are enzymes that remove phosphate groups from specific tyrosine residues on target proteins, thereby regulating their activity. Tyrosine phosphatases play critical roles in many physiological processes, such as cell growth, differentiation, and survival. Dysregulation of tyrosine phosphatase signaling has also been implicated in a variety of diseases, including cancer, autoimmune disorders, and developmental disorders. Examples of tyrosine phosphatase genes include PTPN1, PTPN6, and PTPN11.

<h4>Synthetase</h4>

The **synthetase** gene family is a group of genes that encode for enzymes called aminoacyl-tRNA synthetases. These enzymes are responsible for attaching specific amino acids to their corresponding tRNA molecules during protein synthesis. There are 20 different aminoacyl-tRNA synthetases, one for each amino acid, and each enzyme recognizes and binds to its specific amino acid and tRNA molecule. The synthetase gene family is highly conserved across all living organisms and mutations in these genes can lead to various genetic disorders.

<h4>Synthase</h4>

The **synthase** gene family is a group of genes that encode enzymes responsible for synthesizing various molecules within cells. These enzymes are involved in a wide range of biological processes, including the synthesis of lipids, nucleotides, and amino acids. Different members of the synthase gene family may be involved in different aspects of these processes, and mutations in these genes can lead to a variety of diseases and disorders. Examples of synthase genes include fatty acid synthase, which is involved in the synthesis of fatty acids, and adenylyl cyclase, which synthesizes the signaling molecule cyclic AMP.

<h4>Ion Channel</h4>

The **ion channel** gene family is a group of genes that encode proteins responsible for the transport of ions across cell membranes. These proteins are integral membrane proteins that form pores or channels in the lipid bilayer of the cell membrane, allowing the selective movement of ions such as sodium, potassium, calcium, and chloride. Ion channels are critical for many physiological processes, including muscle contraction, nerve signaling, and hormone secretion. Dysregulation of ion channel activity has been implicated in a variety of diseases, including epilepsy, cardiac arrhythmias, and cystic fibrosis. Examples of ion channel genes include SCN1A, KCNQ1, and CFTR.

<h4>Transcription Factors</h4>

The **transcription factor** gene family is a group of genes that encode proteins responsible for regulating the expression of other genes. These proteins bind to DNA and control the rate at which genes are transcribed into mRNA, which is then translated into proteins. Transcription factors are involved in a wide range of biological processes, including development, differentiation, and response to environmental stimuli. Different members of the transcription factor gene family may have different target genes and regulatory mechanisms, allowing for precise control of gene expression. Mutations in these genes can lead to a variety of diseases and disorders, including cancer and developmental disorders. Examples of transcription factor genes include homeobox genes, which regulate embryonic development, and p53, which regulates cell cycle progression and DNA repair

## Dataset

We're loading three DNA datasets, our dataset is in the form of a `sequence` & subsequent **gene family** label, `class`

```python
import pandas as pd

human_dna = pd.read_table('../input/dna-sequence-dataset/human.txt')
chimp_dna = pd.read_table('../input/dna-sequence-dataset/chimpanzee.txt')
dog_dna = pd.read_table('../input/dna-sequence-dataset/dog.txt')
```

## DNA Encoding

Biological sequences come in the format:

> GTGCCCAGGTTCAGTGAGTGACACAGGCAG

This mimics a standard **NLP** based problem, in which we need to convert text into numerical representation before we can feed this data into our models

There are 3 general approaches to encode biological sequence data:

1. Ordinal encoding DNA Sequence
2. One-Hot encoding DNA Sequence
3. DNA sequence as a “language”, known as `k-mer` counting

So let us implement each of them and see which gives us the perfect input features.

### Encoding Sequences

#### Ordinal Encoding


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

#### One-Hot Encoding

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

#### K-MER Counting

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

For **DNA** sequence classification methods, its more logical to utilise the `kmers` approach since `kmers` are commonly used in sequence analysis and genome assembly, as they can provide information about the composition and structure of the sequence.


## Problem Statement

Our objective is to train a classification model that is trained on the human DNA sequence and can predict a gene family based on the DNA sequence of the coding sequence. To test the model, we will use the DNA sequence of humans, dogs, and chimpanzees and compare model accuracies.



