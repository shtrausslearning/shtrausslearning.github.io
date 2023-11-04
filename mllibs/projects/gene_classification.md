
## :material-tag-edit-outline: **Gene Family Classification**	

![](https://img.shields.io/badge/multiclass%20classification-2E4053) ![](https://img.shields.io/badge/bioinformatics-2E4053) [![](https://img.shields.io/badge/colab_notebook-blue?logo=Jupyter)](https://colab.research.google.com/drive/1TU9_w1eWTsnObTKPKWNcR265EVUHmNWW?usp=sharing)

!!! abstract

	![](https://img.shields.io/badge/approval-pending-oranges) ![](https://img.shields.io/badge/module name-bio-blue)

	A project that involves the classification of DNA stips into groups of gene classes[^1]. The project resembles a typical **NLP** related classification problem, which requires text (or in this case **biological sequence**) preprocessing.

	### :material-sitemap-outline: Project Steps

	* :fontawesome-solid-square-caret-right: 1. Problem Definition (**not required**)
	* :fontawesome-solid-square-caret-right: 2. Loading Biological Data ()
	* :fontawesome-solid-square-caret-right: 3. Biological Data Preprocessing ()

		* create_ordinal_X_fixed
		* create_kmer_X


	* :fontawesome-solid-square-caret-right: 4. Data Exploration ()
	* :fontawesome-solid-square-caret-right: 5. Feature Engineering (**not required**)
	...
	
### :octicons-git-pull-request-16: <sub>![](https://img.shields.io/badge/functions-FF5733)</sub> Biological Encoding 

Activation functions related biological encoding

!!! tip "create_ordinal_X_fixed"

	<h4>label encoding</h4>

	Functions (in order) that can be used to define a function that creates an encoding of DNA sequences in a pandas dataframe column into a **numpy matrix** 

	??? "Raw Functions"

		Define preset label encoder & helper function

		```python
		import numpy as np
		import re

		'''

		Convert/Preprocess string into a list of nucleotide bases

		'''

		# preprocess string
		def lst_string(seq_string:str) -> list[str]:
		    seq_string = seq_string.lower()

		    # check validity
		    # if(len(re.findall('[^acgt]',seq_string)) > 0):
		    #     print(re.findall('[^acgt]',seq_string))

		    # seq_string = re.sub('[^acgt]', 'n', seq_string) # replace all not acgt w/ n
		    seq_string = re.sub('[^acgt]', '', seq_string) # replace all not acgt w/ n
		    seq_string = np.array(list(seq_string))
		    return seq_string

		# create a label encoder with 'acgtn' alphabet (initialise)
		from sklearn.preprocessing import LabelEncoder
		label_encoder = LabelEncoder()
		label_encoder.fit_transform(np.array(['a','c','g','t','z']))
		```

		Convert a dataframe column into a list of encodings

		```python
		'''

		Encode DataFrame Column containg biological sequence

		'''

		# use with .apply() in column
		# input: [str] biological sequence
		# output: [pd.Series] list of ordinal encoding

		# encode list of strings
		def ordinal_encoder_full(seq:str):

		    # convert string to nucleotide
		    my_array = lst_string(seq)

		    # transform ordinally
		    integer_encoded = label_encoder.transform(my_array)
		    float_encoded = integer_encoded.astype(float)

		    # substitute value
		    float_encoded[float_encoded == 0] = 0.25 # A
		    float_encoded[float_encoded == 1] = 0.50 # C
		    float_encoded[float_encoded == 2] = 0.75 # G
		    float_encoded[float_encoded == 3] = 1.00 # T
		    float_encoded[float_encoded == 4] = 0.00 # anything else
		    return float_encoded

		# use 
		human_dna['test'] = human_dna['sequence'].apply(ordinal_encoder_str)
		human_dna['test']
		```

		Truncate/Pad encoded sequences based on the desired length

		```python
		'''

		Truncate Sequences to smallest sequence

		'''

		# truncate or pad sequence depending on its size

		from itertools import repeat, chain, islice

		# helper function
		def trimmer(seq, size=10, filler=0):
		    return np.array(list(islice(chain(seq, repeat(filler)), size)))

		# main function
		def truncate_sequence(df:pd.DataFrame,column:str,length:int):
		    df['test2'] = df['test'].apply(lambda x: trimmer(x,size=length))
		    return df

		# add vector encoded data into "test2"
		truncated_encoded = truncate_sequence(human_dna,'test',length=100)
		```

		Merge all vectors in dataframe column into one matrix

		```python
		'''

		Merge all vectors created in pandas dataframe column

		'''

		# merge all arrays in column into one matrix
		def merge_encoding_arrays(df:pd.DataFrame,column:str):
		    return np.array(df[column].tolist())

		merge_encoding_arrays(truncated_encoded,'test2')
		```

	!!! success "Created activation functions"

		```python
		from itertools import repeat, chain, islice
		from sklearn.preprocessing import LabelEncoder
		import numpy as np
		import re

		'''

		# requires: data: [pd.DataFrame] Input Dataset containing column with biological sequences
		#         : column [str] Column name containing biological sequence data / corpus
		          : seq_len [int] Sequence Lenth Fixation Count (Pad/Truncate)

		# returns : X [np.array] matrix of encoded dataset

		'''

		def create_ordinal_X_fixed(inputs:dict):

		    # preprocess string
		    def lst_string(seq_string:str) -> list[str]:
		        seq_string = seq_string.lower()
		        seq_string = re.sub('[^acgt]', '', seq_string) # replace all not acgt w/ n
		        seq_string = np.array(list(seq_string))
		        return seq_string

		    # create a label encoder with 'acgtn' alphabet (initialise)
		    label_encoder = LabelEncoder()
		    label_encoder.fit_transform(np.array(['a','c','g','t','z']))

		    # encode list of strings
		    def ordinal_encoder_full(seq:str):

		        # convert string to nucleotide
		        my_array = lst_string(seq)

		        # transform ordinally
		        integer_encoded = label_encoder.transform(my_array)
		        float_encoded = integer_encoded.astype(float)

		        # substitute value
		        float_encoded[float_encoded == 0] = 0.25 # A
		        float_encoded[float_encoded == 1] = 0.50 # C
		        float_encoded[float_encoded == 2] = 0.75 # G
		        float_encoded[float_encoded == 3] = 1.00 # T
		        float_encoded[float_encoded == 4] = 0.00 # anything else
		        return float_encoded

		    # series containing ordinally encoded vectors
		    inputs['data']['ordinal'] = inputs['data'][inputs['column']].apply(ordinal_encoder_full)

		    # truncate or pad sequence depending on its size
		    def trimmer(seq, size=10, filler=0):
		        return np.array(list(islice(chain(seq, repeat(filler)), size)))

		    def truncate_sequence(df:pd.DataFrame,column:str,length:int):
		        df[f'ordinal_{length}'] = df['ordinal'].apply(lambda x: trimmer(x,size=length))
		        return df

		    # truncate/pad sequences to desired lenght
		    truncated_encoded = truncate_sequence(inputs['data'],'test',length=inputs['seq_len'])

		    # delete 
		    del inputs['data']['ordinal']

		    # merge all arrays in column into one matrix
		    def merge_encoding_arrays(df:pd.DataFrame,column:str):
		        return np.array(df[column].tolist())

		    # need to store data
		    X = merge_encoding_arrays(truncated_encoded,f"ordinal_{inputs['seq_len']}")

		# required data
		dict_data = {'data':human_dna,'column':'sequence','seq_len':100}

		# execute
		create_ordinal_X_fixed(dict_data)
		```

!!! tip "bencode_kmerX"

	<h4>kmer encoding</h4>

	The next encoding approach that can be added is **k-mers** encoding, it uses `kmers_count_all` should be used with column `.apply()`, which creates a list of kmers. The next step would be to utilise `CountVectorizer` in order to create a dictionary of kmers from which we can obtain a uniform length array for each row

	??? "**Raw Functions**"

		```python
		# create list of kmers 
		def kmers_count(seq:str, size:int=6) -> list:
		    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

		# create string of concatenated kmers (inactive)
		def kmers_count_string(seq:str,size:int=6) -> str:
		    lst_kmers = [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]
		    return  ' '.join(lst_kmers)

		'''

		Create a list of kmers

		'''

		# create kmers lists for all sequences in dataset
		def kmers_count_all(df:pd.DataFrame,column:str):

		    # add kmers list to dataframe
		    df['kmers_list'] = df.apply(lambda x: kmers_count(x[column]), axis=1)
		    df = df.drop(column, axis=1)

		    # create list of strings containing kmers
		    df_list = list(df['kmers_list'])
		    for item in range(len(df_list)):
		        df_list[item] = ' '.join(df_list[item])

		    return df_list

		```

		```python
		'''

		Create kmer list for all rows in dataframe

		'''

		lst_kmers_human = kmers_count_all(human_dna,'sequence')
		lst_kmers_dog = kmers_count_all(dog_dna,'sequence')
		```

		Create bag of words encoding using dataframe column string data

			```python
			# fit and predict on one corpus (list)
			# input: lst [list] list containing corpus
			# output: cv,X encoder & matrix

			def create_bow_fit_predict(lst:list):
			    cv = CountVectorizer(ngram_range=(4,4))
			    X = cv.fit_transform(lst)
			    return cv,X

			# predict on one corpus (list), using an existing encoder
			# input: lst [list] list containing corpus
			#         cv [CountVectorizer] fitted instance
			# output: X  

			def create_bow_predict(lst:list,cv=None):

			    try:
			        X = cv.transform(lst)  
			    except:
			        X = None
			    return X

			cv,X = create_bow_fit_predict(lst_kmers_human)
			X_dog = create_bow_predict(lst_kmers_dog,cv)
			```

	!!! success "created activation functions"

		Purpose: create a numpy matrix that can be used as input for a machine learning model feature matrix

		```python
		'''

		create kmers column + encode

		'''

		# requires: data: [pd.DataFrame] Input Dataset containing column with biological sequences
		#         : column [str] Column name containing biological sequence data / corpus

		# returns : cv [CountVectorizer] encoder of kmers corpus
		#            X [np.array] matrix of encoded dataset

		def create_kmer_X(inputs:dict):

		    # create list of kmers
		    def kmers_count(seq:str, size:int=6) -> list:
		        return ' '.join([seq[x:x+size].lower() for x in range(len(seq) - size + 1)])

		    # create kmers lists for all sequences in dataset
		    def kmers_count_all(df:pd.DataFrame,column:str):

		        # add kmers list to dataframe
		        df['kmers_str'] = df.apply(lambda x: kmers_count(x[column]), axis=1)
		        # df = df.drop(column, axis=1)

		        return df

		    # dataframe containing new column [kmers_str]
		    kmers = kmers_count_all(inputs['data'],inputs['column'])

		    def create_bow_fit_predict(lst:list):
		        cv = CountVectorizer(ngram_range=(4,4))
		        X = cv.fit_transform(lst)
		        return cv,X

		    cv,X = create_bow_fit_predict(list(kmers['kmers_str'].values))
		    return cv,X
		    
		# requirement
		dict_data = {'data':human_dna,'column':'sequence'}

		# execute
		create_kmer_X(dict_data)
		```

### :octicons-git-pull-request-16: <sub>![](https://img.shields.io/badge/info-FFC300)</sub> Module Information

!!! abstract

	```json
	{
	  "modules": [
	    
	    {
	      "name": "bencode_ordinalX_fixed",
	    "corpus": [
	                "create biological ordinal encoding",
	                "ordinal encoding for biological data"
	              ],
	      "info": {
	              "module":"pbio_classification",
	              "action":"encode text",
	              "topic":"bioinformatics",
	              "subtopic":"sequence encoding",
	              "input_format":"pd.DataFrame",
	              "description":"create ordinal encoding of a biological sequence",
	              "output":"vectoriser np.array",
	              "token_compat":"data",
	              "arg_compat":"column"
	              }
	    },

	    {
	      "name": "bencode_kmerX",
	    "corpus": [
	                "create biological kmer encoding",
	                "biological kmer encoding",
	                "create biological sequence kmer encoding"
	              ],
	      "info": {
	              "module":"pbio_classification",
	              "action":"encode text",
	              "topic":"bioinformatics",
	              "subtopic":"sequence encoding",
	              "input_format":"pd.DataFrame",
	              "description":"create kmer encoding of a biological sequence",
	              "output":"vectoriser np.array",
	              "token_compat":"data",
	              "arg_compat":"column"
	              }
	    }
	  ]
	}
	```

*[biological sequence]: A typical sequence: ATCGGCTAAGTCCGTAGCCTAGCGGATCGATCGA

[^1]: A gene class is a group of genes that share similar characteristics or functions. Genes can be classified based on various criteria such as their sequence, expression pattern, or biological function. For example, genes involved in the same metabolic pathway can be grouped together as a gene class. Similarly, genes that encode proteins with similar structural features or perform similar biological roles can also be classified into a gene class. Gene classes can be useful for understanding the organization and regulation of genetic information and for predicting the function of newly discovered genes.

