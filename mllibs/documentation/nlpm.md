# 1. Class Code Walkthrough

## 1.1. Loading Added Modules 

Modules are grouped together in ==**nlpm.load**==. Cycling through all **modules**, temporary data label data  <b>`dict_task_names`</b> (module corpus) & `lst_module_info` (module information)

### Code Rundown

| Code        | Description                          |
| ----------- | ------------------------------------ |
| **`14-38`** | Added module instances are saved into ==**self.modules**== & will be passed into `nlpi` instance in order to gain access to activation functions. **Task labels** for **each module** (in text form) are stored in ==**self.label**== (this attribute will end up storing all model labels) |
| **`46-74`** | Prepare ==**self.mod_summary**==, which includes information (`info`) as well as `label` data for each **module task** (activation function) |
| **`46-74`** | Create labels for module selection (==**self.corpus_ms**== : **ms**) & global task selection (==**self.corpus_gt**== : **gt**)
| **`86-92`** | Create labels for **global task model** ==**self.corpus_gt**== & **module selection model** ==**self.corpus_gt**== |
| **`94-104`** | Create labels for additional corpus found in `info` |
| **`118-147`** | Prepare the module task corpus (==**self.corpus_mt**==) |
| **`155-178`** | Prepare all other corpuses; ==**self.corpus_ms**== (module selection corpus), ==**self.corpus_gt**== (global task corpus) & other corpuses based on `info` data. These are created from the ==**self.mod_summary**== subset of data |

```python linenums="1" hl_lines="14-38 46-74 86-92 94-104 118-147 155-178" 
def load(self,modules:list):
        
    print('[note] loading modules ...')
    
    # dictionary for storing model label (text not numeric)
    self.label = {} 
    
    # combined module information/option dictionaries
    
    lst_module_info = []
    lst_corpus = []
    dict_task_names = {}

    for module in modules:  
        
        # get & store module functions
        self.modules[module.name] = module
        
        # get dictionary with corpus
        tdf_corpus = module.nlp_config['corpus']   

        # dictionary of corpus
        df_corpus = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_corpus.items()]))
    
        # module task list
        dict_task_names[module.name] = list(df_corpus.columns)  # save order of module task names

        lst_corpus.append(df_corpus)
        self.task_dict[module.name] = tdf_corpus     # save corpus
        
        # combine info of different modules
        opt = module.nlp_config['info']     # already defined task corpus
        tdf_opt = pd.DataFrame(opt)
        df_opt = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_opt.items()]))
        lst_module_info.append(df_opt)

    # update label dictionary to include loaded modules
    self.label.update(dict_task_names)  
        
    ''' 

    Step 1 : Create Task Corpuses (dataframe) 

    '''
        
    # task corpus (contains no label)
    corpus = pd.concat(lst_corpus,axis=1)
    
    ''' 

    Step 2 : Create Task information dataframe 

    '''
    # create combined_opt : task information data
    
    # task information options
    combined_opt = pd.concat(lst_module_info,axis=1)
    combined_opt = combined_opt.T.sort_values(by='module')
    combined_opt_index = combined_opt.index
    
    
    ''' Step 3 : Create Module Corpus Labels '''         
    print('[note] making module summary labels...')

    # note groupby (alphabetically module order) (module order setter)
    module_groupby = dict(tuple(combined_opt.groupby(by='module')))
    unique_module_groupby = list(module_groupby.keys())  # [eda,loader,...]

    for i in module_groupby.keys():
        ldata = module_groupby[i]
        ldata['task_id'] = range(0,ldata.shape[0])

    df_opt = pd.concat(module_groupby).reset_index(drop=True)
    df_opt.index = combined_opt_index
    
    # module order for ms
    self.mod_order = unique_module_groupby
    
    ''' 

    Step 4 : labels for other models (based on provided info) 

    '''
    
    # generate task labels    
    encoder = LabelEncoder()
    df_opt['gtask_id'] = range(df_opt.shape[0])
    self.label['gt'] = list(combined_opt_index)
    
    encoder = clone(encoder)
    df_opt['module_id'] = encoder.fit_transform(df_opt['module'])   
    self.label['ms'] = list(encoder.classes_)
    
    encoder = clone(encoder)
    df_opt['action_id'] = encoder.fit_transform(df_opt['action'])
    self.label['act'] = list(encoder.classes_)
    
    encoder = clone(encoder)
    df_opt['topic_id'] = encoder.fit_transform(df_opt['topic'])
    self.label['top'] = list(encoder.classes_)
    
    encoder = clone(encoder)
    df_opt['subtopic_id'] = encoder.fit_transform(df_opt['subtopic'])
    self.label['sub'] = list(encoder.classes_)
    
    # Main Summary
    self.mod_summary = df_opt
    
    # created self.mod_summary
    # created self.label
    
    ''' 

    Make Module Task Corpus 

    '''
    
    lst_modules = dict(list(df_opt.groupby('module_id')))
    module_task_corpuses = OrderedDict()   # store module corpus
    module_task_names = {}                 # store module task names
    
    for ii,i in enumerate(lst_modules.keys()):
        
        columns = list(lst_modules[i].index)      # module task names
        column_vals =  corpus[columns].dropna()
        module_task_names[unique_module_groupby[i]] = columns

        lst_module_classes = []
        for ii,task in enumerate(columns):
            ldf_task = column_vals[task].to_frame()
            ldf_task['class'] = ii

            lst_module_classes.append(pd.DataFrame(ldf_task.values))

        tdf = pd.concat(lst_module_classes)
        tdf.columns = ['text','class']
        tdf = tdf.reset_index(drop=True)                
        
        module_task_corpuses[unique_module_groupby[i]] = tdf

    # module task corpus
    # self.module_task_name = module_task_names

    self.label.update(module_task_names) 

    # dictionaries of dataframe corpuses
    self.corpus_mt = module_task_corpuses 

    ''' 

    Make Global Task Selection Corpus 

    '''

    def prepare_corpus(group):
    
        lst_modules = dict(list(df_opt.groupby(group)))

        lst_melted = []                
        for ii,i in enumerate(lst_modules.keys()):    
            columns = list(lst_modules[i].index)
            column_vals = corpus[columns].dropna()
            melted = column_vals.melt()
            melted['class'] = ii
            lst_melted.append(melted)

        df_melted = pd.concat(lst_melted)
        df_melted.columns = ['task','text','class']
        df_melted = df_melted.reset_index(drop=True)
        
        return df_melted

    # generate task corpuses
    self.corpus_ms = prepare_corpus('module_id') # modue selection dataframe
    self.corpus_gt = prepare_corpus('gtask_id')  # global task dataframe
    self.corpus_act = prepare_corpus('action_id') # action task dataframe
    self.corpus_top = prepare_corpus('topic_id') # topic task dataframe
    self.corpus_sub = prepare_corpus('subtopic_id') # subtopic tasks dataframe
```

## 1.2. Training Module Classifiers

Having created a corpus(es) and related items, the activation function models need to be trained, this is done via ==**self.train()**==. All models are trained using exactly the same methodology defined in ==**self.mlloop**==

### ==**self.train**== method code **[rundown](https://www.youtube.com/watch?v=yVnbTEjOhvs&t=41s)**

Below is a table showing the code line & relevant explanation

| Code        | Description                          |
| ----------- | ------------------------------------ |
| **`14-15`** | Global storages for both **encoders** (==**self.vectoriser**==) and their corresponding **models** (==**self.model**==) are created |
| **`23-25`** | Train all module task (activation function) models[^1]; classifiers that can be used to **determine which task** in a selected module to activate (these are module local models). These models can be used together with a **module selection classification model** (==**ms**==)[^2] | 
| **`32`**    | Train a **module selection classification model**, such a model can be used together with **local module task models** (activation function models) in order to create a two step logic function selection process (1. classify which module then classify which module function to activate) | 
| **`36`** |Train a **global task classifer**; all tasks (activation) functions (in all added modules) are given a unique label & a classifier is trained (this is the currently implemented approach for the main interpreter ==**nlpm**== classifier) |
| **`37-39`** | Additional models for **module based selection** can be trained using the `info` section of the training data found in `src/corpus` **json** files. These models use task based info and group them together into one corpus. This is useful if we add activation functions that are related. |
| **`43`** | Train **NER** model, that is used to identify **activation function** parameters, ...

[^1]: These local module models are currently not being utilised
[^2]: The module selection module is also not currently being utilised

```python linenums="1" hl_lines="14-15 23-25 32 36 37-39 43"
'''

TRAIN RELEVANT MODELS

'''

# module selection model [ms]
# > module class models [module name] x n modules

def train(self,type='mlloop'):
                
    if(type == 'mlloop'):
    
        self.vectoriser = {} # stores vectoriser
        self.model = {}   # storage for models

        ''' 

        [1] Create module task model for each module 

        '''

        for ii,(key,corpus) in enumerate(self.corpus_mt.items()):  
            module_name = self.mod_order[ii]
            self.mlloop(corpus,module_name)

        ''' 

        [2] Create Module Selection Model

        '''
        self.mlloop(self.corpus_ms,'ms')

        ''' Other Models '''

        self.mlloop(self.corpus_gt,'gt')
#         self.mlloop(self.corpus_act,'act')
#         self.mlloop(self.corpus_top,'top')
#         self.mlloop(self.corpus_sub,'sub')

        self.toksub_model()  # not used
        self.ner_tokentag_model() # not used
        self.ner_tagger()

        print('models trained...')
```

### ==**self.mlloop**== method code rundown

The models in ==**mllop**== are trained using the following methodology: 

> Input tokens are tokenised using the `nltk` ==**WhitespaceTokenizer**==, text is converted into **numerical vectors** using `TfidfVectorizer`. This vectoriser is saved for future use i ==**self.vectoriser**==. The data is then fed into an enseble model ==**RandomForestClassifier**==. The trained model is saved in ==**self.models**==

```python linenums="1"         
''' 

MACHINE LEARNING LOOP 

'''

def mlloop(self,corpus:dict,module_name:str):

    # corpus : text [pd.Series] [0-...]
    # class : labels [pd.Series] [0-...]
    
    # lemmatiser
#        lemma = WordNetLemmatizer() 
    
    # define a function for preprocessing
#        def clean(text):
#            tokens = word_tokenize(text) #tokenize the text
#            clean_list = [] 
#            for token in tokens:
#                lemmatizing and appends to clean_list
#                clean_list.append(lemma.lemmatize(token)) 
#            return " ".join(clean_list)# joins the tokens

#         clean corpus
#        corpus['text'] = corpus['text'].apply(clean)
    
    ''' 

    Convert text to numeric representation 

    '''
    
    # vect = CountVectorizer()
#        vect = CountVectorizer(tokenizer=lambda x: word_tokenize(x))
    # vect = CountVectorizer(tokenizer=lambda x: WhitespaceTokenizer().tokenize(x))
    # vect = CountVectorizer(tokenizer=lambda x: nltk_wtokeniser(x),
                           # stop_words=['create'])
    vect = TfidfVectorizer(tokenizer=lambda x: nltk_wtokeniser(x))
    vect.fit(corpus['text']) # input into vectoriser is a series
    vectors = vect.transform(corpus['text']) # sparse matrix
    self.vectoriser[module_name] = vect  # store vectoriser 

    ''' 

    Make training data 

    '''
    
    # X = np.asarray(vectors.todense())
    X = vectors
    y = corpus['class'].values.astype('int')

    ''' 

    Train model on numeric corpus 

    '''
    
    # model_lr = LogisticRegression()
    # model_dt = DecisionTreeClassifier()
    model_rf = RandomForestClassifier()

    # model = clone(model_lr)
    model = clone(model_rf)

    # train model
    model.fit(X,y)
    self.model[module_name] = model # store model
    score = model.score(X,y)
    print(f"[note] training  [{module_name}] [{model}] [accuracy,{round(score,3)}]")
```

## 1.3. Testing Models

Once we have trained the classification models/(encoders/vectorisers) using ==**train**==, they are stored in ==**self.models**== & ==**self.vectoriser**== respectively, and thus we can utilise them to test how well our models generalise on new data.

To test how well the models perform, we can utilise the following methods:

**Module Selection Model**

==**self.predict_module**==, ==**self.predict_task**== & ==**self.predict_gtask**== are fundamentally the same function, however they have different activation thresholds, they use method ==**self.test**==

==**name:str**== corresponds to the **key** in ==**self.model**==, ==**self.label**== ...

??? "name options"

	**keys** correspond to the corpuses/labels you have added:

	* **ms** : module selection
	* **gt** : global task selection
	* **(module_name)** : imported module name (as defined in the module)

	...

	When training the model, they are clealy visible:

	* [note] training  [eda_plot] [RandomForestClassifier()] [accuracy,1.0]
	* [note] training  [eda_scplot] [RandomForestClassifier()] [accuracy,1.0]
	* [note] training  [eda_simple] [RandomForestClassifier()] [accuracy,1.0]
	* [note] training  [ms] [RandomForestClassifier()] [accuracy,1.0]
	* [note] training  [gt] [RandomForestClassifier()] [accuracy,1.0]

=== "Module Prediction"

	```python linenums="1"  
	def predict_module(self,name:str,command:str):
	    pred_per = self.test(name,command)     # percentage prediction for all classes
	    val_pred = np.max(pred_per)            # highest probability value
	    if(val_pred > 0.7):
	        idx_pred = np.argmax(pred_per)         # index of highest prob         
	        pred_name = self.label[name][idx_pred] # get the name of the model class
	        print(f"[note] found relevant module [{pred_name}] w/ [{round(val_pred,2)}] certainty!")
	    else:
	        print(f'[note] no module passed decision threshold')
	        pred_name = None

	    return pred_name,val_pred
	```

=== "Task Prediction"

	```python linenums="1"  
	def predict_task(self,name:str,command:str):
	    pred_per = self.test(name,command)     # percentage prediction for all classes
	    val_pred = np.max(pred_per)            # highest probability value
	    if(val_pred > 0.7):
	        idx_pred = np.argmax(pred_per)                    # index of highest prob         
	        pred_name = self.label[name][idx_pred] # get the name of the model class
	        print(f"[note] found relevant activation function [{pred_name}] w/ [{round(val_pred,2)}] certainty!")
	    else:
	        print(f'[note] no activation function passed decision threshold')
	        pred_name = None

	    return pred_name,val_pred
	```

=== "Global Task Prediction"

	```python linenums="1"  
	def predict_gtask(self,name:str,command:str):
	    pred_per = self.test(name,command)     # percentage prediction for all classes
	    val_pred = np.max(pred_per)            # highest probability value
	    if(val_pred > 0.5):
	        idx_pred = np.argmax(pred_per)         # index of highest prob         
	        pred_name = self.label[name][idx_pred] # get the name of the model class
	        print(f"[note] found relevant global task [{pred_name}] w/ [{round(val_pred,2)}] certainty!")
	    else:
	        print(f'[note] no module passed decision threshold')
	        pred_name = None

	    return pred_name,val_pred
	```

For prediction without thresholds, ==**self.dtest**== can be utilised

**Usage**

Some samples are shown below

```python
collection.predict_gtask('gt','create a column boxplot')
```

```
[note] found relevant global task [col_box] w/ [0.57] certainty!
('col_box', 0.57)
```

```python
collection.dtest('gt','create a boxplot')
```

```
available models
dict_keys(['eda_plot', 'eda_scplot', 'eda_simple', 'ms', 'gt', 'token_subset', 'token_ner'])

 	label 	  prediction
6 	sboxplot 	    0.31
13 	col_box 	    0.15
0 	sviolinplot 	0.11
4 	sscatterplot 	0.11
2 	slineplot 	    0.09
``` 


# 2. Class Attributes & Methods

## 2.1 Class Methods

In this section each of the methods is described

=== "constructor"

	The constructor contains only initialisation dictionaries

	```python
	mllibs.nlpm(self.task_dict 
	    		self.modules
	    		self.ner_identifier
	    		)
	```

=== "loading modules"

	==**load(self, modules: list)**==

	> function used to load a list of **instantiated modules**

	??? "options"

		=== "arguments"

			**modules:** : ==**list**== list of instantiated modules 

		=== "usage"

			The method can be called in the following context:

			```python linenums="1" hl_lines="2-6"
			collection = nlpm()
			collection.load([
			                 eda_simple(),    # [eda] simple pandas EDA
			                 eda_splot(),     # [eda] standard seaborn plots
			                 eda_scplot(),    # [eda] seaborn column plots
			                ])
			collection.train()
			```

=== "training models"

	==**mlloop(self, corpus: dict, module_name: str)**==

	> machine learning related training loop, selected by default in ==**train**== 

	??? "options"

		=== "arguments"

			**corpus** [dict] : dictionary containing corpus

		=== "usage"

			The method is called in ==**train**==

			```python linenums="1" hl_lines="3 5 6"
            for ii,(key,corpus) in enumerate(self.corpus_mt.items()):  
                module_name = self.mod_order[ii]
                self.mlloop(corpus,module_name)

            self.mlloop(self.corpus_ms,'ms')
            self.mlloop(self.corpus_gt,'gt')    
            self.toksub_model()
            self.ner_tokentag_model()  
            self.ner_tagger()

            print('models trained...')
			```

	==**train(self, type='mlloop')**==

	> main model training function 

	??? "options"

		=== "arguments"

			**type** [str] : model training approach identifier

		=== "usage"

			```python linenums="1" hl_lines="7"
			collection = nlpm()
			collection.load([
			                 eda_simple(),    # [eda] simple pandas EDA
			                 eda_splot(),     # [eda] standard seaborn plots
			                 eda_scplot(),    # [eda] seaborn column plots
			                ])
			collection.train()
			```

	==**ner_tagger(self)**==

	> Main **NER** tagging model 	

	??? "options"

		=== "arguments"

			**none**

		=== "usage"

			```python linenums="1" hl_lines="7"


	==**ner_tokentag_model(self)**== (inactive)

=== "model prediction"

	:octicons-file-code-16: **dtest(self, corpus: str, command: str)**

	==**predict_gtask(self, name: str, command: str)**==

	==**predict_module(self, name: str, command: str)**==

	==**predict_task(self, name: str, command: str)**==

	==**test(self, name: str, command: str)**==
