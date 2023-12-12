# **Adding New Modules**

## **Overview**

To add a new module into **mllibs**, you need to add two components:

- Module **configuration** file
- Module **components** file 


The **configuration** file will hold text based information about each activation function in the module, whilst the **components** file will contain the relevant activation functions

## **Module Components File**

Module activation functions are grouped together in a class format. Here is an example module, **sample**, which contains an activation function **function** 

### class inheritance

Modules can inherent any class, however as a minimum, it must always inherent the **nlpi** class

### activation functions[^1]

[^1]: Activation functions are **class methods** or **staticmethods** that will be selected if the classifier predicts the relevant activation function tag (defined as "name" in the **configuration** file)

Activation functions require only a single argument, **args:dict** aside from **self**, as the dictionary should contain all the relevant functions that the functions requires

```python
# sample module class structure

class Sample(nlpi):
    
    # initialise module
    def __init__(self,nlp_config):
        self.name = 'sample'          # unique module name identifier (used in nlpm/nlpi)
        self.nlp_config = nlp_config  # text based info related to module (used in nlpm/nlpi)
        
    # function selector
    def sel(self,args:dict):
        
        # additional functionality
        self.select = args['pred_task']
        self.args = args
        
        # selection of function based on prediction
        if(self.select == 'function'):
            self.function(self.args)
        
    # activation function
    def function(self,args:dict):
        pass
```

## **Module Configuration File**

The **configuration** file contains information about the module (eg.`sample`) & its stored functions `info`, as well as the `corpus` used in classificaiton of function labels `name`

``` json
"modules": [

{
  "name": "function",
"corpus": [
          "...",
          ],
  "info": {
          "module":"sample",
          "action":"...",
          "topic":"...",
          "subtopic":"...",
          "input_format":"...",
          "description":"...",
          "output":"...",
          "token_compat":"...",
          "arg_compat":"..."
          }
},

...

]
```

## **Naming Conventions**

### Activation function name

Some important things to note:

- Module class name (eg.`Sample`) can be whatever you choose. The relevant class must then be used as import when grouping together all other modules
- Module `configuration` must contain `name` (function names) that correspond to its relevant module 

### File names

Module `components` file names can be whatever you choose it to be. Module `configuration` file names as well can be anything you choose it to be, however its good practice to choose the same name for both module components so you don't loose track of which files belong together


