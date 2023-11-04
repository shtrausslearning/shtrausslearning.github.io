---
tags:
  - recommendation system
  - group task
  - iterative process	
---

	
## :fontawesome-regular-comment-dots: **Recommendation System**

Providing recommendation during the process of a user using the interpreter can significantly simplify the process of a machine learning project.

## **Phases**

The entire recommendation system can be split into different phases/components

### `phase 1`

Phase one involves the extraction of as much information provided by a user relevant to the project they are about to begin as possible, all relevant data will be stored in the interpreter `nlpi` and utilised during the iterative process

### `phase 2`

TBD

## **Tasks**

??? "**User Description Classifier**"

	### :octicons-tasklist-16: **User Description Classifier**

	<h4>Phase</h4>

	1 

	<h4>GitHub Projects ID</h4>

	rst1[^1]

	[^1]: [github project](https://github.com/mllibs/mllibs/issues/3)

	<h4>Dependencies</h4>

	As the dataset is created from scratch & due to the wide range of topics, the user should know which types of machine learning project related modules are going to be implemented. This is due to the fact that the user is going to create a corpus and subsequent labels that represent different project topics

	<h4>Required Library Knowledge</h4>

	The task does not require knowledge of the `mllibs` code

	<h4>Brief Description</h4>

	A user can input a desciption that would describe the type of project they are about to do

	<h4>Purpose</h4>

	The purpose of such a classifier is to **interpret** what type of project the user is about to do. 

	<h4>Why this is needed</h4>

	Such a feature would help identify which **activation functions** & their corresponding **text examples** a user can input

	<h4>When is it used?</h4>

	The classifier would be utilised after a user has created an `nlpi` instance

	<h4>Dataset</h4>

	The entire dataset needs to be manually assembled

	`Text` : User machine learning project description (corpus) <br>
	`Labels` : Project Type Labels

	<h4>Examples</h4>

	> **[sentiment analysis]**<br>
	> I'm going to be doing a natural language processing related project, in which I want to sentiment analysis. I want to train a model & 

	> **[credit risk assessment]** <br>
	> Today I will be doing a project in which I will create a model that will be able to assess the credit worthiness of individuals and customers

	> **[recommendation system]** <br>
	> My project will involve creating a recommendation system for our clients

	> **[sales forcasting]** <br>
	> I'll be doing a project which will involve me creating a model that will be used to predict sales volumes 

	> **[customer churn]**
	> I want to create a model that will be used to predict which customers are likely to leave our business

	<h4>How will the output be used?</h4>

	Having classified the user text, the label(s) will be stored in the `nlpi` class and utilised during the iterative process of the project

	<h4>Desired Result</h4>

	* Corpus in `csv` format
	* Trained model in `pickle` format (if classical) 
	* Jupyter notebook format (`ipynb`)

	<h4>additional</h4>

	[starter notebook]()[^2]

	[^2]: not yet available


