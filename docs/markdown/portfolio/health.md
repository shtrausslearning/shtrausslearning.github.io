<center>
![img](images/ds_bnb.png#only-light){ width="450" }
![img](images/ds_wnb.png#only-dark){ width="450" }
</center>

## :fontawesome-regular-hospital: <b>Health</b> 

Health is an important topic in machine learning because it has the potential to significantly improve healthcare outcomes. Machine learning algorithms can be used to analyze large amounts of medical data and identify patterns that may not be immediately apparent to human analysts. This can help doctors and researchers make more accurate diagnoses, develop more effective treatments, and even predict and prevent certain diseases.

### :material-label-variant-outline: <b><span style='color:#FFCA58;text-align:center'></span>Lower Back Pain Symptoms Modeling</b>

!!! tip "Lower Back Pain Symptoms Modeling"

	[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://www.kaggle.com/code/shtrausslearning/lower-back-pain-symptoms)

	In this study we investigate patient back pain **[biomedical data](https://doi.org/10.24432/C5K89B)** obtained from a medical resident in Lyon. We create a classification model which is able to determine the difference between **normal patients** and patients who have either **disk hernia** or **spondylolisthesis**, which is a binary classification problem. We utilise **PyTorch** and created a **custom dataset class** to load the tabular CSV data & load the data into batches using **data loaders**. A rather simple **neural network structure** that utilises standard **generalisation strategies** such as **dropout** and **batch normalisation** was assembled & the model was trained and tested in the validation dataset.

	<center>
	![](images/pairplot3.png)
	</center>

### :material-label-variant-outline: <b><span style='color:#FFCA58;text-align:center'></span>Ovarian Phase Classification in Felids</b>

!!! tip "Ovarian Phase Classification in Felids"

	[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://www.kaggle.com/code/shtrausslearning/ovarian-phase-classification-in-felids)
	[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/shtrausslearning/Data-Science-Portfolio/blob/main/ovarian-phase-classification-in-felids.ipynb)

	In this study, we investigate feline reproductology data, conducting an exploratory data analysis of experimental measurements of **estradiol** and **progesterone** levels and attempt to find the relation between different hormone levels during different phases of pregnancy. We  then use the available data to create machine learning models that are able to predict at which stage of an estrous cycle a feline is at the time of testing for different measurement methods, which is a **multiclass classification problem**.

	<center>
	![](images/roc_curve.png)
	</center>

### :material-label-variant-outline: <b><span style='color:#FFCA58;text-align:center'></span>Heart Disease Classification</b>

!!! tip "Heart Disease Classification"

	[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://www.kaggle.com/code/shtrausslearning/heart-disease-gaussian-process-models)
	[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/shtrausslearning/Data-Science-Portfolio/tree/main/Heart%20Disease%20Classification)

	In this study, we explore different **feature engineering** approaches, we group features into different combinations based on their subgroup types and attempt to find the best combination for classifying patients with heart disease. Having found the best feature combinations, we utilise brute force grid searches for hyperparameter optimisation in order to find the best performing model.

	We utilise an sklearn compatible custom Regressor model **([model found here](https://github.com/shtrausslearning/Data-Science-Portfolio/blob/main/Heart%20Disease%20Classification/ml-models/src/mlmodels/gpr_bclassifier.py))** based on **Kriging**, which we turned in a classifier by simply setting the threshold to 0.5 (basically the **prediction** method in sklearn models). We also tried different ensembles of different models in order to improve the model accuracy even further.

	<center>
	![](images/heart1.png)
	![](images/heart2.png)
	</center>

---

**Thank you for reading!**

Any questions or comments about the posts below can be addressed to the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**
