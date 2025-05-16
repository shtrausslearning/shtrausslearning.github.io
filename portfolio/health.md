---
comments: true
---

**Health** is an important topic in **machine learning** because it has the potential to significantly improve healthcare outcomes. Machine learning algorithms can be used to **analyse large amounts of medical data** and identify patterns that may not be immediately apparent to human analysts. This can help doctors and researchers make more accurate diagnoses, develop more effective treatments, and even predict and prevent certain diseases.

<br>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Lower Back Pain Symptoms Modeling](https://www.kaggle.com/code/shtrausslearning/lower-back-pain-symptoms)</b>

	---

    In this study we investigate patient back pain **[biomedical data](https://doi.org/10.24432/C5K89B)** obtained from a medical resident in Lyon. 
     
    - We create a classification model which is able to determine the difference between **normal patients** and patients who have either **disk hernia** or **spondylolisthesis**, which is a binary classification problem. 
    - We utilise **PyTorch** and created a **custom dataset class** to load the tabular CSV data & load the data into batches using **data loaders**. 
    - A rather simple **neural network structure** that utilises standard **generalisation strategies** such as **dropout** and **batch normalisation** was assembled & the model was trained and tested in the validation dataset.															

</div>

<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Ovarian Phase Classification in Felids](https://www.kaggle.com/code/shtrausslearning/ovarian-phase-classification-in-felids)</b>

	---

    In this study, we investigate **feline reproductology data**, conducting an **exploratory data analysis** of experimental measurements of **estradiol** and **progesterone** levels and attempt to find the relation between different hormone levels during different phases of pregnancy. 
    - We  then use the available data to create machine learning models that are able to predict at which stage of an estrous cycle a feline is at the time of testing for different measurement methods, which is a **multiclass classification problem**.
														

</div>


<div class="grid cards" markdown >

  - ## :material-book-check:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Heart Disease Classification](https://www.kaggle.com/code/shtrausslearning/ovarian-phase-classification-in-felids)</b>

	---

    In this study, we explore different **feature engineering** approaches, we group features into different combinations based on their subgroup types and attempt to find the best combination for classifying patients with heart disease. 
    
    - Having found the best feature combinations, we utilise brute force grid searches for hyperparameter optimisation in order to find the best performing model
    - We utilise an sklearn compatible custom Regressor model **([model found here](https://github.com/shtrausslearning/Data-Science-Portfolio/blob/main/Heart%20Disease%20Classification/ml-models/src/mlmodels/gpr_bclassifier.py))** based on **Kriging**, which we turned in a classifier by simply setting the threshold to 0.5 (basically the **prediction** method in sklearn models). 
    - We also tried different ensembles of different models in order to improve the model accuracy even further.
														

</div>

---

**Thank you for reading!**

Any questions or comments about the posts below can be addressed to the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**
