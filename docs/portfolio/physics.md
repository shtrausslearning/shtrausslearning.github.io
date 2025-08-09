---
comments: true
---

**Machine learning** plays a significant role in physics-related applications by enabling the analysis, prediction, and modeling of complex physical phenomena. Here is an example of how machine learning is used in physics:

Data analysis and pattern recognition: Machine learning algorithms can be used to analyze large datasets generated from experiments or simulations. They can identify patterns, correlations, and anomalies that may not be apparent to human researchers, helping to gain insights into physical processes. 

<br>

<div class="grid cards" markdown >

  - ## :octicons-bookmark-fill-24:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[CFD Trade-Off Study Visualisation | Response Model](https://www.kaggle.com/code/shtrausslearning/cfd-trade-off-study-visualisation-response-model)</b>

  	:simple-github:{ .lg .middle }&nbsp; <b>[Github Repository](https://github.com/shtrausslearning/mllibs/blob/main/src/mlmodels/kriging_regressor.py)</b>

	---

    In this study, we do an **exploratory data analysis** of a **CFD optimisation study**, having extracted table data for different variables in a simulation, we aim to find the most optimal design using different visualisation techniques. 
    
    - The data is then utilised to create a response model for **L/D** (predict L/D based on other parameters), we investigate which machine learning models work the best for this problem
														

</div>


<div class="grid cards" markdown >

  - ## :octicons-bookmark-fill-24:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Gaussian Processes | Airfoil Noise Modeling](https://www.kaggle.com/code/shtrausslearning/gaussian-processes-airfoil-noise-modeling)</b>

	---

    In this study, we do an exploratory data analysis of [experimental measurement data](https://doi.org/10.24432/C5VW2C) associated with NACA0012 airfoil noise measurements. 
    
    - We outline the dependencies of parameter and setting variation and its influence on SPL noise level. 
    - The data is then used to create a machine learning model, which is able to predict the sound pressure level (SPL) for different combinations of airfoil design parameters
														

</div>

<div class="grid cards" markdown >

  - ## :octicons-bookmark-fill-24:{ .hover-icon-bounce .success-hover title="Jan,2024" } <b>[Spectogram Broadband Model & Peak Identifier](https://www.kaggle.com/code/shtrausslearning/spectogram-broadband-model-peak-identifier)</b>

    ---

    Noise generally can be divided into **broadband noise** (general noise level) & **tonal noises** (peaks at specific frequency bins). 
    
    - They don't have precise definitions, but broadband noises can be abstractly defined as the general noise level in an environement coming from various locations, creating a broad frequency range noise relation to output noise level. 
    - Tonal noise sources tend be associated to very clearly distinguishible noise peaks at specific frequencies ( or over a small frequency range ). 
    - When we look at a spectogram, each bird specie tends to create quite a repetitive collection of freq vs time structures, usually across a specific frequency range, usually it's a combination of tonal peaks that make up an entire bird call. In this approach, the two terms are used even looser, since there is a time element to this model from the STFT, which can be useful in a variety of scenarios. 
    - The **tonal peak frequency identification approach** relies on the assumption that the more data is fed into the system, the more precise the result should get, as occasional secondary birds & other noises should eventually start to show more dissipative distribution in the entire subset that is analysed.

    Looping over all desired audio files of a subset of interest to us (a particular primary label subset):

    - First, we load an audio recording that we wish to convert to desired to be used as inputs for CNN models.
    - The audio is then split into segments that will define the spectogram time domain limits. Usually we would start with the entire frequency range [0,12.5kHz] and split the recording into a 5 second chunks, creating a time & frequency domain relation.
    - For reference, we find the maximum dB value in the entire frequency range, this model will define the peaks of the tonal noises and will always be the maximum.
    - The spectogram is then divided into time bins, cfg.model_bins & for each time bin, the maximum value for each frequency is determined.
    - A model (**kriging**) for each time bin is created and a simple enemble of all time segments is constructed, this should always create a model that is lower in dB level than the global peak model mentioned earlier. There are certain cases where this is not the case, usually an indicator that there exist an anomaly in the structure of the curve (as shown in the example below).
    - The peaks of the model are then found using scipy's find_peaks module, stored into a global list & the Counter module counts all list entries.
    - The results are subsequently plotted for each pixel value. The corresponding frequency values can be extracted using the function pxtohz.


</div>


---

**Thank you for reading!**

Any questions or comments about the above post can be addressed on the :fontawesome-brands-telegram:{ .telegram } **[mldsai-info channel](https://t.me/mldsai_info)** or to me directly :fontawesome-brands-telegram:{ .telegram } **[shtrauss2](https://t.me/shtrauss2)**, on :fontawesome-brands-github:{ .github } **[shtrausslearning](https://github.com/shtrausslearning)** or :fontawesome-brands-kaggle:{ .kaggle} **[shtrausslearning](https://kaggle.com/shtrausslearning)**

