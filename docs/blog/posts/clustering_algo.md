---
date:
  created: 2024-02-06
  updated: 2024-02-13
description: |
  Overview of clustering algorithms in Machine Learning.
categories:
  - ml
slug: clustering-algo
title: Clustering Algorithms in ML
draft: true
---

# :material-chart-scatter-plot:{ title="2024-02-06" } Clustering Algorithms in ML

## Applications of Clustering

- **Customer Segmentation**: To show personalized ADs to customers.
- **Data Analysis**: Perform analysis to each cluster after performing clustering on the whole dataset.
- **Semi Supervised Learning**: Google Photos uses this technique to identify person's face and put them into a separate folder.
- **Image Segmentation**: You can create segments in photos to represent different objects in the photo.

## :material-scatter-plot:  KMeans

<!-- more -->

### Working Steps

1. You tell the algorithm how many clusters are there in the data (this is assumption which is taken before initialization).
2. The cluster's centroids are initialize with some random values.
3. The distance of each data point is calculated with each cluster and then the data points are assigned to nearest clusters.
4. After data points assignment to clusters, the centroid of the clusters is being calculated.
5. If the centroid's position is:
    - **Same as before** then the algorithm stops.
    - **Not same as before** then the **STEP 3** is being re-calculated and the process goes on.

??? example "KMeans in Diagram"

    ```mermaid
    graph TD
      subgraph Initialization
        A(Initialize Centroids)
        B[Calculate Distance and \n Assign Data Points]
      end

      subgraph Iterative Process
        C[Update Centroids]
        D[Reassign Data Points]
      end

      subgraph Convergence Check
        E[/Converged?/]
      end

      A -->|Random Initialization| B
      B -->|Assign Nearest Centroid| C
      C -->|Recalculate Centroids| D
      D -->|Assign New Clusters| E
      E -->|Yes| stop{Stop}
      E -->|No| B
    ```

!!! info "Some important terms!"

    - **Inertia** also known as **within-cluster sum of squares** (WCSS) in the context of K-Means clustering, is a measure that quantifies the compactness of clusters. It is calculated as the sum of the squared distances between each data point in a cluster and the centroid of that cluster.
    - **Elbow Method** is way to decide the number of clusters present in a data. However, this is not a very good method to estimate clusters _but it there to help you for that_.

### Assumptions of KMeans

- **Spherical Cluster Shape**: K-means assumes that the clusters are spherical and isotropic, meaning they are uniform in all directions. Consequently, the algorithm works best when the actual clusters in the data are circular (in 2D) or spherical (in higher dimensions).
- **Similar Cluster Size**: The algorithm tends to perform better when all clusters are of approximately the same size. If one cluster is much larger than others, K-means might struggle to correctly assign the points to the appropriate cluster.
- **Equal Variance of Clusters**: K-means assumes that all clusters have similar variance. The algorithm uses the Euclidean distance metric, which can bias the clustering towards clusters with lower variance.
- **Clusters are Well Separated**: The algorithm works best when the clusters are well separated from each other. If clusters are overlapping or intertwined, K-means might not be able to distinguish them effectively.
- **Number of Clusters (k) is Predefined**: K-means requires the number of clusters (k) to be specified in advance. Choosing the right value of k is crucial, but it is not always straightforward and typically requires domain knowledge or additional methods like the Elbow method or Silhouette analysis.
- **Large n, Small k**: K-means is generally more efficient and effective when the dataset is large (large n) and the number of clusters is small (small k).

### Resources to Learn From

??? abstract "Abstract &nbsp; :robot:{ .bounce }"

    **Introduction**

    - **Objective:** Divide data into k clusters.
    - **Algorithm:** Iterative process that minimizes the sum of squared distances between data points and their assigned cluster centroids.

    **Algorithm Steps**

    - **Initialization:** Randomly select k initial centroids.
    - **Assignment:** Assign each data point to the nearest centroid.
    - **Update Centroids:** Recalculate centroids based on assigned points.
    - **Repeat:** Iteratively repeat assignment and centroid update until convergence.

    **Advantages**

    - Simple and computationally efficient.
    - Scales well to large datasets.

    **Disadvantages**

    - Sensitive to initial centroid selection.
    - Assumes clusters are spherical and equally sized.

## :material-chart-scatter-plot-hexbin: DBSCAN

!!! quote "FULL FORM"

    Density-Based Spatial Clustering of Applications with Noise

DBSCAN is a clustering algorithm commonly used in data mining and machine learning. It's particularly effective in identifying clusters of arbitrary shapes and handling noise in the data.</br>
DBSCAN does not require the number of clusters _(like KMeans)_ to be specified beforehand and can discover clusters based on the density of data points.

### Important Terms

- **Core Points:** A data point is considered a core point if there are at least **"min_samples"**{ title="HyperParameter" } data points (including itself) within a specified distance, usually denoted as **"epsilon" (ε)**{ title="HyperParameter" }. Core points are the central points of clusters.
- **Border Points:** A data point is considered a border point if it is within the specified distance **(ε)**{ title="HyperParameter" } of a core point but doesn't have enough neighboring points to be considered a core point itself. Border points are part of a cluster but are not central to it.
- **Noise Points:** Data points that are neither core points nor border points are considered noise points or outliers. These points do not belong to any cluster.

### Working Steps

1. **Initialization:** Select an arbitrary data point that has not been visited.
2. **Density Query:** Find all data points within distance ε from the selected point.
3. **Core Point Check:** If the number of points found is greater than or equal to **"min_samples"**{ title="HyperParameter" }, mark the selected point as a core point, and a cluster is formed.
4. **Expand Cluster:** Expand the cluster by recursively repeating the process for all the newly found core points.
5. **Next Point Selection:** Choose a new unvisited point and repeat the process until all data points have been visited.

### Resources to Learn From

??? tip "CampusX &nbsp; :simple-youtube:{ .youtube .bounce }"
    <figure>
        <iframe width="700" height="400" src="https://www.youtube-nocookie.com/embed/1_bLnsNmhCI?si=B6ym1DnPcFv5bnIG" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </figure>

??? tip "StatQuest &nbsp; :simple-youtube:{ .youtube .bounce }"
    <figure>
        <iframe width="700" height="400" src="https://www.youtube-nocookie.com/embed/RDZUdRSDOok?si=ynfnAEx7vdjGhCa5" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </figure>

??? abstract "Abstract &nbsp; :robot:{ .bounce }"

    **Introduction**

    - **Objective:** Identify clusters based on dense regions in data space.
    - **Algorithm:** Utilizes density information, considering data points as core, border, or noise.

    **Algorithm Steps**

    - **Density Estimation:** Define ε (eps) and minimum points for a neighborhood.
    - **Core Points:** Identify dense regions with at least 'minPts' neighbors within ε.
    - **Expand Clusters:** Connect core points and expand clusters with border points.

    **Advantages**

    - Doesn't assume spherical clusters.
    - Can find clusters of arbitrary shapes.
    - Robust to outliers.

    **Disadvantages**

    - Sensitivity to parameter settings (ε, minPts).
    - Struggles with varying density clusters.

## :material-dots-hexagon: Gaussian Mixture Models

Gaussian Mixture Models (GMMs) are probabilistic models used for clustering and density estimation. Unlike k-means, which assigns data points to a single cluster, GMMs allow each data point to belong to multiple clusters with different probabilities. GMMs assume that the data is generated from a mixture of several Gaussian distributions.

Here are the key concepts associated with Gaussian Mixture Models:

1. **Gaussian Distribution (Normal Distribution)**: A probability distribution that is characterized by its mean (μ) and standard deviation (σ). The probability density function of a Gaussian distribution is given by:

$$ f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) $$

2. **Mixture Model**: A combination of multiple probability distributions. In the case of GMMs, these are Gaussian distributions.

3. **Parameters of GMM**:

    - **Weights $(πi)$**: The probabilities associated with each component (Gaussian distribution). They represent the likelihood of a data point belonging to a specific cluster.
    - **Means $(μi)$**: The mean values of the Gaussian distributions.
    - **Covariance Matrices $(Σi)$**: The covariance matrices representing the shape and orientation of the Gaussian distributions.

4. **Probability Density Function of GMM**:

$$ P(x) = \sum_{i=1}^{k} \pi_i \cdot \mathcal{N}(x; \mu_i, \Sigma_i) $$

   where $\mathcal{N}(x; \mu_i, \Sigma_i)$ is the probability density function of the $i^{th}$ Gaussian distribution.

### Working Steps

1. **Initialization**: Initialize the parameters of the model, including the weights, means, and covariance matrices.

2. **Expectation-Maximization (EM) Algorithm**:
    - **Expectation Step (E-step)**: Compute the probabilities that each data point belongs to each cluster (responsibility).
    - **Maximization Step (M-step)**: Update the model parameters (weights, means, covariance matrices) based on the assigned responsibilities.

3. **Convergence**: Repeat the E-step and M-step until the model converges, i.e., the parameters stabilize.

4. **Prediction**: Once trained, the model can be used to predict the cluster assignments or estimate the density of new data points.

GMMs are flexible and can model complex data distributions. They are widely used in various applications, such as image segmentation, speech recognition, and anomaly detection.

### Resources to Learn From

??? tip "Serrano Academy &nbsp; :simple-youtube:{ .bounce .youtube }"
    <figure>
        <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/q71Niz856KE?si=Zz23kSbbfnmRQPHK" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </figure>

??? abstract "Abstract &nbsp; :robot:{ .bounce }"

    **Introduction**

    - **Objective:** Model data as a mixture of Gaussian distributions.
    - **Algorithm:** Probability-based approach using the Expectation-Maximization (EM) algorithm.

    **Algorithm Steps**

    - **Initialization:** Assign initial parameters for Gaussian distributions.
    - **Expectation Step:** Estimate probability of each data point belonging to each cluster.
    - **Maximization Step:** Update parameters based on the expected assignments.
    - **Repeat:** Iteratively repeat the E-M steps until convergence.

    **Advantages**

    - More flexible in capturing different cluster shapes.
    - Provides probabilistic cluster assignments.

    **Disadvantages**

    - Sensitive to initialization.
    - Computationally more expensive than K-Means.
