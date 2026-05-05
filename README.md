---

# **PCA Module for Eigenfaces (Facial Recognition)**

## **Overview**

This module implements **Principal Component Analysis (PCA)** for facial recognition using the **Eigenfaces** approach. The objective is to extract the most significant patterns from high-dimensional facial image data and represent them in a reduced-dimensional subspace.

Eigenfaces correspond to the principal components of the dataset and capture the dominant variations across facial images. This representation enables efficient face projection, reconstruction, and recognition.

---

## **Responsibilities**

### **1. Data Loading**

* Load preprocessed and mean-centered training data:

  * `X_centered.npy` (shape: *N × D*)
* Load auxiliary data:

  * `mean_face.npy` (shape: *D*)
  * `L.npy` (reduced covariance matrix, shape: *N × N*)

---

### **2. Eigen Decomposition**

* Perform eigenvalue decomposition on the reduced covariance matrix.
* Extract eigenvalues and eigenvectors using:

  ```python
  np.linalg.eigh(L)
  ```
* Sort eigenvalues and corresponding eigenvectors in descending order based on explained variance.

---

### **3. Eigenfaces Computation**

* Map eigenvectors from reduced space to original image space:

  ```python
  eigenfaces = X_centered.T @ eigenvectors
  ```
* Normalize each eigenface:

  ```python
  eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])
  ```

---

### **4. Visualization**

* Display the top 10 eigenfaces as grayscale images.
* Reshape each eigenface from a vector of size *D* to *(100 × 100)*.
* Plot cumulative explained variance to evaluate dimensionality reduction.

---

### **5. Output Generation**

Save computed results for downstream processing:

* `eigenfaces.npy` (shape: *D × N*)
* `eigenvalues.npy` (shape: *N*)

These outputs are used in subsequent stages for:

* Face projection
* Image reconstruction
* Face recognition (e.g., via Euclidean distance)

---

## **Dataset Assumptions**

* Each image has a resolution of **100 × 100 pixels**.
* Images are:

  * Preprocessed
  * Flattened into vectors
  * Normalized within the range [0, 1]

---

## **Technical Details**

### **Reduced Covariance Trick**

To improve computational efficiency, PCA is applied using a reduced covariance matrix:

```
L = X_centered · X_centeredᵀ   (N × N)
```

This avoids computing the full covariance matrix of size *(D × D)*, where *D* is very large.

---

### **Eigenfaces Pipeline**

1. Compute eigenvalues and eigenvectors of `L`.
2. Project eigenvectors into the original feature space.
3. Normalize resulting eigenfaces.
4. Sort by importance (variance explained).

---

### **Variance Explained**

* Each eigenvalue represents the variance captured by its corresponding eigenface.
* The cumulative variance plot helps determine how many components are required to retain most of the dataset’s information.

---

## **Outputs**

### **Files**

* `eigenfaces.npy`
* `eigenvalues.npy`

### **Visualizations**

* Top 10 eigenfaces
* Cumulative explained variance plot

---

## **Integration**

This module provides outputs required for the next stage of the pipeline, including:

* Projection into eigenface space
* Image reconstruction
* Face recognition tasks

---

## **Dependencies**

* `numpy`
* `matplotlib`

---

## **Reference**

* Turk, M., & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience, 3(1), 71–86.

---
