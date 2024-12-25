# Image Thresholding Project

This project compares different thresholding methods in image processing using Python. Two scripts are included:

1. **`custom_threshold.py`**: Implements custom thresholding and clustering-based techniques.
2. **`opencv_threshold.py`**: Uses OpenCV's built-in thresholding methods.

---



## Contributions

- This project was completed as part of a collaborative effort by our student group. 
Each member contributed to different aspects of the project:


**İsmail Atalay**
-Authored the project report in LaTeX and designed the presentation slides.



**Fatmanur Köse**
-Enhanced the scripts by adding timing functionalities for performance analysis.
-Created questions for understanding of the project. 


**Sema Nur Yılmaz**
-Developed the main functionality in custom_threshold.py and opencv_threshold.py, implementing global, Otsu's, and adaptive thresholding methods. Gathered the data.



**Muhammed Talha Şatır** 
-Contributed additional questions


**Adnan Arda Simsar**
-Added clustering functionality to the scripts, enhancing the thresholding techniques with KMeans and connected components.


## Methods

### 1. Global Thresholding
- Fixed intensity threshold to separate foreground and background.

### 2. Otsu's Thresholding
- Automatically calculates the best threshold value.

### 3. Adaptive Thresholding
- Computes a local threshold for each pixel using its surrounding region.

### 4. Clustering Thresholding
- Groups pixel intensities into clusters and thresholds based on the brightest cluster.

---

## How to Run

### Prerequisites
Install the required Python libraries:
pip install numpy matplotlib scikit-learn opencv-python
