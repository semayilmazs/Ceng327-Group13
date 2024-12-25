# Image Thresholding Project

This project compares different thresholding methods in image processing using Python. Two scripts are included:

1. **`custom_threshold.py`**: Implements custom thresholding and clustering-based techniques.
2. **`opencv_threshold.py`**: Uses OpenCV's built-in thresholding methods.

---

## Methods
S
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
