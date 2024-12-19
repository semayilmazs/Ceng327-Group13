# Ceng327-Group13

This project demonstrates different thresholding techniques for image segmentation using Python and OpenCV.

1. **Global Thresholding**  
   Applies a fixed threshold value to segment the image.

2. **Otsu's Thresholding**  
   Automatically determines the optimal threshold by minimizing intra-class variance.

3. **Adaptive Thresholding**  
   Computes thresholds locally for different regions of the image, suitable for varying lighting conditions.

# Different Implementations on The Same Topic
  
   One of the implementations uses OpenCv built in thresholding functions but the other one contains totally customized thresholding methods.
# The following steps walk through getting the application running. 


1. **Clone the repository**




         git clone https://github.com/semayilmazs/Ceng327-Group13.git

3. **Put the images in the correct folder**


   Place your images in the dataset_for_project folder. Ensure the images are in supported formats like .png or .jpg.

4. **Required Libraries**


   Make sure the following libraries are installed
 
       pip install matplotlib imageio opencv-python-headless numpy


5. **Run the script**
   python opencv_thresholding.py
   python custom_thresholding.py
