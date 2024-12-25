import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np



# Change the dataset for your computer
dataset_path = os.path.join("dataset_for_project", "*.jpg")
image_paths = glob.glob(dataset_path)
print(f"Number of images found: {len(image_paths)}")

if not image_paths:
    print("No images found. Check the dataset path.")
else:
    for img_path in image_paths:
        image = imageio.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        #Global Thresholding
        mean_intensity = np.mean(gray)
        _, global_thresh = cv2.threshold(gray, mean_intensity, 255, cv2.THRESH_BINARY)

        #Otsu's Thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #Adaptive Thresholding (Gaussian)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2
        )

        plt.figure(figsize=(12, 6), facecolor='#F0FFF0')

        plt.subplot(1, 5, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 5, 2)
        plt.imshow(gray, cmap="gray")
        plt.title("Grayscale Image")
        plt.axis("off")

        plt.subplot(1, 5, 3)
        plt.imshow(global_thresh, cmap="gray")
        plt.title("Global Thresholding")
        plt.axis("off")

        plt.subplot(1, 5, 4)
        plt.imshow(otsu_thresh, cmap="gray")
        plt.title("Otsu's Thresholding")
        plt.axis("off")

        plt.subplot(1, 5, 5)
        plt.imshow(adaptive_thresh, cmap="gray")
        plt.title("Adaptive Thresholding")
        plt.axis("off")

        plt.suptitle(f"Thresholding Results for {os.path.basename(img_path)}")
        plt.show()
