import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
import time


# Change the dataset for your computer
dataset_path = os.path.join("dataset_for_project", "*.jpg")
image_paths = glob.glob(dataset_path)
print(f"Number of images found: {len(image_paths)}")

output_folder = "processed_images_opencv"
os.makedirs(output_folder, exist_ok=True)

if not image_paths:
    print("No images found. Check the dataset path.")
else:
    for img_path in image_paths:
        start_time = time.time()
        image = imageio.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grayscale_time=time.time() - start_time
        print(f"Grayscale Conversion Time: {grayscale_time:.4f} seconds")
        
        #Global Thresholding
        start_time = time.time()
        mean_intensity = np.mean(gray)
        _, global_thresh = cv2.threshold(gray, mean_intensity, 255, cv2.THRESH_BINARY)
        global_time = time.time() - start_time
        print(f"Global Thresholding Time: {global_time:.4f} seconds")

        #Otsu's Thresholding
        start_time = time.time()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_time = time.time() - start_time
        print(f"Otsu's Thresholding Time Time: {otsu_time:.4f} seconds")

        #Adaptive Thresholding (Gaussian)
        start_time = time.time()
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2
        )
        adaptive_time = time.time() - start_time
        print(f"Adaptive Thresholding Time: {adaptive_time:.4f} seconds")
        
      
        # Clustering Thresholding
        start_time = time.time()
        num_labels, labels = cv2.connectedComponents(global_thresh)
        cluster_image = np.zeros_like(gray)
        for label in range(1, num_labels):
            cluster_image[labels == label] = 255
        cluster_time =  time.time() - start_time    

        plt.figure(figsize=(15, 8), facecolor='#F0FFF0')

        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(gray, cmap="gray")
        plt.title("Grayscale Image")
        plt.axis("off")
        plt.text(0.5, -0.05, f"Time: {grayscale_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)
    
        plt.subplot(2, 3, 3)
        plt.imshow(global_thresh, cmap="gray")
        plt.title("Global Thresholding")
        plt.axis("off")
        plt.text(0.5, -0.05, f"Time: {global_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)
        

        plt.subplot(2, 3, 4)
        plt.imshow(otsu_thresh, cmap="gray")
        plt.title("Otsu's Thresholding")
        plt.axis("off")
        plt.text(0.5, -0.05, f"Time: {otsu_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)

        plt.subplot(2, 3, 5)
        plt.imshow(adaptive_thresh, cmap="gray")
        plt.title("Adaptive Thresholding")
        plt.axis("off")
        plt.text(0.5, -0.05, f"Time: {adaptive_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)
        

        plt.subplot(2, 3, 6)
        plt.imshow(cluster_image, cmap="gray")
        plt.title("Clustering Threshold")
        plt.axis("off")
        plt.text(0.5, -0.05, f"Time: {cluster_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)
       


        output_path = os.path.join(output_folder, f"opencv_threshold_{os.path.basename(img_path)}.jpg")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

        plt.suptitle(f"Thresholding Results for {os.path.basename(img_path)}")
        plt.show()