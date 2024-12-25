import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time

def global_threshold(image, threshold):
    return np.where(image > threshold, 255, 0).astype(np.uint8)

def otsu_threshold(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total_pixels = image.size
    current_max, threshold = 0, 0
    sum_total, sum_background, weight_background = 0, 0, 0

    for i in range(256):
        sum_total += i * hist[i]
    
    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i

    return global_threshold(image, threshold)

def adaptive_threshold(image, block_size, C):
    result = np.zeros_like(image)
    pad_size = block_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            block = padded_image[i:i + block_size, j:j + block_size]
            local_thresh = np.mean(block) - C
            result[i, j] = 255 if image[i, j] > local_thresh else 0

    return result.astype(np.uint8)

# Main code
dataset_path = os.path.join("dataset_for_project", "*.jpg")
image_paths = glob.glob(dataset_path)
print(f"Number of images found: {len(image_paths)}")

print(f"Number of images found: {len(image_paths)}")
output_folder = "processed_images"
os.makedirs(output_folder, exist_ok=True)

for img_path in image_paths:
    image = imageio.imread(img_path)
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale_time=time.time() - start_time
    print(f"Grayscale Conversion Time: {grayscale_time:.4f} seconds")
    
    start_time = time.time()
    global_thresh = global_threshold(gray, 127)
    global_time = time.time() - start_time
    print(f"Global Thresholding Time: {global_time:.4f} seconds")
    
    start_time = time.time()
    otsu_thresh = otsu_threshold(gray)
    otsu_time = time.time() - start_time
    print(f"Otsu's Thresholding Time Time: {otsu_time:.4f} seconds")
    
    start_time = time.time()
    adaptive_thresh = adaptive_threshold(gray, 33, 2)
    adaptive_time = time.time() - start_time
    print(f"Adaptive Thresholding Time: {adaptive_time:.4f} seconds")
    
    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 5, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")
    plt.text(0.5, -0.1, f"Time: {grayscale_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes)
    
    plt.subplot(1, 5, 3)
    plt.imshow(global_thresh, cmap="gray")
    plt.title("Global Thresholding")
    plt.axis("off")
    plt.text(0.5, -0.1, f"Time: {global_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes)
     
    plt.subplot(1, 5, 4)
    plt.imshow(otsu_thresh, cmap="gray")
    plt.title("Otsu's Thresholding")
    plt.axis("off")
    plt.text(0.5, -0.1, f"Time: {otsu_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes)
    
    plt.subplot(1, 5, 5)
    plt.imshow(adaptive_thresh, cmap="gray")
    plt.title("Adaptive Thresholding")
    plt.axis("off")
    plt.text(0.5, -0.1, f"Time: {adaptive_time:.4f}s", ha='center', va='top', transform=plt.gca().transAxes)
    # Save the plot to the output folder
    output_path = os.path.join(output_folder, f"threshold_{os.path.basename(img_path)}.jpg")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.suptitle(f"Thresholding Results for {os.path.basename(img_path)}")
    
    plt.tight_layout()
    plt.show()
