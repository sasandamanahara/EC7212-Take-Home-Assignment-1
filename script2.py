## ---------- 1. Reduce the number of intensity levels ----------

import cv2
import numpy as np
import os


def reduce_intensity_levels(img, levels):
    factor = 256 // levels
    return (img // factor) * factor


def task1_reduce_levels(image_path, levels):
    # Validate if levels is a power of 2
    if levels < 2 or levels > 256 or (levels & (levels - 1)) != 0:
        raise ValueError("Levels must be a power of 2 between 2 and 256.")

    # Make output directory
    os.makedirs("task1_output", exist_ok=True)

    # Load grayscale and color images
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(image_path)

    if gray is None or color is None:
        print("Error: Image not found.")
        return

    # Apply reduction
    reduced_gray = reduce_intensity_levels(gray, levels)
    reduced_color = reduce_intensity_levels(color, levels)

    # Save output images
    cv2.imwrite(f"task1_output/original_gray.jpg", gray)
    cv2.imwrite(f"task1_output/original_color.jpg", color)
    cv2.imwrite(f"task1_output/gray_reduced_{levels}.jpg", reduced_gray)
    cv2.imwrite(f"task1_output/color_reduced_{levels}.jpg", reduced_color)

    print(f"Task 1 done. Output saved in 'task1_output/' folder.")

    # Display the original and quantized image
    cv2.imshow("Original Gray Image", gray)
    cv2.imshow(f"Quantized Gray to {levels} levels", reduced_gray)
    cv2.imshow("Original Color Image", color)
    cv2.imshow(f"Quantized Color to {levels} levels", reduced_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the task
image_path = 'bird.jpg'
desired_levels = 4
task1_reduce_levels(image_path, desired_levels)

# import cv2
# import numpy as np

# def reduce_intensity_levels(image_path, levels):
#     # Validate if levels is a power of 2
#     if levels < 2 or levels > 256 or (levels & (levels - 1)) != 0:
#         raise ValueError("Levels must be a power of 2 between 2 and 256.")

#     # Read the image in grayscale
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Calculate the quantization factor
#     factor = 256 // levels

#     # Quantize the image
#     quantized_img = (img // factor) * factor

#     # Display the original and quantized image
#     cv2.imshow("Original Image", img)
#     cv2.imshow(f"Quantized to {levels} levels", quantized_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# image_path = 'lena.jpg'
# desired_levels = 4          # Change this to 2, 4, 8, 16, ..., 256
# reduce_intensity_levels(image_path, desired_levels)


## ---------- 2. Average Blur with 3x3, 10x10, 20x20 ----------

# Make output directory
os.makedirs("task2_output", exist_ok=True)

# Load grayscale image
image = cv2.imread('lena.jpg')

# Apply 3x3 average filter
blur_3 = cv2.blur(image, (3, 3))

# Apply 10x10 average filter
blur_10 = cv2.blur(image, (10, 10))

# Apply 20x20 average filter
blur_20 = cv2.blur(image, (20, 20))

# Save output images
cv2.imwrite(f"task2_output/original.jpg", image)
cv2.imwrite(f"task2_output/3x3_blur.jpg", blur_3)
cv2.imwrite(f"task2_output/10x10_blur.jpg", blur_10)
cv2.imwrite(f"task2_output/20x20_blur.jpg", blur_20)

# Show results
cv2.imshow("Original", image)
cv2.imshow("3x3 Blur", blur_3)
cv2.imshow("10x10 Blur", blur_10)
cv2.imshow("20x20 Blur", blur_20)
cv2.waitKey(0)
cv2.destroyAllWindows()

## ---------- 3. Rotate Image by 45 and 90 Degrees ----------

import cv2

# Load image
image = cv2.imread('bird.jpg')

# Get image center
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# Rotate 45 degrees
M_45 = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated_45 = cv2.warpAffine(image, M_45, (w, h))

# Rotate 90 degrees (can also use cv2.rotate for exact 90°)
rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Show images
cv2.imshow("Original", image)
cv2.imshow("Rotated 45 Degrees", rotated_45)
cv2.imshow("Rotated 90 Degrees", rotated_90)
cv2.waitKey(0)
cv2.destroyAllWindows()

## ---------- 4. Block Averaging Function ----------

import cv2
import numpy as np


def block_average(image, block_size):
    (h, w) = image.shape
    output = np.zeros_like(image)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y + block_size, x:x + block_size]
            avg = np.mean(block, dtype=np.uint8)
            output[y:y + block_size, x:x + block_size] = avg

    return output


# Load image in grayscale
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Apply block averaging
avg_3 = block_average(image, 3)
avg_5 = block_average(image, 5)
avg_7 = block_average(image, 7)

# Show results
cv2.imshow("Original", image)
cv2.imshow("3x3 Block Averaging", avg_3)
cv2.imshow("5x5 Block Averaging", avg_5)
cv2.imshow("7x7 Block Averaging", avg_7)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("blur_3x3.jpg", blur_3)
cv2.imwrite("blur_10x10.jpg", blur_10)
cv2.imwrite("blur_20x20.jpg", blur_20)

cv2.imwrite("rotated_45.jpg", rotated_45)
cv2.imwrite("rotated_90.jpg", rotated_90)

cv2.imwrite("block_avg_3x3.jpg", avg_3)
cv2.imwrite("block_avg_5x5.jpg", avg_5)
cv2.imwrite("block_avg_7x7.jpg", avg_7)

# import cv2
# import numpy as np

# # Load image in grayscale
# image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# if image is None:
#     print("Error: Image not found. Make sure 'lena.jpg' is in the same folder.")
#     exit()

# # ---------- 1. Average Blur with 3x3, 10x10, 20x20 ----------
# blur_3 = cv2.blur(image, (3, 3))
# blur_10 = cv2.blur(image, (10, 10))
# blur_20 = cv2.blur(image, (20, 20))

# cv2.imwrite("blur_3x3.jpg", blur_3)
# cv2.imwrite("blur_10x10.jpg", blur_10)
# cv2.imwrite("blur_20x20.jpg", blur_20)

# # ---------- 2. Rotate Image by 45 and 90 Degrees ----------
# (h, w) = image.shape
# center = (w // 2, h // 2)

# # Rotate 45 degrees
# M_45 = cv2.getRotationMatrix2D(center, 45, 1.0)
# rotated_45 = cv2.warpAffine(image, M_45, (w, h))

# # Rotate 90 degrees
# rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# cv2.imwrite("rotated_45.jpg", rotated_45)
# cv2.imwrite("rotated_90.jpg", rotated_90)

# # ---------- 3. Block Averaging Function ----------
# def block_average(img, block_size):
#     (h, w) = img.shape
#     output = np.copy(img)
#     for y in range(0, h - block_size + 1, block_size):
#         for x in range(0, w - block_size + 1, block_size):
#             block = img[y:y+block_size, x:x+block_size]
#             avg_val = int(np.mean(block))
#             output[y:y+block_size, x:x+block_size] = avg_val
#     return output

# # Apply block averaging
# avg_3 = block_average(image, 3)
# avg_5 = block_average(image, 5)
# avg_7 = block_average(image, 7)

# cv2.imwrite("block_avg_3x3.jpg", avg_3)
# cv2.imwrite("block_avg_5x5.jpg", avg_5)
# cv2.imwrite("block_avg_7x7.jpg", avg_7)

# # ---------- Optional: Show all results ----------
# cv2.imshow("Original", image)
# cv2.imshow("Blur 3x3", blur_3)
# cv2.imshow("Rotated 45", rotated_45)
# cv2.imshow("Block Avg 3x3", avg_3)

# cv2.waitKey(0)
# cv2.destroyAllWindows()