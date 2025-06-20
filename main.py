import cv2
import numpy as np
import os

# -------- 1. Reduce Intensity Levels -------- #
def reduce_intensity_levels(image, levels):
    max_val = 256
    step = max_val // levels
    reduced_image = (image // step) * step
    return reduced_image.astype(np.uint8)

# -------- 2. Spatial Averaging Filter -------- #
def average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

# -------- 3. Rotate Image -------- #
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

# -------- 4. Block Averaging to Simulate Resolution Reduction -------- #
def block_average(image, block_size):
    h, w = image.shape[:2]
    new_image = image.copy()

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y+block_size, x:x+block_size]
            avg = np.mean(block, axis=(0, 1), dtype=int)
            new_image[y:y+block_size, x:x+block_size] = avg
    return new_image

# -------- Main Function -------- #
def main():
    # Create output folder
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    input_path = "bird.jpg"  # Change this to your image file name
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load image: {input_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Task 1: Intensity Reduction ----
    for levels in [2, 4, 8, 16, 32, 64, 128]:
        reduced = reduce_intensity_levels(gray, levels)
        cv2.imwrite(f"{output_dir}/intensity_{levels}_levels.png", reduced)

    # ---- Task 2: Spatial Averaging ----
    for k in [3, 10, 20]:
        blurred = average_filter(gray, k)
        cv2.imwrite(f"{output_dir}/average_{k}x{k}.png", blurred)

    # ---- Task 3: Rotation ----
    rotated_45 = rotate_image(img, 45)
    rotated_90 = rotate_image(img, 90)
    cv2.imwrite(f"{output_dir}/rotated_45.png", rotated_45)
    cv2.imwrite(f"{output_dir}/rotated_90.png", rotated_90)

    # ---- Task 4: Block Average for Spatial Resolution ----
    for block_size in [3, 5, 7]:
        downscaled = block_average(img, block_size)
        cv2.imwrite(f"{output_dir}/block_average_{block_size}x{block_size}.png", downscaled)

    print("All operations completed. Check the 'output_images' folder.")

# -------- Run the script -------- #
if __name__ == "__main__":
    main()
