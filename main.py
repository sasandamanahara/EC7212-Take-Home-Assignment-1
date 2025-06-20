import cv2
import numpy as np
import os

# -------- Utility: Create directory -------- #
def create_dir(path):
    os.makedirs(path, exist_ok=True)

# -------- Task 1: Reduce Intensity Levels -------- #
def reduce_intensity_levels(image, levels):
    max_val = 256
    step = max_val // levels
    reduced_image = (image // step) * step
    return reduced_image.astype(np.uint8)

# -------- Task 2: Spatial Averaging Filter -------- #
def average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

# -------- Task 3: Rotate Image -------- #
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - (w / 2)
    M[1, 2] += (new_h / 2) - (h / 2)

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)
    return rotated

# -------- Task 4: Block Averaging to Reduce Resolution -------- #
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
    # Paths
    input_path ="bird.jpg"
    base_output_dir = "output_images"

    # Create task-specific folders
    task_dirs = {
        "task1": os.path.join(base_output_dir, "task1"),
        "task2": os.path.join(base_output_dir, "task2"),
        "task3": os.path.join(base_output_dir, "task3"),
        "task4": os.path.join(base_output_dir, "task4")
    }
    for path in task_dirs.values():
        create_dir(path)

    # Load image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load image: {input_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------- Task 1: Intensity Reduction ----------
    for levels in [2, 4, 8, 16, 32, 64, 128]:
        reduced = reduce_intensity_levels(gray, levels)
        out_path = os.path.join(task_dirs["task1"], f"intensity_{levels}_levels.png")
        cv2.imwrite(out_path, reduced)

    # ---------- Task 2: Spatial Averaging ----------
    for k in [3, 10, 20]:
        blurred = average_filter(gray, k)
        out_path = os.path.join(task_dirs["task2"], f"average_{k}x{k}.png")
        cv2.imwrite(out_path, blurred)

    # ---------- Task 3: Rotation ----------
    rotated_45 = rotate_image(img, 45)
    rotated_90 = rotate_image(img, 90)
    cv2.imwrite(os.path.join(task_dirs["task3"], "rotated_45.png"), rotated_45)
    cv2.imwrite(os.path.join(task_dirs["task3"], "rotated_90.png"), rotated_90)

    # ---------- Task 4: Block Averaging ----------
    for block_size in [3, 5, 7]:
        downscaled = block_average(img, block_size)
        out_path = os.path.join(task_dirs["task4"], f"block_average_{block_size}x{block_size}.png")
        cv2.imwrite(out_path, downscaled)

    print("✅ All tasks completed. Check the 'output_images' folder.")

# -------- Run the script -------- #
if __name__ == "__main__":
    main()
