import cv2 as cv
import numpy as np
import sys, os
from tqdm import tqdm

#setting recursion limit to 1B for the system to not to stop to early.
sys.setrecursionlimit(int(1e9))

THRESHOLD = 0
SPLIT_CONDITION = lambda r, g, b: (r < THRESHOLD) or (g< THRESHOLD) or (b < THRESHOLD)

def segmentation(image, x1, x2, y1, y2):
    if x2 - x1 + 1 <= 2 or y2 - y1 + 1 <= 2:
        return image
    
    # Separating the channels.
    blue_ch = image[int(x1):int(x2) + 1, int(y1):int(y2) + 1, 0]
    green_ch = image[int(x1):int(x2) + 1, int(y1):int(y2) + 1, 1]
    red_ch = image[int(x1):int(x2) + 1, int(y1):int(y2) + 1, 2]

    bl_std = np.std(blue_ch)
    gr_std = np.std(green_ch)
    rd_std = np.std(red_ch)

    if SPLIT_CONDITION(rd_std, gr_std, bl_std):
        rd_mean = np.mean(red_ch)
        gr_mean = np.mean(green_ch)
        bl_mean = np.mean(blue_ch)

        # Replace image.
        image[int(x1):int(x2) + 1, int(y1):int(y2) + 1, 0] = np.full_like(blue_ch, bl_mean)  # blue channel
        image[int(x1):int(x2) + 1, int(y1):int(y2) + 1, 1] = np.full_like(green_ch, gr_mean)  # green channel
        image[int(x1):int(x2) + 1, int(y1):int(y2) + 1, 2] = np.full_like(red_ch, rd_mean)  # red channel
        
        return image
    
    #split into 4 parts otherwise.
    else:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        image = segmentation(image, x1, mid_x, y1, mid_y)
        image = segmentation(image, mid_x, x2, y1, mid_y) 
        image = segmentation(image, x1, mid_x, mid_y, y2) 
        image = segmentation(image, mid_x, x2, mid_y, y2)

        return image



if __name__ == "__main__":
    IMAGE_NAME = "lena.jpg"

    image = cv.imread(IMAGE_NAME)
    h, w, _ = image.shape

    x1, x2 = 0, w
    y1, y2 = 0, h


    #setting threshold. If -1, loop till thres_range
    threshold_value = int(input("Threshold value: "))
    THRESHOLD = threshold_value

    if THRESHOLD == -1:
        os.makedirs("out", exist_ok=True)
        th_range = int(input("Threshold value range: "))
        th_step_size = int(input("Threshold segmentation step size: "))
        for th_val in tqdm(range(th_range + 1)):
            if th_val % th_step_size == 0:
                THRESHOLD = th_val
                output_image = segmentation(image, 0, w, 0, w)
                cv.imwrite(f"out/thres_{THRESHOLD}.jpg", output_image)

    else:
        output_image = segmentation(image, 0, w, 0, w)
        cv.imwrite(f"out/thres_{THRESHOLD}.jpg", output_image)