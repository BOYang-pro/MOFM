import numpy as np
import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Test on dataset
def f_sobel(img):
    dst_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dst_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    dst=np.sqrt(dst_y ** 2 + dst_x ** 2)
    return dst

def combine_show(gt,warp_image):
    gradient_magnitude=f_sobel(warp_image/255)
    colors  = ['#000000', '#330033', '#660066', '#990099', '#CC00CC', '#E6E6FA']
    
    cmap_custom = LinearSegmentedColormap.from_list('light_purple', colors, N=256)

    colormap2 = cmap_custom  # colormap2

    img2_color = colormap2(gradient_magnitude)[:, :, :3]  # plasma

    combined_img = np.clip(gt + img2_color*255, 0, 255).astype(np.uint8)
    return combined_img
