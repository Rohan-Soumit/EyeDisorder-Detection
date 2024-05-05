import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

def analyze_image(image_path, darkness_threshold=10):
    def preprocess_image(image_path):
        image = Image.open(image_path).convert('L')
        preprocessed_image = image.filter(ImageFilter.MedianFilter(size=3))
        return preprocessed_image

    def find_dark_and_bright_regions(np_image, darkness_threshold=10, focus_central=True):
        if focus_central:
            h, w = np_image.shape
            central_region = np_image[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        else:
            central_region = np_image

        darkness_threshold_adjusted = np.max([darkness_threshold, np.max(central_region)*0.1])  
        central_region_adjusted = np.where(central_region <= darkness_threshold_adjusted, np.nan, central_region)

        if not np.isnan(central_region_adjusted).all():
            darkest_idx = np.nanargmin(central_region_adjusted)
            darkest_point_position = np.unravel_index(darkest_idx, central_region.shape)
            darkest_point_position = (darkest_point_position[0] + int(h*0.25), darkest_point_position[1] + int(w*0.25))
        else:
            darkest_point_position = None

        if not np.isnan(np_image).all():
            brightest_idx = np.nanargmax(np_image)
            brightest_point_position = np.unravel_index(brightest_idx, np_image.shape)
        else:
            brightest_point_position = None

        return darkest_point_position, brightest_point_position

    preprocessed_image = preprocess_image(image_path)
    np_image = np.array(preprocessed_image, dtype=float)
    np_image[np_image == 0] = np.nan
    
    darkest_point, brightest_point = find_dark_and_bright_regions(np_image, darkness_threshold)
    
    img = cv2.imread(image_path)
    is_inverted = False
    
    if darkest_point and brightest_point:
        if darkest_point[0] < brightest_point[0]:
            img = cv2.rotate(img, cv2.ROTATE_180)
            is_inverted = True
            
    return is_inverted, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
