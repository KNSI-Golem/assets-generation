from PIL import Image
import numpy as np
import pandas as pd
import image_processing

def get_statistics(input_path):
    
    image_paths = image_processing.get_images_paths(input_path)
    
    non_background_pixels_counts = []
    pixels_counts = []
    image_widths = []
    image_heights = []
    image_background_percentages = []
    
    for image_path in image_paths:
        image = Image.open(image_path)
        arr = np.array(image)
        
        width, height = image.size
        pixels_nr = width * height
        if arr.shape[2] == 4:   
            non_background_pixels_count = (arr[:, :, 3] != 0).sum()
            image_background_percentage = 1.0 - non_background_pixels_count / pixels_nr
            
            non_background_pixels_counts.append(non_background_pixels_count)
            image_background_percentages.append(image_background_percentage)
        pixels_counts.append(pixels_nr)
        image_widths.append(width)
        image_heights.append(height)
        
        image.close()
    if arr.shape[2] == 4:  
        return pd.DataFrame({'Path': image_paths,
                             'Pixels_count': pixels_counts,
                             'Width': image_widths,
                             'Height': image_heights,
                             'Background_percentage': image_background_percentages
                             })
    else:
        return pd.DataFrame({'Path': image_paths,
                             'Pixels_count': pixels_counts,
                             'Width': image_widths,
                             'Height': image_heights,
                             })
        

if __name__ == '__main__':
    input_path = './training_set'
    
    df = get_statistics(input_path)
