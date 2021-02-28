import image_processing
import numpy as np
from PIL import Image
from scipy import ndimage
import warnings

def make_preview(data, output=None, n_cols=100, margin_width=16, margin_height=16, background=0):
    n_images = data.shape[0]
    n_rows = int(np.ceil(n_images / n_cols))
    img_width = data.shape[2]
    img_height = data.shape[1]
    
    preview_arr = np.full((n_rows * img_height + (n_rows + 1) * margin_height,
                           n_cols * img_width + (n_cols + 1) * margin_width,
                           3), background, np.uint8)
    
    for im_num in range(n_images):
        row = int(im_num / n_cols)
        col = im_num % n_cols
        x = col * (img_width + margin_width) + margin_width
        y = row * (img_height + margin_height) + margin_height
        
        preview_arr[y: y + img_height, x: x + img_width, :] = data[im_num, :, :, 0:3]

    if output is not None:
        image = Image.fromarray(preview_arr)
        image.save(output)
        return preview_arr

def apply_aplha_as_mask(data):
    n_imgs = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]
    n_channels = data.shape[3]
    analysis_set = np.reshape(data, (n_imgs, height * width, n_channels)) 
    mask = analysis_set[:, :, 3] == 255.0
    analysis_set = analysis_set[:, :, 0: n_channels-1]
    masked_images = []
    for i in range(n_imgs):
       arr = analysis_set[i, :, :][mask[i]]
       masked_images.append(arr)
    
    return masked_images

def calculate_colors_devs(data):
    colors_devs = []
    for arr in data:
        arr = np.std(arr, axis=0)
        val = np.mean(arr, axis=0)
        if np.isnan(val):
            val = 0.0
        colors_devs.append(val)    
    return colors_devs

def calculate_size(data):
    mask = data[:, :, :, 3] == 255.0
    pixels = np.sum(mask, axis=tuple(range(1, mask.ndim)))
    return pixels

def calculate_colors_num(data):
    colors_num = []
    for i in range(data.shape[0]):
        img = Image.fromarray(data[i].astype(np.uint8))
        colors_num.append(len(img.getcolors()))
    return colors_num

def calculate_brightness(data):
    mean_brightness = []
    for arr in data:
        brightness = arr.mean()
        if np.isnan(brightness):
            brightness = 0.0
        mean_brightness.append(brightness)
    return mean_brightness

def calculate_mass_center(data):
    height_centers = []
    width_centers = []
    mask = data[:, :, :, data.shape[3] - 1]
    mask[mask > 0.0] = 1.0
    
    for arr in mask:
        mass_center = ndimage.measurements.center_of_mass(arr)
        height_centers.append(mass_center[0])
        width_centers.append(mass_center[1])
    return height_centers, width_centers

def calculate_distance(image, other):
    temp_dist = np.abs(image - other)
    dist_l1 = temp_dist.sum()
    dist_l2 = np.power(temp_dist, 2).sum() ** 0.5
    return dist_l1, dist_l2
    
def calculate_distances(data, other):
    distances_l1 = []
    distances_l2 = []
    for arr in data:
        distances = calculate_distance(arr, other)
        distances_l1.append(distances[0])
        distances_l2.append(distances[1])
    return distances_l1, distances_l2

def direction_change_score(data, n_pixels):
    height = data.shape[1]
    width = data.shape[2]
    final_distances = np.full((data.shape[0]), 0, float)
    data = data[:, :, :, :data.shape[3] - 1]
    
    for i in range(height - 1):
        distance = data[:, i, :, :] - data[:, i + 1, :, :]
        distance = np.abs(distance)
        distance = distance.mean(axis=(1, 2))
        final_distances += distance
        
    for i in range(width - 1):
        distance = data[:, :, i, :] - data[:, :, i + 1, :]
        distance = np.abs(distance)
        distance = distance.mean(axis=(1, 2))
        final_distances += distance
    final_distances = final_distances / (height - 1) / (width - 1)
    #final_distances /= n_pixels
    return final_distances

def calculate_width(data, threshold=750.):
    begginings = []
    endings = []
    for img in data:
        beggining = img.shape[1]
        ending = 0
        for c in range(img.shape[1]):
            if beggining == img.shape[1]:
                col = img[:, c, :-1]
                col = col.sum(axis=1)
                is_character = (col < threshold).sum()
                if is_character:
                    beggining = c
            if ending == 0:
                col = img[:, img.shape[1] - 1 - c, :-1]
                col = col.sum(axis=1)
                is_character = (col < threshold).sum()
                if is_character:
                    ending = img.shape[1] - c
        begginings.append(beggining)
        endings.append(ending)
    
    begginings = np.array(begginings)
    endings = np.array(endings)
    return endings - begginings

def calculate_height(data, threshold=750.):
    begginings = []
    endings = []
    for img in data:
        beggining = img.shape[0]
        ending = 0
        for r in range(img.shape[0]):
            if beggining == img.shape[0]:
                row = img[r, :, :-1]
                row = row.sum(axis=1)
                is_character = (row < threshold).sum()
                if is_character:
                    beggining = r
            if ending == 0:
                row = img[img.shape[0] - 1 - r, :, :-1]
                row = row.sum(axis=1)
                is_character = (row < threshold).sum()
                if is_character:
                    ending = img.shape[0] - r
        begginings.append(beggining)
        endings.append(ending)
    
    begginings = np.array(begginings)
    endings = np.array(endings)
    return endings - begginings

def bounding_box_area(data):
    return calculate_height(data) * calculate_width(data)

def sort_by_stat(data, stat):
    idx = np.argsort(stat)
    data = data[idx]
    return data
    
def get_heatmap(data):
    heatmap = data.mean(axis=0)
    return heatmap

if __name__ == '__main__':
    input_path = './bin'
    output_shape = (32, 48)
    processing_output = './processed/results_processing'
    data = image_processing.get_data_from_images(processing_output)
    data += 1.0
    data *= 127.5
    
    masked_images = apply_aplha_as_mask(data)
    data_no_alpha = data[:, :, :, :data.shape[3] - 1]
    heatmap = get_heatmap(data_no_alpha)
    
    b = calculate_size(data)
    colors_devs = calculate_colors_devs(masked_images)
    colors_num = calculate_colors_num(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        brighntess = calculate_brightness(masked_images)
    height_centers, width_centers = calculate_mass_center(data)
    distances_l1, distances_l2 = calculate_distances(data_no_alpha, heatmap)
    dir_change_score = direction_change_score(data, n_pixels)
    widths = calculate_width(data)
    heights = calculate_height(data)
    bb_area = bounding_box_area(data)
    boxness = n_pixels / bb_area
    
    stats = [n_pixels, colors_devs, colors_num, brighntess,
             height_centers, width_centers,
             distances_l1, distances_l2, dir_change_score,
             widths, heights, bb_area, boxness]
    preview_names = ['preview_size.png', 'preview_colors_dev.png',
                     'preview_colors_num.png', 'preview_brightness.png',
                     'preview_height_centers.png', 'preview_width_centers.png',
                     'preview_distances_l1.png', 'preview_distances_l2.png',
                     'preview_dir_change_score.png', 'preview_width.png',
                     'preview_height.png', 'bounding_box_area.png', 'boxness.png']
    base_dir = './previews/stats/'
    for stat, preview_name in zip(stats, preview_names):
        new_data = sort_by_stat(data_no_alpha, stat)
        make_preview(new_data, output=base_dir + preview_name,
                     n_cols=200, margin_width=8, margin_height=8)
