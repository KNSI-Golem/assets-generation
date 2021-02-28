from PIL import Image
import numpy as np
import pandas as pd
import image_processing
import matplotlib.pyplot as plt
from scipy import ndimage
#import wgan as gan
import gan
import os
#import tensorflow_addons
from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

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

def get_colors_heatmap(images):
    n_images = images.shape[0]
    heatmaps = []
    
    for i in range(images.shape[3]):
        images_band = images[:, :, :, i]
        heatmap = images_band.sum(axis=0)
        heatmap = heatmap / n_images / 2.0 + 0.5 # values in range [0, 1]
        heatmap = heatmap * 255. # values in range [0, 255]
        heatmaps.append(heatmap)
        
    #whole = ((heatmaps[0] + heatmaps[1] + heatmaps[2]) / 3.0)
    #heatmaps.append(whole)
    
    return heatmaps

def get_heatmaps_stats(heatmaps):
    averages = []
    deviations = []
    height_centers = []
    weight_centers = []
    medians = []
    
    for heatmap in heatmaps:
        average = np.mean(heatmap)
        deviation = np.std(heatmap)
        mass_center = ndimage.measurements.center_of_mass(heatmap)
        height_center = mass_center[0]
        weight_center = mass_center[1]
        median = ndimage.measurements.median(heatmap)
        
        averages.append(average)
        deviations.append(deviation)
        height_centers.append(height_center)
        weight_centers.append(weight_center)
        medians.append(median)
        
    df_stats = pd.DataFrame({'Average': averages,
                             'Median': medians,
                             'Std': deviations,
                             'MassCenterHeight': height_centers,
                             'MassCenterWidth': weight_centers},
                            index=['red', 'green', 'blue'])
        
    return df_stats

def direction_change_distance(array):
    iterations_height = array.shape[0] - 1
    iterations_width = array.shape[1] - 1
    final_distance = 0
    
    for i in range(iterations_height):
        distance = array[i, :, :] - array[i + 1, :, :]
        distance = np.abs(distance)
        distance = distance.mean()
        final_distance += distance
        
    for i in range(iterations_width):
        distance = array[:, i, :] - array[:, i + 1, :]
        distance = np.abs(distance)
        distance = distance.mean()
        final_distance += distance
        
    final_distance = final_distance / iterations_height / iterations_width
    return final_distance
    
def generate_images(generator_path, g_model, n_images=1000):
    
    latent_dim = 100
    g_model.load_weights(generator_path)
    
    seeds = gan.generate_latent_points(latent_dim, n_images)
    images = g_model.predict(seeds)
    images = images.clip(-1, 1)
        
    return images
        
def calculate_distance(stats_original, stats_gen, heatmaps_original, heatmaps_gen):
    stats_diffs = (stats_original - stats_gen) / stats_original
    #stats_diffs_sum = stats_diffs.loc[['sum'], :]
    
    stats_diffs_others = stats_diffs.loc[['red', 'green', 'blue'], :]
    stats_diffs_others = stats_diffs_others.abs().mean(axis=0)
    
    l1_distance = 0
    l2_distance = 0
    for heatmap_original, heatmap_gen in zip(heatmaps_original, heatmaps_gen):
        temp_dist = heatmap_original.astype(int) - heatmap_gen.astype(int)
        temp_dist = np.abs(temp_dist)
        temp_dist_l1 = temp_dist.sum()
        temp_dist_l2 = np.power(temp_dist, 2).sum() ** 0.5
        l1_distance += temp_dist_l1
        l2_distance += temp_dist_l2
        
    x_dist = stats_original['MassCenterWidth'] - stats_gen['MassCenterWidth']
    y_dist = stats_original['MassCenterHeight'] - stats_gen['MassCenterHeight']
    stats_diffs_others['CenterDistance'] = (x_dist.pow(2) + y_dist.pow(2)).pow(0.5).mean()
    
    stats_diffs_others['L1_Dist'] = l1_distance
    stats_diffs_others['L2_Dist'] = l2_distance
    
    return stats_diffs_others

def heatmap2d(arr, cmap_mode, epoch=False):
    plt.imshow(arr, cmap=cmap_mode)
    if epoch:
        title_obj = plt.title(f'epoch: {epoch}') #get the title property handler
        plt.setp(title_obj, color='r')         #set the color of title to red
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    
    input_path_original = 'for_training/images_for_32x48'
    #images_original = np.load(input_path_original)
    images_original = image_processing.get_data_from_images(input_path_original)
    image_dim = (32, 48, 3)
    heatmaps_original = get_colors_heatmap(images_original)
    stats_original = get_heatmaps_stats(heatmaps_original)
    heatmaps_output = 'heatmaps'
    
    temp_array = np.reshape(heatmaps_original, (image_dim[2], image_dim[1], image_dim[0])).transpose(1, 2, 0)
    original_dist = direction_change_distance(temp_array)
    temp_array = temp_array.clip(0, 255).astype(np.uint8)
        
    image = Image.fromarray(temp_array)
    image.save(f'{heatmaps_output}/original.png')
    image.close()
    
    base_path = './results/gan_results_new_architecture_32x48_final_3/generator_models/'
    generators = pd.Series(os.listdir(base_path))
    generators.index = generators.str.extract('(\d+)', expand=False).astype(int)
    generators = generators.sort_index()
    generators = generators[1:]
    i = 10
    distances = {}
    test_array = np.full((image_dim[1], image_dim[0], image_dim[2]), 0, float)
    dirchange_dists = []
    g_model = gan.define_generator(100, output_shape=(image_dim[0], image_dim[1]))
    for generator in generators:
        print('epoch:', i)
    
        images_gen = generate_images(base_path + generator, g_model)
        heatmaps_gen = get_colors_heatmap(images_gen)
        temp_array = np.reshape(heatmaps_gen, (image_dim[2], image_dim[1], image_dim[0])).transpose(1, 2, 0)
        dirchange_dists.append(direction_change_distance(temp_array))
        temp_array = temp_array.clip(0, 255).astype(np.uint8)
        stats_gen = get_heatmaps_stats(heatmaps_gen)
        
        test_array = test_array + temp_array
        image = Image.fromarray(temp_array)
        image.save(f'{heatmaps_output}/{i}.png')
        image.close()
        
        distance = calculate_distance(stats_original, stats_gen, heatmaps_original, heatmaps_gen)
        distances[i] = distance
        
        del images_gen
        del heatmaps_gen
        del stats_gen
        i += 10
        
    test_array = (test_array / len(generators)).astype(np.uint8)
    image = Image.fromarray(test_array)
    image.save(f'{heatmaps_output}/all_of_them.png')
    image.close()
    df = pd.DataFrame.from_dict(distances, orient='index')
    df['PixelDiff_Score'] = dirchange_dists
    
    for column in df.columns:
        plt.figure()
        df[column].plot(legend=True, label=column)
        
    plt.axhline(y=original_dist, color='orange', linestyle='-', label='original_score')
