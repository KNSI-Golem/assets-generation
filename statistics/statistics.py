import numpy as np
from PIL import Image
import re
import os
from models import gan, wgan, utils
import pandas as pd
from scipy.signal import convolve2d
import time

def get_models(path, latent_dim=100, output_shape=(32, 48, 4), model_type=None):
    if model_type == 'gan':
        generator = gan.define_generator(latent_dim, output_shape)
    elif model_type == 'wgan':
        generator = wgan.define_generator(latent_dim, output_shape)
    else:
        # try to get model_type from path
        match = re.search('wgan', path)
        if match:
            generator = wgan.define_generator(latent_dim, output_shape)
        else: # we assume it's gan (DCGAN)
            generator = gan.define_generator(latent_dim, output_shape)
    
    if path[-1] != '/':
        path += '/'
    models = os.listdir(path)
    nrs = [int(re.search('\d+(?!$)', model).group()) for model in models]
    models = [path + model for model in models]
    models = np.array(models)
    nrs = np.array(nrs)
    idx = np.argsort(nrs)
    nrs = nrs[idx]
    models = models[idx]
    return generator, models, nrs

def standarize_outputs(images):
    if images.shape[3] == 4:
        images = images[:, :, :, :-1]
    images = (images + 1.0) * 127.5
    images = np.round(images).astype(np.uint8)
    images = np.clip(images, 0, 255)
    return images

def get_most_frequent_color(data):
    lst = [(i,j,k) for i,j,k in zip(data[:,:,:,0].ravel(),
                                    data[:,:,:,1].ravel(),
                                    data[:,:,:,2].ravel())]
    lst = lst[:1000]
    most_frequent_color = max(set(lst), key=lst.count)
    return most_frequent_color

def get_mask(data, most_frequent_color):
    mask = (data[:, :, :, 0] != most_frequent_color[0]) |\
           (data[:, :, :, 1] != most_frequent_color[1]) |\
           (data[:, :, :, 2] != most_frequent_color[2])
           
    lst = []
    for i in range(mask.shape[0]):
        lst.append(np.pad(mask[i], 1, 'constant', constant_values=False))
    mask_padded = np.array(lst)
    filter = np.ones((3, 3))
    for i in range(mask_padded.shape[0]):
        conv_out = convolve2d(mask_padded[i], filter, mode='valid')
        m = conv_out >= 7
        conv_out[m] = True
        conv_out[~m] = False
        conv_out = conv_out.astype(bool)
        mask[i] = mask[i] | conv_out
        mask[i] = mask[i] | conv_out
    return mask

def get_widths(data):
    widths = []
    for arr in data:
        for i in range(data.shape[2]):
            if arr[:, i].any():
                start = i
                break
        for i in reversed(range(data.shape[2])):
            if arr[:, i].any():
                end = i
                break
        widths.append(end - start)
    return np.array(widths)

def get_heights(data):
    heights = []
    for arr in data:
        for i in range(data.shape[1]):
            if arr[i, :].any():
                start = i
                break
        for i in reversed(range(data.shape[1])):
            if arr[i, :].any():
                end = i
                break
        heights.append(end - start)
    return np.array(heights)

def calculate_colors_variety(data, *args, **kwargs):
    stds = np.std(data, axis=(1, 2))
    return np.mean(stds, axis=1)

def calculate_size(*args, mask, **kwargs):
    return mask.sum(axis=(1, 2))

def calculate_height(*args, heights, **kwargs):
    return heights

def calculate_width(*args, widths, **kwargs):
    return widths

def calculate_area(*args, bounding_box_areas, **kwargs):
    return bounding_box_areas

def calculate_brightness(data, *args, mask, **kwargs):
    l = []
    for i in range(len(data)):
        arr = data[i][mask[i]]
        if arr.size:
            l.append(np.mean(arr))
        else:
            l.append(0)
    return np.array(l)

def calculate_redness(data, *args, **kwargs):
    return calculate_brightness(data[:, :, :, 0], *args, **kwargs)

def calculate_greeness(data, *args, **kwargs):
    return calculate_brightness(data[:, :, :, 1], *args, **kwargs)

def calculate_blueness(data, *args, **kwargs):
    return calculate_brightness(data[:, :, :, 2], *args, **kwargs)

def calculate_l1_dist(data, *args, other=None, **kwargs):
    l1_distances = []
    if other is None:
        other = data
        div = len(data) - 1
    else:
        div = len(other)
        
    for i in range(len(data)):
        current = data[i]
        temp_sum = 0
        for j in range(len(other)):
            if (i != j) or (other is not data):
                temp_sum += np.mean(np.abs(current - other[j]))
        l1_distances.append(temp_sum / div)
    return np.array(l1_distances)

def calculate_l2_dist(data, *args, other=None, **kwargs):
    l2_distances = []
    if other is None:
        other = data
        div = len(data) - 1
    else:
        div = len(other)
        
    for i in range(len(data)):
        current = data[i]
        temp_sum = 0
        for j in range(len(other)):
            if (i != j) or (other is not data):
                temp_sum += np.mean((current - other[j]) ** 2)
        l2_distances.append(temp_sum / div)
    return np.array(l2_distances)

def get_stats(real_data, generator, models, n_samples=1000, statistics={},
              distances={}):
    latent_shape = generator.input.shape[1]
    idx = np.random.choice(real_data.shape[0], n_samples, replace=False)
    real_data = real_data[idx]
    most_frequent_color = get_most_frequent_color(real_data)
    
    mask = get_mask(real_data, most_frequent_color)
    heights = get_heights(mask)
    widths = get_widths(mask)
    bounding_box_areas = heights * widths

    args = [real_data]
    kwargs = {'mask': mask,
              'heights': heights,
              'widths': widths,
              'bounding_box_areas': bounding_box_areas}
    real_data_stats = {}
    for stat in statistics:
        start = time.time()
        real_data_stats[stat] = statistics[stat](*args, **kwargs)
        print(stat + ':', time.time() - start)
        
    for model in models:
        print(model)
        generator.load_weights(model)
        latent_vec = utils.generate_latent_points(latent_shape, n_samples)
        images = generator.predict(latent_vec)
        images = standarize_outputs(images)
        
    return real_data_stats

if __name__ == '__main__':
    path = './results/gan/gan_v1/generator_models'
    generator, models, numbers = get_models(path)
    real_data = np.load('dataset.npy')
    #mask = real_data[:, :, :, -1] == 1
    real_data = real_data[:, :, :, :-1] # drop alpha channel
    #real_data[mask, :] = (0.5, 0.5, 0.5)
    statistics = {'color_variety': calculate_colors_variety,
                  'size': calculate_size,
                  'hight': calculate_height,
                  'width': calculate_width,
                  'area': calculate_area,
                  'brightness': calculate_brightness,
                  'redness': calculate_redness,
                  'greeness': calculate_greeness,
                  'blueness': calculate_blueness,
                  'l1': calculate_l1_dist,
                  'l2': calculate_l2_dist}
    stats = get_stats(real_data, generator, models[-1:], statistics=statistics)