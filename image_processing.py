import os
#import imageio
#import cv2
from PIL import Image
import numpy as np
import pandas as pd
import re
import shutil
import sys
import collections
import six
import image_statistics
import gifextract

def is_iterable(arg):
    return (
        isinstance(arg, collections.Iterable) 
        and not isinstance(arg, six.string_types)
    )

def get_images_paths(input_path):
    final_images_paths = []
    
    items = []
    temp_items = []

    if is_iterable(input_path):
        temp_items.extend(input_path)
    else:
        temp_items.append(input_path)
    
    while len(temp_items):
        items = temp_items
        temp_items = []
        
        for item in items:
            extension = item[-4:]
            if extension != '.gif' and extension != '.png':
                # we assume that item is actually a directory
                new_items = os.listdir(item)
                for new_item in new_items:
                    temp_items.append(item + f'/{new_item}')
            else:
                final_images_paths.append(item)
    
    return final_images_paths

def close_all_frames(frames):
    for frame in frames:
        frame.close()

def process_images(input_path, output_path, res_needed=None, resize_source_to=None,
                   resize_output_to=None, change_mode_to=None, use_all_frames=True,
                   crop_images=False, n_boxes_width=1, n_boxes_height=1, 
                   change_background_to=None):
    
    images_paths = get_images_paths(input_path)
    
    if output_path is not None and not os.path.exists(f'{output_path}'):
            os.makedirs(f'{output_path}')
            
    count = 0
    length = len(images_paths)
    for image_path in images_paths:
        
        if not count % 10:
            print(f'Processing progress: {count / length * 100.}%')
        count += 1
        
        if use_all_frames and image_path[-4:] == '.gif':
            frames = gifextract.processImage(image_path)
        else:
            frames = [Image.open(image_path)]
        
        if res_needed is not None and res_needed != frames[0].size:
            close_all_frames(frames)
            continue
        
        image_name = re.search('[^\/]*\.(?=png|gif)', image_path).group()[: -1] # the last character is a dot so we don't want it
        for frame_nr, frame in zip(range(len(frames)), frames):
            
            if change_background_to is not None:
                mode = frame.mode
                if mode != 'RGBA':
                    frame = frame.convert('RGBA')
                
                array = np.array(frame)
                frame.close()
                
                mask = array[:, :, 3] == 0
                array[mask] = change_background_to
                
                frame = Image.fromarray(array, mode='RGBA')
                if change_mode_to is not None and mode != change_mode_to:
                    frame = frame.convert(change_mode_to)
                elif mode != 'RGBA':
                    frame = frame.convert('RGBA')
            elif change_mode_to is not None and change_mode_to != frame.mode:
                frame = frame.convert(change_mode_to)
            
            if resize_source_to is not None and resize_source_to != frame.size:
                frame = resize_image(frame, resize_source_to)
            
            if crop_images:
                imgwidth, imgheight = frame.size
                box_height = imgheight // n_boxes_height
                box_width = imgwidth // n_boxes_width

                for h in range(0, imgheight, box_height):
                    y_pos = h // box_height
                    for w in range(0, imgwidth, box_width):
                        box = (w, h, w+box_width, h+box_height)
                        box = frame.crop(box)
                        
                        if resize_output_to is not None and box.size != resize_output_to:
                            box = resize_image(box, resize_output_to)
                           
                        image_name_full = f'{image_name}_frame{frame_nr}_y{y_pos}_x{w//box_width}.png'                            
                        box.save(f'{output_path}/' + image_name_full)
                        box.close()
            else:
                if resize_output_to is not None and frame.size != resize_output_to:
                    frame = resize_image(frame, resize_output_to)
                image_name_full = f'{image_name}_frame{frame_nr}.png'                            
                frame.save(f'{output_path}/' + image_name_full)
            frame_nr += 1
            frame.close()
            
def resize_image(image, size):
    bands = image.split()
    bands = [b.resize(size) for b in bands]
    image = Image.merge('RGBA', bands)
    return image

def get_data_from_images(input_path):
    
    images_paths = get_images_paths(input_path)
    training_data = []
    
    for image_path in images_paths:
        image = Image.open(image_path)
        training_data.append(np.asarray(image))
        image.close()
        
    width, height = image.size
    bands_nr = len(image.getbands())
    new_shape = (len(training_data), width, height, bands_nr)

    training_data = np.reshape(training_data, new_shape)
    training_data = training_data.astype(np.float32)
    training_data = training_data / 127.5 - 1.

    return training_data

def fetch_images(input_path, output_path, frame=None, y=None, x=None):
    
    images_paths = get_images_paths(input_path)
    
    if not os.path.exists(f'{output_path}'):
            os.makedirs(f'{output_path}')
        
    count = 0
    images_nr = len(images_paths)
    for image_path in images_paths:
        
        if not count % 100:
            print(f'Fetching progress: {count / images_nr * 100.}%')
        count += 1
        
        image_name = re.search('[^\/]*\.(?=png|gif)', image_path).group()[: -1] # the last character is a dot so we don't want it
        
        frame_nr = re.search('(?<=frame)\d', image_name)
        current_x = int(re.search('(?<=_x)\d+', image_name).group())
        current_y = int(re.search('(?<=_y)\d+', image_name).group())
        frame_nr = int(frame_nr.group())

        if (frame is not None and frame_nr != frame) or\
           (y is not None and current_y != y) or\
           (x is not None and current_x != x):
            continue
        
        shutil.copy(image_path, f'{output_path}/{image_name}.png')
