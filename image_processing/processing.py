import numpy as np
from PIL import Image
import os

def get_images_paths(input_path):
    images_paths = []
    if input_path[-1] != '/':
        input_path += '/'
    
    items = os.listdir(input_path)
    for item in items:
        extension = item[-3:]
        if extension in ['gif', 'png', 'jpg']:
            images_paths.append(input_path + item)
        else:
            # we assume that item is a directory
            images_paths.extend(get_images_paths(input_path + item))
    return images_paths

def normalize_img(img):
    return img / 127.5 - 1.0 # normalize images

def denormalize_img(img):
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0.0, 255.0)
    img = np.round(img)
    img = img.astype(np.uint8)
    return img

def change_img_on_background(img, new_background_color):
    background_mask = img[:, :, -1] == 0
    img = img.astype(float)
    for channel in range(img.shape[-1] - 1):
        img[background_mask, channel] = new_background_color[channel]
    return img

def pad_img(img, pad_left=0, pad_right=0, pad_top=0, pad_bottom=0):
    new_img = np.full((img.shape[0] +  pad_top + pad_bottom,
                       img.shape[1] + pad_left + pad_right,
                       4), 0.0)
    for channel in range(new_img.shape[2] - 1):
        new_img[:, :, channel] = background[channel]
        new_img[pad_top: img.shape[0] + pad_top,
                pad_left: img.shape[1] + pad_left, :] = img
    return new_img

def process_images(path, base_res=None, row_col_coords=[(0, 0)], resize_to=None,
                   resample=Image.NEAREST, background=(255.0, 255.0, 255.0),
                   pad_left=0, pad_right=0, pad_top=0, pad_bottom=0,
                   normalize=True):
    paths = get_images_paths(path) # get all images paths
    images = []
    for path in paths:
        img = Image.open(path)
        img = img.convert('RGBA') # convert to RGBA format
        if base_res and img.size != base_res:
            # discard images with wrong resolution
            continue
        
        arr = np.array(img)
        # crop image
        height = arr.shape[0] // 4
        width = arr.shape[1] // 4
        for coords in row_col_coords:
            r = coords[0] * height
            c = coords[1] * width
            box = arr[r: r + height, c: c + width, :]
            
            # resize
            if resize_to:
                img = Image.fromarray(box)
                img = img.resize(resize_to, resample=resample)
                box = np.array(img)
                
            # standarize background
            if background:
                box = change_img_on_background(box, background)
            
            # pad image
            if pad_left or pad_right or pad_top or pad_bottom:
               box = pad_img(img, pad_left, pad_right, pad_top, pad_bottom)

            images.append(box)
            
    images = np.array(images)
    if normalize:
        images = normalize_img(images)
    else:
        images = np.round(images)
        images = images.astype(np.uint8)
        
    return images

def make_preview(images, cols, margin=(8, 12), filepath=None, denormalize=True):
    n_images = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    rows = np.ceil(n_images / cols) 
    
    preview_height = int(rows * height + (rows - 1) * margin[1])
    preview_width = (cols * width + (cols - 1) * margin[0])
    arr = np.full((preview_height, preview_width, images.shape[3]), 255, dtype=np.uint8)
    if denormalize:
        images = denormalize_img(images)
    
    for i in range(n_images):
        row = i // cols
        col = i % cols
        
        y = row * (height + margin[1])
        x = col * (width + margin[0])
        arr[y: y+height, x: x+width, :] = images[i]
        
    return arr

