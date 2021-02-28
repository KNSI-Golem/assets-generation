import numpy as np
import pandas as pd
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

def process_images(path, base_res=None, row_col_coords=[(0, 0)], resize_to=None,
                   resample=Image.NEAREST, background=(255.0, 255.0, 255.0),
                   pad_left=0, pad_right=0, pad_top=0, pad_bottom=0):
    paths = get_images_paths(path) # get all images paths
    used_paths = []
    if row_col_coords == 'all':
        row_col_coords = []
        for i in range(4):
            for j in range(4):
                row_col_coords.append((i, j))
    images = []
    i = 0
    for path in paths:
        img = Image.open(path)
        if base_res and img.size != base_res:
            # discard images with wrong resolution
            continue
        img = img.convert('RGBA') # convert to RGBA format
        i += 1
        
        arr = np.array(img)
        # crop image
        height = arr.shape[0] // 4
        width = arr.shape[1] // 4
        positions = []
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
            mask = box[:, :, 3] == 0
            box = box.astype(float)
            for channel in range(box.shape[2] - 1):
                box[mask, channel] = background[channel]
                
            if pad_left or pad_right or pad_top or pad_bottom:
                new_box = np.full((box.shape[0] +  pad_top + pad_bottom,
                                   box.shape[1] + pad_left + pad_right,
                                   4), 0.0)
                for channel in range(new_box.shape[2] - 1):
                    new_box[:, :, channel] = background[channel]
                new_box[pad_top: box.shape[0] + pad_top,
                        pad_left: box.shape[1] + pad_left, :] = box
                box = new_box

            box[box[:, :, 3] == 0.0, 3] = 127.5
            box = box / 127.5 - 1.0 # normalize images
            positions.append(box)
        positions = np.stack(positions, axis=-1)
        positions = positions.reshape(48, 32, positions.shape[-1] * positions.shape[-2])
        images.append(positions)
        used_paths.append(path)
            
    return pd.Series(used_paths), np.array(images)

def make_preview(images, cols, margin=(8, 12), filepath=None):
    n_images = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    rows = np.ceil(n_images / cols) 
    
    preview_height = int(rows * height + (rows - 1) * margin[1])
    preview_width = (cols * width + (cols - 1) * margin[0])
    arr = np.full((preview_height, preview_width, images.shape[3]), 255, dtype=np.uint8)
    
    for i in range(n_images):
        row = i // cols
        col = i % cols
        
        y = row * (height + margin[1])
        x = col * (width + margin[0])
        arr[y: y+height, x: x+width, :] = images[i]
        
    return arr

if __name__ == '__main__':
    base_res = (128, 192)
    background = (255, 255, 255)
    path = 'Data/Skins/'
    paths, images = process_images(path, base_res=base_res, background=background,
                            row_col_coords=((0, 0),))
    images_un, indices = np.unique(images, return_index=True, axis=0) # drop duplicates
    np.save('Data/dataset.npy', images_un)
    paths = paths[indices]
    paths.to_csv('Data/images_paths.csv', index=0)
    