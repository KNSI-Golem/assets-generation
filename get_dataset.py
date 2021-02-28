import numpy as np
from image_processing.processing import process_images

base_res = (128, 192)
background = (255.0, 255.0, 255.0) # 127.5 would become 0.0 after normalizing
path = 'bin'
images = process_images(path, base_res=base_res, background=background,
                        row_col_coords=[(0, 0)], normalize=True)
images = np.unique(images, axis=0) # drop duplicates
np.save('dataset.npy', images)