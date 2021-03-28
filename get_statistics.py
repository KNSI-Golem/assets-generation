import pandas as pd
import numpy as np
from stats import statistics

images = np.load('dataset.npy')
images = statistics.denormalize_img(images)
im_size = (images.shape[1] * images.shape[2])
colors = ['Red', 'Green', 'Blue']
max_value = images.max()
masks = statistics.estimate_character_mask(images)
paths = pd.read_csv('paths.csv', index_col=0).squeeze()
df = pd.DataFrame({'Path': paths})

df['LeftOffset'] = statistics.get_left_offsets(masks)
df['RightOffset'] = statistics.get_right_offsets(masks)
df['TopOffset'] = statistics.get_top_offsets(masks)
df['BottomOffset'] = statistics.get_bottom_offsets(masks)

df['Width'] = masks.shape[2] - df['RightOffset'] - df['LeftOffset']
df.loc[df['RightOffset'] == -1, 'Width'] = 0
df['Height'] = masks.shape[1] - df['TopOffset'] - df['BottomOffset']
df.loc[df['RightOffset'] == -1, 'Height'] = 0

df['BoundingBoxArea'] = df['Width'] * df['Height']
df['CharacterSize'] = statistics.calculate_pixels_number(masks)
df['SizeToImageRatio'] = df['CharacterSize'] / im_size
df['SizeToBoundingBoxRatio'] = df['CharacterSize'] / df['BoundingBoxArea']
df['SizeToBoundingBoxRatio'].fillna(0.0, inplace=True)
df['BoundingBoxNonCharacterPixels'] = df['BoundingBoxArea'] - df['CharacterSize']

variety = statistics.calculate_colors_std(images)
for i in range(variety.shape[1]):
    df['Variety' + colors[i]] = variety[:, i]

df['VerticalDirectionChangeScoreMasked'] = statistics.vertical_direction_change_score(images, masks)
df['HorizontalDirectionChangeScoreMasked'] = statistics.horizontal_direction_change_score(images, masks)

histograms = np.array(statistics.calculate_colors_histograms(images, mask=masks))
for i in range(histograms.shape[0]):
    for j in range(histograms.shape[1]):
        df['HistogramMasked' + colors[i] + f'Bin{j+1}'] = histograms[i, j]

df['BrightnessMasked'] = statistics.calculate_brightness(images, masks)

level_statistics = pd.read_csv('scraping/Data/SkinStatistics.csv', index_col=0)
labels_encodings = pd.read_csv('scraping/Data/SkinLabelsEncodings.csv', index_col=0)
reg_pat = '([\d\w_\-\.]+\/[\d\w_\-\.]+\.\w+$)'
df.index = df['Path'].str.extract(reg_pat, expand=False)
level_statistics.index = level_statistics.index.str.extract(reg_pat, expand=False)
labels_encodings.index = labels_encodings.index.str.extract(reg_pat, expand=False)

df = df.join(level_statistics)
df = df.join(labels_encodings)
df = df.drop(columns='Path')
df.to_csv('statistics.csv')
