from PIL import Image
import os
import json

import matplotlib.pyplot as plt
import numpy as np

outfit_color_pairs = {}

graph_list = []

for filename in os.listdir('generated'):
    img = Image.open('generated/' + filename) 
    pixels = img.load() 
    unique_colors = []
    for i in range(img.size[0]): # for every pixel:
        for j in range(img.size[1]):
            pixel = pixels[i,j]
            maximum = max(pixel)
            newcolor=[]
            if maximum == 0:
                newcolor = [0, 0, 0]
            else:
                for color in pixel:
                    newcolor.append(int(round(color/16)))

            if newcolor not in unique_colors:
                unique_colors.append(newcolor)

    outfit_color_pairs[filename] = len(unique_colors)
    graph_list.append(len(unique_colors))

pairs_sorted = {k: v for k, v in sorted(outfit_color_pairs.items(), key=lambda item: item[1])}

plt.hist(graph_list, density=True, bins=50)
plt.show()

with open('colors_in_generated.json', 'w') as f:
    json.dump(pairs_sorted, f)
