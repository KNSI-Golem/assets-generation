import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from statistics_v2 import get_most_frequent_color, get_mask

def improve_colors(image, f1, f2, stride, threshold=None):
  threshold *= threshold  
  image = np.clip((image+1.0)*127.5,0,255).astype(np.uint8)
  processed_image = np.copy(image)
  y_mid = int(np.median(range(f1)))
  x_mid = int(np.median(range(f2)))
  for y in range(0, image.shape[0]-f1+1, stride):
    for x in range(0, image.shape[1]-f2+1, stride):
      cut = image[y:y+f1, x:x+f2, :]
      mid_color = cut[y_mid, x_mid, :]
      if np.all(cut == mid_color, axis=2).sum() > 1:
          break
      else:
          distances = np.square(cut - mid_color).sum(axis=2)
          distances[y_mid, x_mid] = distances.max()
          idx = np.argmin(distances)
          y_coord = idx // f2
          x_coord = idx % f2
          if not threshold or distances[y_coord, x_coord] < threshold:
              print('OK')
              processed_image[y+y_mid, x+x_mid, :] = cut[y_coord, x_coord]
          else:
              print('not OK')
  return (processed_image/127.5)-1.0

originals = np.load('generated_dataset.npy')
originals = originals[np.random.choice(len(originals), replace=False, size=10)]
improved = []
for i in range(10):
    improved.append(improve_colors(originals[i].copy(), 3, 3, 1, threshold=2))
improved = np.array(improved)
preview = np.concatenate((originals, improved), axis=1)
preview = np.concatenate(preview, axis=1)
preview = (preview + 1.0) * 127.5
preview = np.clip(preview, 0, 255)
preview = np.round(preview)
preview = preview.astype(np.uint8)
Image.fromarray(preview).show()


'''
originals = np.load('generated_dataset.npy')
originals = originals[np.random.choice(len(originals), replace=False, size=10)]
originals = (originals + 1.0) * 127.5
originals = np.clip(originals, 0, 255)
originals = np.round(originals)
originals = originals.astype(np.uint8)
#Image.fromarray(arr[100]).show()


less_colors = originals.copy()
masks = get_mask(less_colors, get_most_frequent_color(less_colors))

flattened = np.reshape(less_colors.copy(), (less_colors.shape[0], less_colors.shape[1] * less_colors.shape[2], -1))
masks = np.reshape(masks.copy(), (less_colors.shape[0], less_colors.shape[1] * less_colors.shape[2]))
n_clusters = 32
for i in range(len(flattened)):
    print(i)
    img = flattened[i]
    current_colors = flattened[i][masks[i]]
    kmeans = KMeans(n_clusters=n_clusters, algorithm='elkan').fit(current_colors)
    labels = kmeans.labels_
    centers = np.round(kmeans.cluster_centers_).astype(np.uint8)
    for n in range(n_clusters):
        colors = current_colors[labels == n]
        current_indices = np.isin(img, colors)
        current_indices = np.all(current_indices, axis=1)
        img[current_indices & masks[i]] = centers[n]
    flattened[i] = img
kmeaned = np.reshape(flattened, (-1, 48, 32, 3))
        
preview = np.concatenate((originals, kmeaned), axis=1)
preview = np.concatenate(preview, axis=1)
Image.fromarray(preview).show()
'''