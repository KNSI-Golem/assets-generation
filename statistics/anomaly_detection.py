from autoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
import image_processing
import image_statistics
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import some_analysis

processing_output = './processed/results_processing'
data = image_processing.get_data_from_images(processing_output)
mask = data[:, :, :, -1] != -1.0
#data = data[:, :, :, :-1]

_, _, autoencoder = ConvAutoencoder.build(32, 48, 4,
                                          filters=(32, 64),
                                          latentDim=128)
opt = Adam(lr=1e-3)
#autoencoder.load_weights('autoencoder.h5')
autoencoder.load_weights('autoencoder.h5')
predictions = autoencoder.predict(data)
errors = []
shape_errors = []
mask_original_mean = data[:, :, :, -1].mean()
mask_pred_mean = predictions[:, :, :, -1].mean()
stds = (data - predictions).std(axis=0)
for i in range(mask.shape[0]):
    img = data[i][mask[i]]
    pred = predictions[i][mask[i]]
    mahalanobis_distance =  (((img - pred) / stds[mask[i]]) ** 2).sum() ** 0.5
    
    img = data[i][~mask[i]]
    pred = predictions[i][~mask[i]]
    mask_error = (((img - pred) / stds[~mask[i]]) ** 2).sum() ** 0.5
    
    mask_original = mask[i] > mask_original_mean
    mask_pred = predictions[i, :, :, -1] > mask_pred_mean
    mask_sum = mask_original + mask_pred
    mask_mul = mask_original * mask_pred
    shape_error = (mask_sum.sum() - mask_mul.sum()) / mask_original.sum()
    errors.append(mahalanobis_distance + mask_error + shape_error)
    shape_errors.append(shape_error)
    
    #errors.append(mahalanobis_distance)
#data = np.hstack((data, predictions))
data = data - data.min()
data = data / data.max() * 255.0

sorted_data = some_analysis.sort_by_stat(data, shape_errors)
some_analysis.make_preview(sorted_data, output='./previews/a_new_anomalies.png',
                           n_cols=200, margin_width=8, margin_height=8)
