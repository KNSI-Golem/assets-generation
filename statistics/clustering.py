from sklearn.cluster import KMeans
import image_processing
import numpy as np
import some_analysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from autoencoder import ConvAutoencoder

input_path = './bin'
output_shape = (32, 48)
processing_output = './processed/results_processing'
data = image_processing.get_data_from_images(processing_output)
data = data[:, :, :, :-1]
encoder, _, _ = ConvAutoencoder.build(32, 48, 3,
                                      filters=(32, 64),
                                      latentDim=512)
encoder.load_weights('encoder.h5')
data_encoded = encoder.predict(data)
#data_reshaped = data.reshape((data.shape[0], -1))
n_clusters = 200
# Runs in parallel 4 CPUs

kmeans = KMeans(n_clusters=n_clusters, n_init=15, n_jobs=8)
# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(data_encoded)


data += 1.0
data *= 127.5
array = np.empty((n_clusters), dtype=object)
for i in range(n_clusters):
    array[i] = []
    
for cluster, idx in zip(y_pred_kmeans, range(data.shape[0])):
    array[cluster].append(idx)
    
i = 1
for l in array:
    cluster = data[l]
    some_analysis.make_preview(cluster, f'./previews/cluster_v3_{i}.png', n_cols=5)
    i += 1
    

'''
data_embedded = TSNE(learning_rate=200).fit_transform(data_reshaped)
plt.scatter(data_embedded[:, 0], data_embedded[:, 1])
'''