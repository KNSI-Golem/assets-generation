from sklearn.cluster import KMeans
import image_processing
import numpy as np
import some_analysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from autoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
import wgan
import gan
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

def generate_images(g_model, generator_path, n_images=1000):
    latent_dim = 100
    g_model.load_weights(generator_path)
    seeds = gan.generate_latent_points(latent_dim, n_images)
    images = g_model.predict(seeds)
        
    return images

def cluster_entropy(original, generated, encoder, kmeans, n_clusters=10):
    generated = encoder.predict(generated)
    n_original_images = original.shape[0]
    data = np.concatenate((original, generated))
    data = (data - data.mean()) / data.std()
    evaluation = 0.0

    #kmeans = KMeans(n_clusters=n_clusters, n_init=15, n_jobs=4)
    preds = kmeans.fit_predict(data)
    
    preds_original = preds[:n_original_images]
    preds_generated = preds[n_original_images:]
    for i in range(n_clusters):
        n_original = (preds_original == i).sum()
        n_generated = (preds_generated == i).sum()
        #n_images = n_original + n_generated
        #evaluation += n_generated * n_generated / n_images
        evaluation += abs(n_generated - n_original)
        
    evaluation /= n_original_images * 2
    #evaluation = abs(evaluation - 0.5) * 2.0
    print(evaluation)
    return evaluation
    

if __name__ == '__main__':
    encoder, decoder, autoencoder = ConvAutoencoder.build(32, 48, 3,
                                                          filters=(32, 64),
                                                          latentDim=128)
    try:
        encoder.load_weights('encoder.h5')
        data_encoded = np.load('encoded_training_data.npy')
    except (ValueError, OSError):
        input_path = './bin'
        output_shape = (32, 48)
        processing_output = './processed/results_processing'
        data = image_processing.get_data_from_images(processing_output)
        data = data[:, :, :, :-1]
        
        autoencoder.compile(loss="mae", optimizer=Adam(lr=1e-3))
        autoencoder.fit(data, data, epochs=100, batch_size=32)
        encoder.save('encoder.h5')
        decoder.save('decoder.h5')
        autoencoder.save('autoencoder.h5')
        data_encoded = encoder.predict(data)
        np.save('encoded_training_data.npy', data_encoded)

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=15, n_jobs=4)
    kmeans.fit(data_encoded)
    
    n_original = data_encoded.shape[0]
    image_dim = (32, 48, 3)
    #base_path = './results/gan_results_new_architecture_32x48_final_3/generator_models/'
    base_path = './results/wgan_withbatchnorm_results_new_architecture_32x48_final_2/generator_models/'
    generators = pd.Series(os.listdir(base_path))
    generators.index = generators.str.extract('(\d+)', expand=False).astype(int)
    generators = generators.sort_index()
    generators = generators[1::10]
    g_model = wgan.define_generator(100, output_shape=(image_dim[0], image_dim[1]))
    
    evaluations = []
    for generator_path in generators:
        print(generator_path)
        generator_path = base_path + generator_path
        gen_data = generate_images(g_model, generator_path, n_images=n_original)
        evaluation = cluster_entropy(data_encoded, gen_data, encoder, kmeans, n_clusters=n_clusters)
        print('='*50)
        evaluations.append(evaluation)
        
    plt.plot(range(10, (len(generators) + 1) * 10, 10), evaluations)
