import gan
import utils
import numpy as np
import json

output_path = '../results/gan/gan_v1'
dataset = np.load('../dataset.npy')
with open('gan_config.json') as handle:
    params = json.loads(handle.read())
params['n_epochs'] = 11
shape = (dataset.shape[1], dataset.shape[2], dataset.shape[3])
latent_dim = 100
d_model = gan.define_discriminator(shape)
g_model = gan.define_generator(latent_dim, shape)
gan_model = gan.define_gan(g_model, d_model)
history = utils.train(g_model, d_model, gan_model, dataset, latent_dim, output_path, **params)