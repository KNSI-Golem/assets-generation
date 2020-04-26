# example of a dcgan on cifar10
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
#from keras.layers import Conv2D
#from keras.layers import Conv2DTranspose
#from keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras import Input
from tensorflow.keras import Model

import numpy as np

from PIL import Image
import os
 
# define the standalone discriminator model
def define_discriminator(in_shape):
    model = Sequential()
	# normal
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
	# downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
	# downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
	# downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))   
    model.add(LeakyReLU(alpha=0.2))
	# classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
	# compile model
    opt = Adam(lr=0.0004, beta_1=0.5)
    model.compile(loss='mae', optimizer=opt, metrics=['accuracy'])
    return model
 
# define the standalone generator model
def define_generator(latent_dim, output_shape):
    model = Sequential()
    
    start_width = output_shape[0] // 8
    #width_pad = output_shape[0] - start_width * 8
    start_height = output_shape[1] // 8
    #height_pad = output_shape[1] - start_height * 8
    
	# foundation for 4x4 image
    n_nodes = 256 * start_width * start_height
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((start_height, start_width , 256)))
	# upsample to 2*start_width x 2*start_height
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
	# upsample to 4*start_width x 4*start_height
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
	# upsample to 8*start_width x 8*start_height
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

	# output layer
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model

def define_generator_2(latent_dim, output_shape):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(Reshape((1, 1, 256)))
    model.add(Conv2DTranspose(512, (3, 2), kernel_initializer=RandomUniform()))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # 3x3
    model.add(Conv2DTranspose(256, (4, 4), padding='same', strides=(2, 2), kernel_initializer=RandomUniform()))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # 6x6
    model.add(Conv2DTranspose(128, (4, 4), padding='same', strides=(2, 2), kernel_initializer=RandomUniform()))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # 12x12
    model.add(Conv2DTranspose(64, (4, 4), padding='same', strides=(2, 2), kernel_initializer=RandomUniform()))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # 24x24
    # Extra layer
    model.add(Conv2DTranspose(64, (3, 3), padding='same', kernel_initializer=RandomUniform()))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # 24x24
    model.add(Conv2DTranspose(3, (4, 4), padding='same', activation='tanh',
                              strides=(2, 2), kernel_initializer=RandomUniform()))
    # 48x48
    
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0001, beta_1=0.5)
	model.compile(loss='mae', optimizer=opt)
	return model
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1)) * 0.9
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = ones((n_samples, 1)) * 0.1
	return X, y
 
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, output_path, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    seed = generate_latent_points(latent_dim, 54)
	# manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            
        if not (i+1) % 10 or not i:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
            # save images:
            epoch_nr = i + 1
            save_images(epoch_nr, 9, 6, seed, output_path + '/images', g_model,
                        image_size=(X_fake.shape[1], X_fake.shape[2]))
            # save the generator model tile file
            save_generator_path = f'{output_path}/generator_models'
            save_discriminator_path = f'{output_path}/discriminator_models'
            if not os.path.exists(save_generator_path):
                os.makedirs(save_generator_path)
            if not os.path.exists(save_discriminator_path):
                os.makedirs(save_discriminator_path)
            g_model.save(save_generator_path + f'/generator_model_{epoch_nr}.h5')
            d_model.save(save_discriminator_path + f'/discriminator_model_{epoch_nr}.h5')
    
def save_images(epoch, n_cols, n_rows, seed, output_path, model, preview_margin=16, image_size=(32, 32)):
    
    height = image_size[0]
    width = image_size[1]
    
    image_array = np.full((preview_margin + (n_rows * (height+preview_margin)), 
                           preview_margin + (n_cols * (width+preview_margin)), 3), 
                          255, dtype=np.uint8)
                
    generated_images = model.predict(seed)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            r = row * (height+preview_margin) + preview_margin
            c = col * (width+preview_margin) + preview_margin
            
            image_array[r:r+height,c:c+width] = generated_images[image_count] * 255
            image_count += 1
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    filename = os.path.join(output_path,f"train-{epoch}.png")
    im = Image.fromarray(image_array, mode='RGB')
    im.save(filename)
