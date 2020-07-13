import numpy as np
import os
from PIL import Image

# select real samples
def generate_real_samples(dataset, n_samples, multiplier=1.0):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels, -1 for 'real'
    y = np.ones((n_samples, 1)) * multiplier
    return X, y

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, multiplier=1.0):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels with 1.0 for 'fake'
    y = np.ones((n_samples, 1)) * multiplier
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# train the generator and discriminator
def train(genenerator, discriminator, gan_model, dataset, latent_dim, output_path,
          real_samples_multiplier=1.0, fake_samples_multiplier=-1.0, discriminator_batches=1,
          n_epochs=200, n_batch=128):
    batch_per_epoch = dataset.shape[0] // n_batch
    half_batch = n_batch // 2
    seed = generate_latent_points(latent_dim, 54)
    n_steps = batch_per_epoch * n_epochs
	
    history = {'discriminator_real_loss': [],
               'discriminator_fake_loss': [],
               'generator_loss': []}
    for step in range(n_steps):
        epoch = step // batch_per_epoch
        disc_loss_real = 0.0
        disc_loss_fake = 0.0
        for disc_batch in range(discriminator_batches):
            X_real, y_real = generate_real_samples(dataset, half_batch,
                                                   multiplier=real_samples_multiplier)
            loss_real = discriminator.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(genenerator, latent_dim, half_batch,
                                                   multiplier=fake_samples_multiplier)
            loss_fake = discriminator.train_on_batch(X_fake, y_fake)
            disc_loss_real += loss_real
            disc_loss_fake += loss_fake
        disc_loss_real /= discriminator_batches
        disc_loss_fake /= discriminator_batches
        
        X_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1)) * real_samples_multiplier
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        
        history['discriminator_real_loss'].append(disc_loss_real)
        history['discriminator_fake_loss'].append(disc_loss_fake)
        history['generator_loss'].append(g_loss)
            
        if not (step) % batch_per_epoch:
            epoch = step // batch_per_epoch+1
            print('epoch: %d, discriminator_real_loss=%.3f, discriminator_fake_loss=%.3f, generator_loss=%.3f' % (epoch, disc_loss_real, disc_loss_fake, g_loss))
            if not epoch % 10:
                # save images:
                save_images(epoch, 9, 6, seed, output_path + '/images', genenerator,
                            image_size=(X_fake.shape[1], X_fake.shape[2], X_fake.shape[3]))
                # save the generator model tile file
                save_generator_path = f'{output_path}/generator_models'
                save_discriminator_path = f'{output_path}/discriminator_models'
                if not os.path.exists(save_generator_path):
                    os.makedirs(save_generator_path)
                if not os.path.exists(save_discriminator_path):
                    os.makedirs(save_discriminator_path)
                genenerator.save(save_generator_path + f'/generator_model_{epoch}.h5')
                discriminator.save(save_discriminator_path + f'/discriminator_model_{epoch}.h5')
    return history
            
def save_images(epoch, n_cols, n_rows, seed, output_path, model, preview_margin=16, image_size=(32, 32)): 
    height = image_size[0]
    width = image_size[1]
    image_array = np.full((preview_margin + (n_rows * (height+preview_margin)), 
                           preview_margin + (n_cols * (width+preview_margin)), image_size[2]), 
                          255, dtype=np.uint8)
                
    generated_images = model.predict(seed)
    generated_images = (generated_images + 1.0) * 127.5
    generated_images = np.round(generated_images).astype(np.uint8)
    if image_size[2] == 4:
        generated_images[(generated_images[:, :, :, 3] == 127) | (generated_images[:, :, :, 3] == 128)] = 0

    image_count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            r = row * (height+preview_margin) + preview_margin
            c = col * (width+preview_margin) + preview_margin
            image_array[r:r+height,c:c+width] = generated_images[image_count]
            image_count += 1
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    filename = os.path.join(output_path,f"train-{epoch}.png")
    im = Image.fromarray(image_array)
    im.save(filename)
    