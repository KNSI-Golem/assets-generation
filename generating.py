import gan

latent_dim = 100
output_shape = (48, 48)
generator_to_use = 'generator_model_1000.h5'
output_path = 'results'
output_name = '1'

# zmien na preview_rows=1, preview_cols=1, preview_margin=0, jak chcesz jeden obrazek tylko generowac
preview_rows = 1
preview_cols = 1
preview_margin = 0

g_model = gan.define_generator(latent_dim, output_shape=output_shape)
g_model.load_weights('generator_model_1000.h5')

for i in range(500):
    input_noise = gan.generate_latent_points(latent_dim, preview_rows * preview_cols)
    gan.save_images(output_name, preview_cols, preview_rows, input_noise, output_path=output_path,
                    model=g_model, preview_margin=preview_margin, image_size=output_shape, name='generated/train-'+str(i)+'.png')
