import image_processing
import image_statistics
import gan

if __name__ == '__main__':    
    input_path = './bin'
    output_shape = (32, 48)
    processing_output = f'./processed/results_processing_{output_shape[0]}x{output_shape[1]}'
    images_for_training = f'./for_training/images_for_{output_shape[0]}x{output_shape[1]}'
    gan_results = f'./results/gan_results_new_architecture_{output_shape[0]}x{output_shape[1]}_1'
    
    image_processing.process_images(input_path, processing_output, change_mode_to='RGBA',
                                    resize_output_to=output_shape, pad_output=False,
                                    change_background_to=(255, 255, 255, 0),
                                    crop_images=True, n_boxes_rows=4, n_boxes_cols=4,
                                    get_boxes_col=0, get_boxes_row=0)
        
    df = image_statistics.get_statistics(processing_output)
    
    min_val = df['Background_percentage'].quantile(0.08)
    max_val = df['Background_percentage'].quantile(0.92)
    mask1 = df['Background_percentage'] < max_val
    mask2 = df['Background_percentage'] > min_val
    usable_images = df.loc[mask1 & mask2, 'Path']
    
    image_processing.process_images(usable_images, images_for_training, change_mode_to='RGB')
    training_data = image_processing.get_data_from_images(images_for_training)
    
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    d_model = gan.define_discriminator((output_shape[1], output_shape[0], 3))
    #d_model.summary()
    # create the generator
    g_model = gan.define_generator(latent_dim, output_shape=output_shape)

    g_model.summary()
    # create the gan
    gan_model = gan.define_gan(g_model, d_model)
    
    # train model
    gan.train(g_model, d_model, gan_model, training_data, latent_dim, gan_results, n_epochs=50, n_batch=128)
    