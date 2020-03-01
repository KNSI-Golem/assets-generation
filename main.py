import image_processing
import image_statistics
import gan

if __name__ == '__main__':    
    input_path = './bin'
    processing_output = './results_processing'
    fetching_output = './results_fetching'
    images_for_training = './images_for_training'
    gan_results = './gan_results'
    '''
    image_processing.process_images(input_path, processing_output, change_mode_to='RGBA',
                                    resize_output_to=(32, 32), change_background_to=(255, 255, 255, 0),
                                    crop_images=True, n_boxes_width=4, n_boxes_height=4)
    '''
    image_processing.fetch_images(processing_output, fetching_output, frame=0, y=0, x=0)
    
    df = image_statistics.get_statistics(fetching_output)
    
    mask1 = df['Background_percentage'] < 0.70
    mask2 = df['Background_percentage'] > 0.30
    usable_images = df.loc[mask1 & mask2, 'Path']
    
    image_processing.process_images(usable_images, images_for_training, change_mode_to='RGB')
    
    training_data = image_processing.get_data_from_images(images_for_training)
    
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    d_model = gan.define_discriminator()
    # create the generator
    g_model = gan.define_generator(latent_dim)
    # create the gan
    gan_model = gan.define_gan(g_model, d_model)
    # train model
    gan.train(g_model, d_model, gan_model, training_data, latent_dim, gan_results, n_epochs=500)
    