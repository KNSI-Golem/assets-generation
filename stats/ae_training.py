from autoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
import image_processing
import image_statistics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

processing_output = './processed/results_processing'
df = image_statistics.get_statistics(processing_output)
min_val = df['Background_percentage'].quantile(0.02)
max_val = df['Background_percentage'].quantile(0.92)
mask1 = df['Background_percentage'] < max_val
mask2 = df['Background_percentage'] > min_val
usable_images = df.loc[mask1 & mask2, 'Path']
data = image_processing.get_data_from_images(usable_images)
#data = data[:, :, :, :-1]

# initialize the number of epochs to train for and batch size
EPOCHS = 25
BS = 32
output_samples = 100

#trainX, testX = train_test_split(data, test_size=0.1)

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
encoder, decoder, autoencoder = ConvAutoencoder.build(32, 48, 4,
                                                      filters=(32, 64),
                                                      latentDim=128)
opt = Adam(lr=1e-3)
#autoencoder.load_weights('autoencoder.h5')
autoencoder.compile(loss="mae", optimizer=opt)
# train the convolutional autoencoder
H = autoencoder.fit(
    data, data,
    #validation_data=(testX, testX),
    epochs=EPOCHS,
    batch_size=BS)

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
#plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('ae_plot.png')

# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
print("[INFO] making predictions...")
decoded = autoencoder.predict(data)
#encoded = encoder.predict(data)
outputs = None
# loop over our number of output samples
for i in range(0, output_samples):
    # grab the original image and reconstructed image
    original = ((data[i] + 1.0) * 127.5).astype("uint8")
    #blurred = ((testX[i] + 1.0) * 127.5).astype("uint8")
    recon = ((decoded[i] + 1.0) * 127.5).astype("uint8")
    # stack the original and reconstructed image side-by-side
    output = np.hstack([original, recon])
    # if the outputs array is empty, initialize it as the current
    # side-by-side image display
    if outputs is None:
        outputs = output
    # otherwise, vertically stack the outputs
    else:
        outputs = np.vstack([outputs, output])
# save the outputs image to disk

image = Image.fromarray(outputs)
image.save('ae_results.png')
image.show()

encoder.save('encoder.h5')
decoder.save('decoder.h5')
autoencoder.save('autoencoder.h5')
encoded_data = encoder.predict(data)
np.save('encoded_training_data.npy', encoded_data)
