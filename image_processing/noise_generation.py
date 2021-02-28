import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import noise

def generate_perlin_noise(shape, scale=20.0, octaves=5, persistence=0.5, lacunarity=2.0):
    noise_map = np.zeros(shape)
    base = np.random.randint(0, 100)
    for i in range(shape[0]):
        for j in range(shape[1]):
            noise_map[i][j] = noise.pnoise2(i/scale, 
                                            j/scale, 
                                            octaves=octaves, 
                                            persistence=persistence, 
                                            lacunarity=lacunarity, 
                                            repeatx=1024, 
                                            repeaty=1024, 
                                            base=base)
    noise_map = noise_map - noise_map.min()
    noise_map = noise_map / noise_map.max() * 2
    return noise_map

def get_sample(dataset, noisy_dataset, size=10, axis=1):
    idx = np.random.choice(len(dataset), size=size, replace=False)
    samples = np.concatenate(dataset[idx], axis=axis)
    noisy_samples = np.concatenate(noisy_dataset[idx], axis=axis)
    samples = np.concatenate((samples, noisy_samples), axis=axis-1)
    
    return samples

def generate_white_noise(shape):
    noise = np.random.normal(scale=1.0, size=np.prod(shape))
    noise = np.reshape(noise, shape)
    return noise

def randomize_topology(data, mask, threshold_outfit, threshold_background):
    perlin_noise1 = np.full(data.shape[:-1], 0.0)
    perlin_noise2 = np.full(data.shape[:-1], 0.0)
    for i in range(data.shape[0]):
        perlin_noise1[i] = generate_perlin_noise(data[i].shape[:-1])
        perlin_noise2[i] = generate_perlin_noise(data[i].shape[:-1])
    mask_outfit = perlin_noise1 > np.quantile(perlin_noise1, threshold_outfit)
    mask_threshold = perlin_noise2 > np.quantile(perlin_noise2, threshold_background)
    data[mask_outfit] = 1.0
    mask = (~mask) | mask_outfit
    for i in range(data.shape[0]): 
        arr = data[i]
        transparency_mask = mask[i]
        temp_mask1 = ~transparency_mask & mask_threshold[i]
        temp_mask2 = transparency_mask & mask_threshold[i]
        if not transparency_mask.sum():
            continue
        for channel in range(3):
            mean = np.mean(arr[~transparency_mask, channel]) * 0.1
            std = np.std(arr[~transparency_mask, channel]) * 0.1
            if temp_mask1.sum():
                mean += np.mean(arr[temp_mask1, channel]) * 0.9
                std +=  np.std(arr[temp_mask1, channel]) * 0.9
            val = np.random.normal(loc=mean, scale=std*0.1, size=temp_mask2.sum())
            arr[temp_mask2, channel] = val
        if arr.shape[-1] == 4:
            arr[temp_mask2, -1] = 1
        data[i] = arr
    return data

def blur(data, mask, sigma=0.5, background_factor=0.9):
    arr = data[:, :, :, :-1]
    blurred = gaussian_filter(arr, sigma)
    arr[mask] = blurred[mask]
    arr[~mask] = blurred[~mask] * (1.0 - background_factor) + arr[~mask] * background_factor
    data[:, :, :, :-1] = arr
    return data

def apply_noise(data, mask, brightness_factor=0.1, white_noise_factor=0.05, perlin_noise_factor=0.1):
    noise_shape = (data.shape[0], data.shape[1], data.shape[2], data.shape[3]-1)
    noise = np.full(noise_shape, 0.0)
    brightness_noise = generate_white_noise(data.shape[:-1]) * brightness_factor
    for channel in range(noise_shape[-1]):
        noise[:, :, :, channel] += brightness_noise
    for i in range(noise_shape[0]):
        noise[i] += generate_perlin_noise(noise_shape[1:]) * perlin_noise_factor
    noise += generate_white_noise(noise_shape) * white_noise_factor
    data[mask, :-1] += noise[mask]
    data[~mask, :-1] += noise[~mask] * 0.2
    data = np.clip(data, -1.0, 1.0)
    return data

if __name__ == '__main__':
    dataset = np.load('dataset.npy')
    mask = dataset[:, :, :, 3] == 1
    ranking = mask.sum(axis=(1, 2))
    q1 = np.quantile(ranking, 0.10)
    q2 = np.quantile(ranking, 0.92)
    dataset = dataset[(ranking > q1) & (ranking < q2)]
    
    train_indices = np.random.choice(len(dataset), replace=False, size=int(0.8 * len(dataset)))
    test_indices = [i for i in range(len(dataset)) if i not in train_indices]
    train_dataset = dataset[train_indices].copy()
    test_dataset = dataset[test_indices].copy()
    
    new_shape = (train_dataset.shape[0] * 40, 2, 48, 32, 3)
    new_dataset = np.full((new_shape), 0.0, np.float32)
    for i in range(40):
        sigma = np.random.uniform(low=0.2, high=1.0)
        print(f'{i}: {sigma}')
        noisy_dataset = train_dataset.copy()
        mask = train_dataset[:, :, :, 3] == 1
        noisy_blurry_dataset = blur(noisy_dataset, mask, sigma=sigma)
        '''
        img = get_sample(train_dataset, noisy_blurry_dataset)
        img = (img + 1.0) * 127.5
        img = np.round(img)
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        Image.fromarray(img).show()
        raise
        '''
        '''
        noisy_dataset = randomize_topology(noisy_dataset, mask, 0.92, 0.9)
        mask = noisy_dataset[:, :, :, 3] == 1
        
        noisy_blurry_dataset = noisy_dataset.copy()
        noisy_blurry_dataset = blur(noisy_blurry_dataset, mask, sigma=0.4)
        noisy_blurry_dataset = apply_noise(noisy_blurry_dataset, mask)
        '''
        new_dataset[i*train_dataset.shape[0]: (i+1)*train_dataset.shape[0], 0] = train_dataset[:, :, :, :-1]
        new_dataset[i*train_dataset.shape[0]: (i+1)*train_dataset.shape[0], 1] = noisy_blurry_dataset[:, :, :, :-1]
        
    np.save('train_dataset_for_denoising.npy', new_dataset)
    del new_dataset
    
    new_shape = (test_dataset.shape[0] * 15, 2, 48, 32, 3)
    new_dataset = np.full((new_shape), 0.0, np.float32)
    for i in range(15):
        sigma = np.random.uniform(low=0.2, high=1.0)
        print(f'{i}: {sigma}')
        noisy_dataset = test_dataset.copy()
        mask = test_dataset[:, :, :, 3] == 1
        noisy_blurry_dataset = blur(noisy_dataset, mask, sigma=sigma)
        '''
        noisy_dataset = randomize_topology(noisy_dataset, mask, 0.92, 0.9)
        mask = noisy_dataset[:, :, :, 3] == 1
        
        noisy_blurry_dataset = noisy_dataset.copy()
        noisy_blurry_dataset = blur(noisy_blurry_dataset, mask, sigma=0.4)
        noisy_blurry_dataset = apply_noise(noisy_blurry_dataset, mask)
        '''
        new_dataset[i*test_dataset.shape[0]: (i+1)*test_dataset.shape[0], 0] = test_dataset[:, :, :, :-1]
        new_dataset[i*test_dataset.shape[0]: (i+1)*test_dataset.shape[0], 1] = noisy_blurry_dataset[:, :, :, :-1]
       
    np.save('test_dataset_for_denoising.npy', new_dataset)
    del new_dataset
    '''
    sample = get_sample(dataset, noisy_blurry_dataset)
    sample = denormalize(sample)
    Image.fromarray(sample).show()
    '''
    
    
    