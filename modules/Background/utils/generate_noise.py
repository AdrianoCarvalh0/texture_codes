import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds Gaussian noise to an image.

    Parameters:
    - image: input image (numpy array).
    - mean: mean of the Gaussian distribution (default: 0).
    - sigma: standard deviation of the Gaussian distribution (default: 25).

    Returns:
    - Image with Gaussian noise.
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    Adds salt and pepper noise to an image.

    Parameters:
    - image: input image (numpy array).
    - salt_prob: probability of salt pixels (white).
    - pepper_prob: probability of pepper pixels (black).

    Returns:
    - Image with salt and pepper noise.
    """
    row, col, ch = image.shape
    noisy = np.copy(image)

    # Adds salt noise
    salt_pixels = np.random.rand(row, col) < salt_prob
    noisy[salt_pixels] = 255

    # Adds pepper noise
    pepper_pixels = np.random.rand(row, col) < pepper_prob
    noisy[pepper_pixels] = 0

    return noisy.astype(np.uint8)

# Loading the example image (replace 'example.jpg' with the path to your image)
original_image = cv2.imread('example.jpg')

# Adding salt and pepper noise to the image
prob_salt = 0.01  # Probability of salt pixels (white)
prob_pepper = 0.01  # Probability of pepper pixels (black)
noisy_image = add_salt_and_pepper_noise(original_image, prob_salt, prob_pepper)

# Displaying the original and noisy images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Salt and Pepper Noise')
plt.axis('off')

plt.show()
