import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, gaussian_filter

img = np.array(Image.open("input.jpg").convert("RGB"))

box_5 = uniform_filter(img, size=(5, 5, 1))
box_20 = uniform_filter(img, size=(20, 20, 1))

gauss = gaussian_filter(img, sigma=2)

Image.fromarray(box_5).save("box5.png")
Image.fromarray(box_20).save("box20.png")
Image.fromarray(gauss).save("gaussian.png")