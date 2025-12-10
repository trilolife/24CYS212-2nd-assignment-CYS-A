import numpy as np
from PIL import Image

img = np.array(Image.open("input.jpg").convert("L"))

bit_planes = []
for i in range(8):
    plane = ((img >> i) & 1) * 255
    bit_planes.append(plane.astype(np.uint8))
    Image.fromarray(bit_planes[i]).save(f"bit_plane_{i}.png")

low3 = img & 0b00000111
low3 = (low3 * 255 // 7).astype(np.uint8)
Image.fromarray(low3).save("reconstructed_low3.png")

diff = np.abs(img.astype(int) - low3.astype(int)).astype(np.uint8)
Image.fromarray(diff).save("difference.png")