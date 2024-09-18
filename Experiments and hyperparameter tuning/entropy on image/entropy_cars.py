from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

image_path = '/Users/dariadragomir/AI_siemens/Car tracking/Task1/input/01_1.jpg'
colorIm = Image.open(image_path)

greyIm = colorIm.convert('L')
colorIm_np = np.array(colorIm)
greyIm_np = np.array(greyIm)

def entropy(signal):
    lensig = signal.size
    symset = list(set(signal))
    propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]
    ent = np.sum([p * np.log2(1.0 / p) for p in propab])
    return ent

reduced_greyIm_np = np.array(greyIm.resize((greyIm.size[0]//4, greyIm.size[1]//4)))
N = 5
S = reduced_greyIm_np.shape
E = np.array(reduced_greyIm_np)


for row in range(S[0]):
    for col in range(S[1]):
        Lx = np.max([0, col - N])
        Ux = np.min([S[1], col + N])
        Ly = np.max([0, row - N])
        Uy = np.min([S[0], row + N])
        region = reduced_greyIm_np[Ly:Uy, Lx:Ux].flatten()
        E[row, col] = entropy(region)


plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)
plt.imshow(colorIm_np)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(reduced_greyIm_np, cmap=plt.cm.gray)
plt.title('Resized Grayscale Image')

plt.subplot(1, 3, 3)
plt.imshow(E, cmap=plt.cm.jet)
plt.title('Entropy in 10x10 Neighborhood')
plt.colorbar(label='Entropy')

plt.show()
