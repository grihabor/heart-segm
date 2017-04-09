from os.path import join
import os.path
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
import numpy as np

DATA_PATH = 'data/sbdd/dataset'
print(os.path.abspath(os.path.curdir))


with open(join(DATA_PATH, 'test.txt')) as f:
    for line in f:
        img_filename = join(DATA_PATH, 'img', '{}.png'.format(line.strip()))
        cls_filename = join(DATA_PATH, 'cls', '{}.npy'.format(line.strip()))

        img = img_as_float(imread(img_filename))
        mask = np.load(cls_filename)



        plt.subplot(121)
        plt.imshow(img)

        plt.subplot(122)
        plt.imshow(mask + img[:,:,0], cmap='gray')
        plt.show()
