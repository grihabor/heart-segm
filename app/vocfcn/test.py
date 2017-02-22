import caffe
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

model = 'val.prototxt'
weights = 'fcn16s-heavy-pascal.caffemodel'

#caffe.set_mode_gpu();
#caffe.set_device(0);

net = caffe.Net(model, weights, caffe.TEST)

def load_image(img_dir, idx):
    im = imread('{}/img/{}.png'.format(img_dir, idx))


    in_ = im#np.array(im, dtype=np.float32)
    #in_ = in_[:, :, 2::-1]
    #in_ -= (104.00699, 116.66877, 122.67892)  # see val.prototxt
    #in_ = in_.transpose((2, 0, 1))
    return in_


image = load_image('../../data/sbdd/dataset', 'img_0')


res = net.forward()

score = res['score_output2'].transpose((2, 3, 1, 0))
print('score shape:', score.shape)
score = score[:, :, :, 0]
print(score.shape)

width = 3

plt.subplot(1, width, 1)
plt.imshow(image[:, :, :3])
plt.subplot(1, width, 2)
plt.imshow(score[:, :, 0], cmap='gray')
plt.subplot(1, width, 3)
plt.imshow(score[:, :, 1], cmap='gray')
plt.savefig('output.png')
