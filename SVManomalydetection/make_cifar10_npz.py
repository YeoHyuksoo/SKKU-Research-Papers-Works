import glob
import os
import pickle

import cv2
import numpy as np

data_dir = './data'

all_files = glob.glob(os.path.join(data_dir, 'data_batch*'))
test_files = glob.glob(os.path.join(data_dir, 'test_batch*'))

all_files = all_files + test_files
assert all_files, 'File paths to load are empty. Please ensure downloaded files are in {}'.format(data_dir)

def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

all_image = []
all_label = []

print(all_files)

for file in all_files:
    ret = unpickle(file)
    for i, arr in enumerate(ret[b'data']):
        img = np.reshape(arr, (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_image.append(img)
        all_label.append(ret[b'labels'][i])

all_images = np.array(all_image)
all_labels = np.array(all_label)

print('All images array shape')
print(all_images.shape)
print('All labeles array shape')
print(all_labels.shape)

np.savez(os.path.join(data_dir, 'cifar10.npz'), images=all_images, labels=all_labels)
