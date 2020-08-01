import wget, tarfile
import os
# 网络地址
TRAIN_IMAGES = 'https://zenodo.org/record/1133969/files/train_images.hdf5?download=1'
TRAIN_CRATER = 'https://zenodo.org/record/1133969/files/train_craters.hdf5?download=1'
VAL_IMAGES = 'https://zenodo.org/record/1133969/files/dev_images.hdf5?download=1'
VAL_CRATER = 'https://zenodo.org/record/1133969/files/dev_craters.hdf5?download=1'
TEST_IMAGES = 'https://zenodo.org/record/1133969/files/test_images.hdf5?download=1'
TEST_CRATER = 'https://zenodo.org/record/1133969/files/test_craters.hdf5?download=1'


if __name__ == '__main__':
    wget.download(TRAIN_IMAGES)
    wget.download(TRAIN_CRATER)
    wget.download(VAL_IMAGES)
    wget.download(VAL_CRATER)