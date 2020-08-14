import h5py, os
from utils.processing import preprocess
import numpy as np
import pandas as pd
# root here is '/High-Resolution-MoonNet'
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def DeepMoon(args) :
    # Load data
    n_train, n_val, n_test = args['num_train'], args['num_val'], args['num_test']
    dataroot = os.path.join(root, 'data', args["dataset"])
    train_images = os.path.join(dataroot,"train_images.hdf5")
    train_craters = os.path.join(dataroot, "train_craters.hdf5")

    val_images = os.path.join(dataroot, "dev_images.hdf5")
    val_craters = os.path.join(dataroot, "dev_craters.hdf5")

    test_images = os.path.join(dataroot, "test_images.hdf5")
    test_craters = os.path.join(dataroot, "test_craters.hdf5")

    print("=>loading {}".format(train_images))
    train = h5py.File(train_images, 'r')
    print("=>loading {}".format(val_images))
    val = h5py.File(val_images, 'r')
    print("=>loading {}".format(test_images))
    test = h5py.File(test_images, 'r')

    Data = {
        'train': [train['input_images'][:n_train].astype('float32'),
                  train['target_masks'][:n_train].astype('float32')],
        'val': [val['input_images'][:n_val].astype('float32'),
                val['target_masks'][:n_val].astype('float32')],
        'test': [test['input_images'][:n_test].astype('float32'),
                test['target_masks'][:n_test].astype('float32')]
    }
    train.close()
    val.close()
    test.close()

    # Rescale, normalize, add extra dim
    print("=>preprocessing data")
    preprocess(Data)

    # Load ground-truth craters
    Craters = {
        'train': pd.HDFStore(train_craters, 'r'),
        'val': pd.HDFStore(val_craters, 'r'),
        'test': pd.HDFStore(test_craters, 'r')
    }
    return  Data, Craters

def Surface_Crack(args):
    # Load data
    n_train, n_val, n_test = args['num_train'], args['num_val'], args['num_test']
    dataroot = os.path.join(root, 'data', args["dataset"])
    data_path = os.path.join(dataroot, "surface_crack.hdf5")
    print("=>loading {}".format(data_path))
    dataset = h5py.File(data_path, 'r')

    # train, val = split_train_val(dataset, ratio=0.15)
    train = dataset
    val = dataset

    Data = {
        'train': [train['input_images'][:n_train].astype('float32'),
                  train['target_masks'][:n_train].astype('float32')],
        'val': [val['input_images'][:n_val].astype('float32'),
                val['target_masks'][:n_val].astype('float32')],

    }
    dataset.close()

    # Rescale, normalize, add extra dim
    # print("=>preprocessing data")
    preprocess(Data)

    return Data


def Assembled(args):
    n_train, n_val, n_test = args['num_train'], args['num_val'], args['num_test']
    dataroot = os.path.join(root, 'data', "assembled_dataset")
    train_path = os.path.join(dataroot, "assembled_train.hdf5")
    val_path = os.path.join(dataroot, "assembled_val.hdf5")
    print("=>loading {}".format(train_path))
    train = h5py.File(train_path, 'r')
    print("=>loading {}".format(val_path))
    val = h5py.File(val_path, 'r')

    Data = None
    if "crack" in args["dataset"]:
        Data = {
            'train': [train['input_images'][:n_train].astype('float32'),
                      train['crack_mask'][:n_train].astype('float32')],
            'val': [val['input_images'][:n_val].astype('float32'),
                    val['crack_mask'][:n_val].astype('float32')],

        }
    elif "crater" in args["dataset"]:
        Data = {
            'train': [train['input_images'][:n_train].astype('float32'),
                      train['crater_mask'][:n_train].astype('float32')],
            'val': [val['input_images'][:n_val].astype('float32'),
                    val['crater_mask'][:n_val].astype('float32')],
        }

    train.close()
    val.close()

    preprocess(Data)
    return Data

def round_down(x):
    x = int(x)
    return round(x, 1 - len(str(x)))

def split_train_val(dataset, ratio = 0.15):
    train, val = {},{}

    v_images, v_labels = [], []
    t_images, t_labels = list(dataset["input_images"]), list(dataset["target_masks"])
    validation_size = round_down(ratio * len(dataset["input_images"]))

    while (len(v_images) < validation_size):
        index = int(np.random.randint(len(t_images), size=1))
        v_images.append(t_images.pop(index))
        v_labels.append(t_labels.pop(index))

    train["input_images"] = np.array(t_images)
    train["target_masks"] = np.array(t_labels)
    val["input_images"] = np.array(v_images)
    val["target_masks"] = np.array(v_labels)

    return train, val
########################
def custom_image_generator(data, target, batch_size=32):
    """Custom image generator that manipulates image/target pairs to prevent
    overfitting in the Convolutional Neural Network.

    Parameters
    ----------
    data : array
        Input images.
    target : array
        Target images.
    batch_size : int, optional
        Batch size for image manipulation.

    Yields
    ------
    Manipulated images and targets.

    """
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i + batch_size].copy(), target[i:i + batch_size].copy()

            # Random color inversion
            # for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
            #     d[j][d[j] > 0.] = 1. - d[j][d[j] > 0.]

            # Horizontal/vertical flips
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])  # left/right
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])  # up/down

            # Random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix, npix + 1, batch_size)  # Horizontal shift
            v = np.random.randint(-npix, npix + 1, batch_size)  # Vertical shift
            r = np.random.randint(0, 4, batch_size)  # 90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
                              mode='constant')[npix + h[j]:L + h[j] + npix,
                       npix + v[j]:W + v[j] + npix, :]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix,
                       npix + v[j]:W + v[j] + npix]
                d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
            yield (d, t)