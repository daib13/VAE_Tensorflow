import numpy as np
from mnist import MNIST


def shuffle_data(x, y=None):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx, :]
    if y is None:
        return x
    else:
        y = y[idx]
        return x, y


def load_mnist_data(flag='training'):
    mndata = MNIST('../../../data/MNIST')
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
        elif flag == 'testing':
            images, labels = mndata.load_testing()
        else:
            raise Exception('Flag should be either training or testing.')
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images) / 255
    labels_array = np.array(labels)
    one_hot_labels = np.zeros((labels_array.size, labels_array.max() + 1))
    one_hot_labels[np.arange(labels_array.size), labels_array] = 1
    return images_array, one_hot_labels