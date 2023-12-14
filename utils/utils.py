import gzip
import numpy as np
from matplotlib import pyplot as plt
from outliers import smirnov_grubbs as grubbs
from sklearn.preprocessing import OneHotEncoder


def plot_image(pixels: np.array) -> None:
    plt.imshow(pixels.reshape((28, 28)), cmap='gray')
    plt.show()


def clear_anomalies(array: np.ndarray, power: float | int = 2, k: float = 10**(-10)) -> np.ndarray:
    new_array = []
    multiples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    for ind in range(28):
        if ind == 0:
            ind = 1
        new_array.append(array[ind]*multiples[ind]**power)
    clear_indices = grubbs.max_test_indices(new_array, k)
    for ind in clear_indices:
        new_array[ind] = 0.
    return np.array(new_array)


def format_data(
        path_to_labels: str = '../train-labels-idx1-ubyte.gz',
        path_to_images: str = '../train-images-idx3-ubyte.gz',
        image_size: int = 784
) -> dict[str: np.ndarray]:
    with gzip.open(path_to_labels) as train_labels:
        train_data = train_labels.read()[8:]
        labels = [int(data) for data in train_data]
        data_length = len(labels)

    images = []
    with gzip.open(path_to_images) as train_images:
        train_images.read(16)
        for _ in range(data_length):
            img = train_images.read(image_size)
            image = np.frombuffer(img, dtype='uint8') / 255
            images.append(image)
    images = np.array(images)

    encoder = OneHotEncoder(categories='auto')
    np_labels = np.array(labels).reshape((-1, 1))
    np_labels_onehot = encoder.fit_transform(np_labels).toarray()
    data = {
        'images': images,
        'labels': np_labels_onehot,
    }
    return data
