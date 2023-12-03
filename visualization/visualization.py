import itertools
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from utils.utils import format_data


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    tick_marks = np.arange(len(classes))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


image_size = 784
data = format_data(image_size=image_size)
images = data['images']
np_labels_onehot = data['np_labels_onehot']

x_train, x_test, y_train, y_test = train_test_split(images, np_labels_onehot, train_size=55000)

model = load_model('../digits_recognition_model')
expected_outputs = np.argmax(y_test, axis=1)
predicted_outputs = np.argmax(model.predict(x_test), axis=1)

class_names = [str(idx) for idx in range(10)]
cnf_matrix = confusion_matrix(expected_outputs, predicted_outputs)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

plt.show()
