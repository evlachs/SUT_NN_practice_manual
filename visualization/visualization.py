import itertools
import numpy as np
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues) -> None:
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
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_loss_and_accuracy(loss: list, accuracy: list):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Loss and accuracy plots')

    axs[0].plot(loss, 'r-.')
    axs[1].plot(accuracy, 'g-')
    axs[0].set_title('loss')
    axs[1].set_title('accuracy')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epochs')
    plt.show()
