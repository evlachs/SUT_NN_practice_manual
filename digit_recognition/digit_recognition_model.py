import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils.utils import format_data
from visualization.visualization import plot_loss_and_accuracy, plot_confusion_matrix

image_size = 28*28

data = format_data(image_size=image_size)
images = data['images']
labels = data['labels']

x_train, x_test, y_train, y_test = train_test_split(images, labels, train_size=55000)

model = keras.Sequential()
model.add(keras.layers.Dense(input_shape=(image_size,), units=128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, batch_size=128)
model.evaluate(x_test, y_test)
model.save('../model')

expected_outputs = np.argmax(y_test, axis=1)
predicted_outputs = np.argmax(model.predict(x_test), axis=1)

class_names = [str(idx) for idx in range(10)]
cnf_matrix = confusion_matrix(expected_outputs, predicted_outputs)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

plot_loss_and_accuracy(history.history['loss'], history.history['accuracy'])
